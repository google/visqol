"""Public ViSQOL v3.3.3 API (speech and audio modes) and patch aggregation.

Ports ViSQOL v3.3.3 `visqol.cc`, `visqol_manager.cc`, and `visqol_api.cc`.
"""

import dataclasses
import functools
import logging
import os

import jax
import jax.numpy as jnp

from visqol_jax import _check
from visqol_jax import alignment
from visqol_jax import lattice
from visqol_jax import mappers
from visqol_jax import nsim
from visqol_jax import patches
from visqol_jax import selector
from visqol_jax import spectrogram
from visqol_jax import svr

AUDIO_SAMPLE_RATE = 48000
SPEECH_SAMPLE_RATE = 16000
DURATION_MISMATCH_TOLERANCE_S = 1.0
SIMILARITY_EXTREME_THRESHOLD = 0.15
FVNSIM_QUANTILE = 0.10

_LOGGER = logging.getLogger(__name__)
_SvrArrays = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
_LatticeArrays = lattice.LatticeArrays
# pylint: disable-next=protected-access
_as_bool = spectrogram._as_bool
# pylint: disable-next=protected-access
_as_finite_float = spectrogram._as_finite_float
# pylint: disable-next=protected-access
_safe_sqrt = nsim._safe_sqrt
# pylint: disable-next=protected-access
_as_positive_f32 = selector._as_positive_f32
# pylint: disable-next=protected-access
_scalar_int_clamped = selector._scalar_int_clamped
# pylint: disable-next=protected-access
_trunc_i32 = selector._trunc_i32
_BOOL_OPTION_FIELDS = (
    "use_speech_scoring",
    "use_lattice_model",
    "use_unscaled_speech_mos_mapping",
    "skip_mos_mapping",
    "allow_unsupported_sample_rates",
    "disable_global_alignment",
    "disable_realignment",
)


@dataclasses.dataclass(frozen=True)
class VisqolOptions:
  """Options matching C++ `VisqolConfig.VisqolOptions` and `VisqolManager`.

  Attributes:
    use_speech_scoring: Speech mode (21 bands <= 8 kHz, 20-frame VAD patches,
      lattice or exponential mapper) vs audio mode (32 bands <= Nyquist,
      30-frame uniform patches, nu-SVR).
    use_lattice_model: Speech CalibratedLatticeEnsemble (`True`) vs TCD-VOIP
      exponential mapper (`False`).
    use_unscaled_speech_mos_mapping: Omit `float32(1.245063)` scale in speech
      exponential mode.
    search_window_radius: Patch search radius (`search_window = radius *
      patch_size`).
    skip_mos_mapping: Skip MOS mapping and set `result.moslqo` to `NaN`.
    allow_unsupported_sample_rates: Allow sample rates other than 48 kHz in
      audio mode.
    disable_global_alignment: Skip global envelope alignment (lag reported as
      `0.0`).
    disable_realignment: Skip per-patch fine time-domain realignment.
    svr_model_path: Optional custom model path (libsvm `.txt` or `.txt.gz` SVR
      in audio mode, or `.npz` lattice archive in speech lattice mode; ignored
      in speech exponential mode and when `skip_mos_mapping` is `True`).
  """

  use_speech_scoring: bool = False
  use_lattice_model: bool = True
  use_unscaled_speech_mos_mapping: bool = False
  search_window_radius: int = selector.DEFAULT_SEARCH_WINDOW_RADIUS
  skip_mos_mapping: bool = False
  allow_unsupported_sample_rates: bool = False
  disable_global_alignment: bool = False
  disable_realignment: bool = False
  svr_model_path: str | os.PathLike[str] | None = None

  def __post_init__(self) -> None:
    for field_name in _BOOL_OPTION_FIELDS:
      val = _as_bool(getattr(self, field_name), field_name)
      object.__setattr__(self, field_name, val)
    sw = _check.check_nonneg_int(
        self.search_window_radius, "search_window_radius"
    )
    object.__setattr__(self, "search_window_radius", sw)
    if self.svr_model_path is not None:
      path_str = _check.check_nonempty_path(
          self.svr_model_path, "svr_model_path"
      )
      object.__setattr__(self, "svr_model_path", path_str)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class VisqolResult:
  """ViSQOL similarity and quality result, registered as a JAX pytree.

  Float leaves have dtype `float32` and `patch_sims.num_patches` has dtype
  `int32`. `moslqo` is `NaN` when `skip_mos_mapping=True`; score fields are
  `NaN` and `patch_sims` is zeroed (`num_patches == 0`) when either input is
  non-finite (`alignment_lag_s` is also `NaN`) or when a clip is unscorable
  after alignment (zero valid degraded spectrogram frames or zero patches).
  """

  moslqo: jax.Array
  vnsim: jax.Array
  fvnsim: jax.Array
  fvnsim10: jax.Array
  fstdnsim: jax.Array
  fvdegenergy: jax.Array
  alignment_lag_s: jax.Array
  center_freq_bands: jax.Array
  patch_sims: selector.PatchSimilarityResult


def _masked_mean(values: jax.Array, count: jax.Array) -> jax.Array:
  """Mean of active leading prefix `values[:count]` along axis 0."""
  active = jnp.arange(values.shape[0], dtype=jnp.int32) < count
  denom = jnp.maximum(count, jnp.int32(1)).astype(jnp.float32)
  masked = jnp.where(active[:, None], values, jnp.float32(0.0))
  return jnp.sum(masked, axis=0) / denom


def _active_patch_count(
    num_patches: jax.typing.ArrayLike | None, p_max: int
) -> jax.Array:
  """Validate optional `num_patches` and clamp to `[0, p_max]`."""
  if num_patches is None:
    return jnp.int32(p_max)
  return _scalar_int_clamped(num_patches, 0, p_max, "num_patches", nonneg=True)


def calc_per_patch_freq_band_quantile(
    freq_band_means: jax.typing.ArrayLike,
    num_patches: jax.typing.ArrayLike | None = None,
    quantile: float = FVNSIM_QUANTILE,
) -> jax.Array:
  """Mean of lowest `max(1, int(num_patches * quantile))` NSIMs per band."""
  q_val = _as_finite_float(quantile, "quantile")
  if not 0.0 < q_val <= 1.0:
    raise ValueError(f"quantile must be in (0, 1], got {quantile!r}")
  fbm = _check.asarray_f32(freq_band_means, "freq_band_means")
  _check.check_2d_nonempty(fbm, "freq_band_means")

  p_max = fbm.shape[0]
  n_active = _active_patch_count(num_patches, p_max)
  act_2d = (jnp.arange(p_max, dtype=jnp.int32) < n_active)[:, None]
  has_nan = jnp.any(jnp.where(act_2d, jnp.isnan(fbm), False), axis=0)
  padded = jnp.where(act_2d, fbm, jnp.float32(jnp.inf))
  sorted_fbm = jnp.sort(padded, axis=0)

  q_count = jnp.maximum(
      jnp.int32(1),
      _trunc_i32(n_active.astype(jnp.float32) * jnp.float32(q_val)),
  )
  mean_q = _masked_mean(sorted_fbm, jnp.minimum(q_count, n_active))
  return jnp.where(has_nan, jnp.float32(jnp.nan), mean_q)


def calc_per_patch_mean_freq_band_stddevs(
    sim_results: selector.PatchSimilarityResult,
    frame_duration: float | jax.typing.ArrayLike,
) -> jax.Array:
  """Pooled sample stddev per band (`CalcPerPatchMeanFreqBandStdDevs`)."""
  if (
      not isinstance(sim_results, selector.PatchSimilarityResult)
      or len(sim_results) != 9
  ):
    raise ValueError(
        f"sim_results must be a PatchSimilarityResult, got {sim_results!r}"
    )
  fd = _as_positive_f32(frame_duration, "frame_duration")
  fields = tuple(
      _check.asarray_f32(val, f"sim_results.{name}")
      for name, val in zip(
          selector.PatchSimilarityResult._fields[:-1], sim_results[:-1]
      )
  )
  fbm, fstd, ref_s, ref_e = fields[1], fields[2], fields[4], fields[5]
  _check.check_2d_nonempty(fbm, "freq_band_means")
  p_max, n_bands = fbm.shape
  for name, arr in zip(selector.PatchSimilarityResult._fields[:-1], fields):
    want = (p_max, n_bands) if name.startswith("freq_band_") else (p_max,)
    if arr.shape != want:
      raise ValueError(
          f"sim_results.{name} must have shape {want}, got {arr.shape}"
      )
  n_active = _active_patch_count(sim_results.num_patches, p_max)

  active = jnp.arange(p_max, dtype=jnp.int32) < n_active
  act_2d = active[:, None]
  zero = jnp.float32(0.0)
  one = jnp.float32(1.0)
  m_fbm = jnp.where(act_2d, fbm, zero)
  m_fstd = jnp.where(act_2d, fstd, zero)
  p_denom = jnp.maximum(n_active, jnp.int32(1)).astype(jnp.float32)
  fvnsim = jnp.sum(m_fbm, axis=0) / p_denom

  ref_diff = jnp.where(active, ref_e, zero) - jnp.where(active, ref_s, zero)
  fc_f = jnp.where(active, jnp.ceil(ref_diff / fd), zero)
  total_fc = jnp.sum(fc_f)
  mean_fc = total_fc / p_denom

  # Center around fvnsim and mean_fc to avoid float32 sum-of-squares loss.
  dfbm = jnp.where(act_2d, m_fbm - fvnsim[None, :], zero)
  dfc = jnp.where(active, fc_f - mean_fc, zero)
  within = jnp.sum(
      jnp.where(act_2d, (fc_f[:, None] - one) * (m_fstd * m_fstd), zero),
      axis=0,
  )
  between = jnp.sum(fc_f[:, None] * (dfbm * dfbm), axis=0) + (
      jnp.float32(2.0) * fvnsim * jnp.sum(dfc[:, None] * dfbm, axis=0)
  )
  var_denom = jnp.maximum(total_fc - one, one)
  raw_var = jnp.where(
      total_fc > one,
      jnp.maximum((within + between) / var_denom, zero),
      zero,
  )
  return _safe_sqrt(raw_var)


def alter_for_similarity_extremes(
    vnsim: jax.typing.ArrayLike,
    moslqo: jax.typing.ArrayLike,
) -> jax.Array:
  """Clamp MOS to `1.0` for `vnsim < 0.15` (`AlterForSimilarityExtremes`)."""
  v = _check.asarray_f32(vnsim, "vnsim")
  m = _check.asarray_f32(moslqo, "moslqo")
  return jnp.where(
      v < jnp.float32(SIMILARITY_EXTREME_THRESHOLD),
      jnp.float32(mappers.MOS_MIN),
      m,
  )


def _validate_inputs(
    ref: jax.Array,
    deg: jax.Array,
    sample_rate: int,
    options: VisqolOptions,
) -> int:
  """Validate static shapes, sample rate, and minimum signal lengths."""
  fs = _check.check_positive_int(sample_rate, "sample_rate")
  if (
      not options.use_speech_scoring
      and fs != AUDIO_SAMPLE_RATE
      and not options.allow_unsupported_sample_rates
  ):
    raise ValueError(
        f"Currently, {AUDIO_SAMPLE_RATE} Hz is the only sample rate supported"
        f" by ViSQOL Audio (got {fs} Hz). Set"
        " allow_unsupported_sample_rates=True to override."
    )
  min_sr = int(2 * spectrogram.MIN_FREQ)
  if fs < min_sr:
    raise ValueError(
        f"sample_rate ({fs} Hz) must be at least {min_sr} Hz for ERB filterbank"
    )
  signals = (("reference", ref), ("degraded", deg))
  for name, sig in signals:
    _check.check_1d_nonempty(sig, name)

  window_size, hop_size, _ = spectrogram.analysis_window(fs)
  if ref.shape[0] < window_size:
    raise ValueError(
        f"Too few samples ({ref.shape[0]}) in reference signal to build "
        f"spectrogram (must be at least {window_size})."
    )
  if options.disable_global_alignment and deg.shape[0] < window_size:
    raise ValueError(
        f"Too few samples ({deg.shape[0]}) in degraded signal to build "
        f"spectrogram (must be at least {window_size})."
    )

  ref_num_frames = 1 + (ref.shape[0] - window_size) // hop_size
  if options.use_speech_scoring:
    first_patch_idx = patches.SPEECH_PATCH_SIZE // 2 - 1
    min_frames = first_patch_idx + patches.SPEECH_PATCH_SIZE
  else:
    first_patch_idx = 0
    min_frames = patches.AUDIO_PATCH_SIZE + patches.AUDIO_PATCH_SIZE // 2
  if ref_num_frames < min_frames:
    raise ValueError(
        f"Reference spectrum size ({ref_num_frames}) smaller than minimum patch"
        f" size ({min_frames})."
    )
  if options.use_speech_scoring:
    patch_count = (
        ref_num_frames - first_patch_idx
    ) // patches.SPEECH_PATCH_SIZE
    patch_sample_len = int(
        patches.SPEECH_PATCH_SIZE * (window_size * spectrogram.DEFAULT_OVERLAP)
    )
    min_vad_samples = first_patch_idx + patch_count * patch_sample_len
    if ref.shape[0] < min_vad_samples:
      raise ValueError(
          f"Too few samples ({ref.shape[0]}) in reference signal for VAD "
          f"(must be at least {min_vad_samples})."
      )

  ref_dur = ref.shape[0] / float(fs)
  deg_dur = deg.shape[0] / float(fs)
  if abs(ref_dur - deg_dur) > DURATION_MISMATCH_TOLERANCE_S:
    _LOGGER.warning(
        "Mismatch in duration between reference and degraded signal. "
        "Reference is %.2f seconds. Degraded is %.2f seconds.",
        ref_dur,
        deg_dur,
    )
  if options.use_speech_scoring:
    if fs > SPEECH_SAMPLE_RATE:
      _LOGGER.warning(
          "Input audio sample rate (%d Hz) is above 16 kHz, which may have "
          "undesired effects for speech mode. Consider resampling to 16 kHz.",
          fs,
      )
  elif fs != AUDIO_SAMPLE_RATE:
    _LOGGER.warning(
        "Input audio sample rate (%d Hz) does not have the expected sample rate"
        " of 48 kHz; this may negatively affect the MOS-LQO prediction.",
        fs,
    )
  return fs


def _predict_mos(
    vnsim: jax.Array,
    fvnsim: jax.Array,
    fvnsim10: jax.Array,
    fstdnsim: jax.Array,
    fvdegenergy: jax.Array,
    options: VisqolOptions,
    svr_arrays: _SvrArrays | None,
    lattice_arrays: _LatticeArrays | None,
) -> jax.Array:
  """Map per-band similarity features to MOS-LQO according to `options`."""
  if options.skip_mos_mapping:
    return jnp.float32(jnp.nan)
  if not options.use_speech_scoring:
    assert svr_arrays is not None
    raw_mos = svr.eval_svr(fvnsim, *svr_arrays)
  elif options.use_lattice_model:
    assert lattice_arrays is not None
    raw_mos = lattice.eval_lattice(
        fvnsim,
        fvnsim10,
        fstdnsim,
        fvdegenergy,
        jnp.float32(lattice.DEFAULT_LATTICE_TAU),
        *lattice_arrays,
    )
  else:
    raw_mos = mappers.predict_speech_exponential(
        fvnsim, unscaled=options.use_unscaled_speech_mos_mapping
    )
  return alter_for_similarity_extremes(vnsim, raw_mos)


@functools.partial(jax.jit, static_argnames=("sample_rate", "options"))
def _visqol_jit(
    reference: jax.Array,
    degraded: jax.Array,
    svr_arrays: _SvrArrays | None,
    lattice_arrays: _LatticeArrays | None,
    sample_rate: int,
    options: VisqolOptions,
) -> VisqolResult:
  """Static-shape compiled `float32` ViSQOL pipeline."""
  use_speech = options.use_speech_scoring
  patch_size = (
      patches.SPEECH_PATCH_SIZE if use_speech else patches.AUDIO_PATCH_SIZE
  )
  window_size, hop_size, _ = spectrogram.analysis_window(sample_rate)

  if options.disable_global_alignment:
    deg_aligned = degraded
    deg_len = None
    lag_s = jnp.float32(0.0)
  else:
    deg_aligned, deg_len, lag_s = alignment.globally_align(
        reference, degraded, sample_rate
    )

  deg_scaled = spectrogram.scale_to_match_spl(
      reference, deg_aligned, deg_len=deg_len
  )
  # pylint: disable-next=protected-access
  spec_fn = spectrogram._build_gammatone_spectrograms
  (ref_spec, deg_spec), center_freq_bands = spec_fn(
      (reference, deg_scaled), sample_rate, use_speech_mode=use_speech
  )
  deg_valid_cols = (
      None
      if deg_len is None
      else spectrogram.num_spectrogram_frames(deg_len, window_size, hop_size)
  )
  prep_ref, prep_deg = spectrogram.prepare_spectrograms(
      ref_spec, deg_spec, deg_valid_cols=deg_valid_cols
  )

  if use_speech:
    ref_patch_indices, num_ref_patches = patches.create_vad_patch_indices(
        prep_ref.shape[1], reference, sample_rate, patch_size=patch_size
    )
  else:
    ref_patch_indices = patches.create_uniform_patch_indices(
        prep_ref.shape[1], patch_size=patch_size
    )
    num_ref_patches = None
  ref_patches = patches.create_patches_from_indices(
      prep_ref, ref_patch_indices, patch_size
  )

  # C++ CalcFrameDuration(size_t frame_size, size_t sample_rate) truncates hop.
  frame_duration = hop_size / float(sample_rate)
  sim_results = selector.find_most_optimal_deg_patches(
      ref_patches,
      ref_patch_indices,
      prep_deg,
      frame_duration,
      options.search_window_radius,
      num_ref_patches=num_ref_patches,
      deg_valid_cols=deg_valid_cols,
  )
  if not options.disable_realignment:
    sim_results = selector.finely_align_and_recreate_patches(
        sim_results,
        reference,
        deg_scaled,
        sample_rate,
        use_speech_mode=use_speech,
        patch_size=patch_size,
        deg_len=deg_len,
    )

  fvnsim = _masked_mean(sim_results.freq_band_means, sim_results.num_patches)
  fvnsim10 = calc_per_patch_freq_band_quantile(
      sim_results.freq_band_means,
      sim_results.num_patches,
      quantile=FVNSIM_QUANTILE,
  )
  fstdnsim = calc_per_patch_mean_freq_band_stddevs(sim_results, frame_duration)
  fvdegenergy = _masked_mean(
      sim_results.freq_band_deg_energy, sim_results.num_patches
  )
  vnsim = jnp.mean(fvnsim)
  moslqo = _predict_mos(
      vnsim,
      fvnsim,
      fvnsim10,
      fstdnsim,
      fvdegenergy,
      options,
      svr_arrays,
      lattice_arrays,
  )

  # Return NaN when inputs are non-finite or alignment leaves zero patches.
  non_finite = (~jnp.all(jnp.isfinite(reference))) | (
      ~jnp.all(jnp.isfinite(degraded))
  )
  unscorable = non_finite | (sim_results.num_patches == jnp.int32(0))
  if deg_valid_cols is not None:
    unscorable = unscorable | (deg_valid_cols == jnp.int32(0))
  nan = jnp.float32(jnp.nan)
  nan_bands = jnp.full_like(fvnsim, nan)
  zero = jnp.float32(0.0)
  patch_sims = selector.PatchSimilarityResult(
      similarity=jnp.where(unscorable, zero, sim_results.similarity),
      freq_band_means=jnp.where(unscorable, zero, sim_results.freq_band_means),
      freq_band_stddevs=jnp.where(
          unscorable, zero, sim_results.freq_band_stddevs
      ),
      freq_band_deg_energy=jnp.where(
          unscorable, zero, sim_results.freq_band_deg_energy
      ),
      ref_patch_start_time=jnp.where(
          unscorable, zero, sim_results.ref_patch_start_time
      ),
      ref_patch_end_time=jnp.where(
          unscorable, zero, sim_results.ref_patch_end_time
      ),
      deg_patch_start_time=jnp.where(
          unscorable, zero, sim_results.deg_patch_start_time
      ),
      deg_patch_end_time=jnp.where(
          unscorable, zero, sim_results.deg_patch_end_time
      ),
      num_patches=jnp.where(unscorable, jnp.int32(0), sim_results.num_patches),
  )
  return VisqolResult(
      moslqo=jnp.where(unscorable, nan, moslqo),
      vnsim=jnp.where(unscorable, nan, vnsim),
      fvnsim=jnp.where(unscorable, nan_bands, fvnsim),
      fvnsim10=jnp.where(unscorable, nan_bands, fvnsim10),
      fstdnsim=jnp.where(unscorable, nan_bands, fstdnsim),
      fvdegenergy=jnp.where(unscorable, nan_bands, fvdegenergy),
      alignment_lag_s=jnp.where(non_finite, nan, lag_s),
      center_freq_bands=center_freq_bands,
      patch_sims=patch_sims,
  )


def visqol(
    reference: jax.typing.ArrayLike,
    degraded: jax.typing.ArrayLike,
    sample_rate: int,
    options: VisqolOptions | None = None,
) -> VisqolResult:
  """Compute ViSQOL v3.3.3 similarity and MOS-LQO between two 1D mono waveforms.

  Coerces inputs to `float32` and runs in `float32`; safe inside outer
  `jax.jit` and `jax.vmap`.

  Args:
    reference: 1D reference mono waveform.
    degraded: 1D degraded mono waveform.
    sample_rate: Sample rate in Hz (positive integer).
    options: Optional `VisqolOptions` (defaults to 48 kHz audio mode).

  Returns:
    `VisqolResult` pytree containing MOS-LQO, similarity vectors, lag, and
    per-patch results.
  """
  if options is None:
    opts = VisqolOptions()
  elif isinstance(options, VisqolOptions):
    opts = options
  else:
    raise ValueError(
        f"options must be a VisqolOptions or None, got {options!r}"
    )
  ref = _check.asarray_f32(reference, "reference")
  deg = _check.asarray_f32(degraded, "degraded")
  fs = _validate_inputs(ref, deg, sample_rate, opts)

  # Load mapper models outside _visqol_jit so file edits avoid stale constants.
  svr_arrays: _SvrArrays | None = None
  lattice_arrays: _LatticeArrays | None = None
  if not opts.skip_mos_mapping:
    if not opts.use_speech_scoring:
      svr_model = svr.load_svr_model(
          opts.svr_model_path, num_features=spectrogram.AUDIO_MODE_NUM_BANDS
      )
      svr_arrays = (
          svr_model.support_vectors,
          svr_model.sv_coef,
          jnp.float32(svr_model.gamma),
          jnp.float32(svr_model.rho),
      )
    elif opts.use_lattice_model:
      lat = lattice.load_lattice_model(
          opts.svr_model_path, num_bands=spectrogram.SPEECH_MODE_NUM_BANDS
      )
      lattice_arrays = lattice.lattice_arrays(lat)
  static_opts = (
      opts
      if opts.svr_model_path is None
      else dataclasses.replace(opts, svr_model_path=None)
  )
  return _visqol_jit(ref, deg, svr_arrays, lattice_arrays, fs, static_opts)


def visqol_speech(
    reference: jax.typing.ArrayLike,
    degraded: jax.typing.ArrayLike,
    sample_rate: int = SPEECH_SAMPLE_RATE,
    *,
    use_lattice_model: bool = True,
    use_unscaled_speech_mos_mapping: bool = False,
    search_window_radius: int = selector.DEFAULT_SEARCH_WINDOW_RADIUS,
    skip_mos_mapping: bool = False,
    disable_global_alignment: bool = False,
    disable_realignment: bool = False,
    svr_model_path: str | os.PathLike[str] | None = None,
) -> VisqolResult:
  """ViSQOL v3.3.3 speech mode (21 ERB bands <= 8 kHz, RMS-VAD)."""
  return visqol(
      reference,
      degraded,
      sample_rate,
      VisqolOptions(
          use_speech_scoring=True,
          use_lattice_model=use_lattice_model,
          use_unscaled_speech_mos_mapping=use_unscaled_speech_mos_mapping,
          search_window_radius=search_window_radius,
          skip_mos_mapping=skip_mos_mapping,
          disable_global_alignment=disable_global_alignment,
          disable_realignment=disable_realignment,
          svr_model_path=svr_model_path,
      ),
  )


def visqol_audio(
    reference: jax.typing.ArrayLike,
    degraded: jax.typing.ArrayLike,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    *,
    allow_unsupported_sample_rates: bool = False,
    search_window_radius: int = selector.DEFAULT_SEARCH_WINDOW_RADIUS,
    skip_mos_mapping: bool = False,
    disable_global_alignment: bool = False,
    disable_realignment: bool = False,
    svr_model_path: str | os.PathLike[str] | None = None,
) -> VisqolResult:
  """ViSQOL v3.3.3 audio mode (32 ERB bands <= Nyquist, nu-SVR)."""
  return visqol(
      reference,
      degraded,
      sample_rate,
      VisqolOptions(
          use_speech_scoring=False,
          allow_unsupported_sample_rates=allow_unsupported_sample_rates,
          search_window_radius=search_window_radius,
          skip_mos_mapping=skip_mos_mapping,
          disable_global_alignment=disable_global_alignment,
          disable_realignment=disable_realignment,
          svr_model_path=svr_model_path,
      ),
  )
