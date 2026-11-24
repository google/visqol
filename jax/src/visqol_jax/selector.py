"""Coarse dynamic-programming patch matching and fine time-domain realignment.

Ports ViSQOL v3.3.3 `comparison_patches_selector.cc`, preserving DP
tie-breaking, packet-loss skip semantics, `Slice` `end_index` / prepended
postsilence quirks, and static-shape `jax.jit` execution.
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from visqol_jax import _check
from visqol_jax import alignment
from visqol_jax import nsim
from visqol_jax import patches
from visqol_jax import spectrogram

DEFAULT_SEARCH_WINDOW_RADIUS = 60
_INT32_MAX = 0x7FFFFFFF
_MAX_I32_F32 = float(0x7FFFFF80)
_MIN_I32_F32 = -float(1 << 31)
# pylint: disable-next=protected-access
_valid_len = alignment._valid_len
# pylint: disable-next=protected-access
_as_bool = spectrogram._as_bool


class PatchSimilarityResult(NamedTuple):
  """Matched reference/degraded patch results in static `(P, ...)` layout.

  Active entries occupy `[:num_patches]`; slots `[num_patches:]` are zero.
  """

  similarity: jax.Array
  freq_band_means: jax.Array
  freq_band_stddevs: jax.Array
  freq_band_deg_energy: jax.Array
  ref_patch_start_time: jax.Array
  ref_patch_end_time: jax.Array
  deg_patch_start_time: jax.Array
  deg_patch_end_time: jax.Array
  num_patches: jax.Array


def _scalar_int_clamped(
    value: jax.typing.ArrayLike,
    lo_bound: int,
    hi_bound: int,
    name: str,
    *,
    nonneg: bool = False,
) -> jax.Array:
  """Validate scalar integer `value` and clamp to `[lo_bound, hi_bound]`."""
  if _check.is_bool(value):
    raise ValueError(f"{name} must be an integer scalar, got {value!r}")
  if isinstance(value, int):
    if nonneg and value < 0:
      raise ValueError(f"{name} must be non-negative, got {value!r}")
    return jnp.int32(min(max(value, lo_bound), hi_bound))
  try:
    arr = jnp.asarray(value)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError(
        f"{name} must be an integer scalar, got {value!r}"
    ) from exc
  if arr.ndim != 0 or not jnp.issubdtype(arr.dtype, jnp.integer):
    raise ValueError(f"{name} must be an integer scalar, got {value!r}")
  if nonneg and not isinstance(arr, jax.core.Tracer) and int(arr) < 0:
    raise ValueError(f"{name} must be non-negative, got {value!r}")
  if arr.dtype.itemsize < 4:
    arr = arr.astype(jnp.int32)
  if jnp.issubdtype(arr.dtype, jnp.unsignedinteger):
    eff_lo = max(lo_bound, 0)
    eff_hi = min(hi_bound, int(jnp.iinfo(arr.dtype).max))
  else:
    eff_lo, eff_hi = lo_bound, hi_bound
  lo = jnp.asarray(eff_lo, dtype=arr.dtype)
  hi = jnp.asarray(eff_hi, dtype=arr.dtype)
  return jnp.clip(arr, lo, hi).astype(jnp.int32)


def _check_ref_indices(
    ref_patch_indices: jax.typing.ArrayLike,
    *,
    expected_len: int | None = None,
) -> jax.Array:
  """Validate 1D non-negative integer `ref_patch_indices`."""
  if _check.is_bool(ref_patch_indices):
    raise ValueError("ref_patch_indices must be 1D, got bool")
  if isinstance(ref_patch_indices, (list, tuple)):
    if not ref_patch_indices:
      raw = jnp.zeros((0,), dtype=jnp.int32)
    else:
      for item in ref_patch_indices:
        if _check.is_bool(item) or getattr(item, "dtype", None) == jnp.bool_:
          raise ValueError(
              "ref_patch_indices must have an integer dtype, got bool"
          )
        if isinstance(item, int) and (item < 0 or item > _INT32_MAX):
          raise ValueError(
              "ref_patch_indices must be non-negative int32, got"
              f" {ref_patch_indices}"
          )
      try:
        raw = jnp.asarray(ref_patch_indices)
      except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"ref_patch_indices must be a 1D integer array, got"
            f" {ref_patch_indices!r}"
        ) from exc
  else:
    try:
      raw = jnp.asarray(ref_patch_indices)
    except (TypeError, ValueError, OverflowError) as exc:
      raise ValueError(
          f"ref_patch_indices must be a 1D integer array, got"
          f" {ref_patch_indices!r}"
      ) from exc
  if raw.ndim != 1:
    raise ValueError(f"ref_patch_indices must be 1D, got shape {raw.shape}")
  if expected_len is not None and raw.shape[0] != expected_len:
    raise ValueError(
        f"ref_patch_indices must have shape ({expected_len},), got {raw.shape}"
    )
  if not jnp.issubdtype(raw.dtype, jnp.integer):
    raise ValueError(
        f"ref_patch_indices must have an integer dtype, got {raw.dtype}"
    )
  if not isinstance(raw, jax.core.Tracer) and raw.shape[0] > 0:
    lo_i = int(jnp.min(raw))
    hi_i = int(jnp.max(raw))
    if lo_i < 0 or hi_i > _INT32_MAX:
      raise ValueError(
          f"ref_patch_indices must be non-negative int32, got {raw}"
      )
  if raw.dtype == jnp.int32 or raw.dtype.itemsize < 4:
    return jnp.maximum(raw.astype(jnp.int32), jnp.int32(0))
  lo = jnp.zeros((), dtype=raw.dtype)
  hi = jnp.asarray(_INT32_MAX, dtype=raw.dtype)
  return jnp.clip(raw, lo, hi).astype(jnp.int32)


def _as_finite_f32(
    value: float | jax.typing.ArrayLike, name: str
) -> jax.Array:
  """Validate finite numeric scalar `value` and convert to `float32`."""
  arr = _check.asarray_f32(value, name)
  if arr.ndim != 0:
    raise ValueError(f"{name} must be a finite scalar, got {value!r}")
  if not isinstance(arr, jax.core.Tracer) and not math.isfinite(float(arr)):
    raise ValueError(f"{name} must be a finite scalar, got {value!r}")
  return arr


def _as_positive_f32(
    value: float | jax.typing.ArrayLike, name: str
) -> jax.Array:
  """Validate finite positive scalar `value` and convert to `float32`."""
  arr = _as_finite_f32(value, name)
  if not isinstance(arr, jax.core.Tracer) and float(arr) <= 0.0:
    raise ValueError(f"{name} must be positive, got {value!r}")
  return arr


def _trunc_i32(x: jax.Array) -> jax.Array:
  return jnp.clip(
      jnp.trunc(x), jnp.float32(_MIN_I32_F32), jnp.float32(_MAX_I32_F32)
  ).astype(jnp.int32)


def calc_max_num_patches(
    ref_patch_indices: jax.typing.ArrayLike,
    deg_num_frames: jax.typing.ArrayLike,
    patch_size: int,
    *,
    num_ref_patches: jax.typing.ArrayLike | None = None,
) -> jax.Array:
  """Count patches supported by `deg_num_frames` (`CalcMaxNumPatches`)."""
  ps = _check.check_positive_int(patch_size, "patch_size")
  ref_idx = _check_ref_indices(ref_patch_indices)
  n_deg = _scalar_int_clamped(
      deg_num_frames, 0, _INT32_MAX, "deg_num_frames", nonneg=True
  )
  p_max = ref_idx.shape[0]
  n_active = (
      jnp.int32(p_max)
      if num_ref_patches is None
      else _valid_len(num_ref_patches, p_max, "num_ref_patches")
  )
  if p_max == 0:
    return jnp.int32(0)
  half = jnp.int32(min(ps // 2, _INT32_MAX))
  pos = jnp.arange(p_max, dtype=jnp.int32)
  ok = (pos < n_active) & ((ref_idx - half) <= n_deg)
  return jnp.max(jnp.where(ok, pos + jnp.int32(1), jnp.int32(0)))


def build_degraded_patch(
    deg_spectrogram: jax.typing.ArrayLike,
    window_beginning: int | jax.typing.ArrayLike,
    window_width: int,
    *,
    deg_valid_cols: jax.typing.ArrayLike | None = None,
) -> jax.Array:
  """Extract `(num_bands, window_width)` zero-padded outside valid bounds."""
  spec = _check.asarray_f32(deg_spectrogram, "deg_spectrogram")
  _check.check_2d_nonempty(spec, "deg_spectrogram")
  ww = _check.check_positive_int(window_width, "window_width")
  if ww > _INT32_MAX:
    raise ValueError(f"window_width ({ww}) exceeds int32 limit ({_INT32_MAX})")
  t_buf = spec.shape[1]
  n_deg = (
      jnp.int32(t_buf)
      if deg_valid_cols is None
      else _valid_len(deg_valid_cols, t_buf, "deg_valid_cols")
  )
  start = _scalar_int_clamped(
      window_beginning, -ww, t_buf, "window_beginning"
  )
  cols = start + jnp.arange(ww, dtype=jnp.int32)
  in_bounds = (cols >= jnp.int32(0)) & (cols < n_deg)
  safe_cols = jnp.clip(cols, jnp.int32(0), jnp.int32(t_buf - 1))
  return jnp.where(in_bounds[None, :], spec[:, safe_cols], jnp.float32(0.0))


def _prefix_max_right(
    prev_dp: jax.Array, lower_limit: jax.Array, lowest: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """For each `s`, max and largest argmax of `prev_dp[lower_limit : s]`."""
  t_deg = prev_dp.shape[0]
  s = jnp.arange(t_deg, dtype=jnp.int32)
  has = s >= lower_limit
  vals = jnp.where(has, prev_dp, lowest)
  idxs = jnp.where(has, s, jnp.int32(-1))

  def _combine(
      left: tuple[jax.Array, jax.Array, jax.Array],
      right: tuple[jax.Array, jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    h1, v1, i1 = left
    h2, v2, i2 = right
    take_right = h2 & ((~h1) | (v2 >= v1))
    return (
        h1 | h2,
        jnp.where(take_right, v2, v1),
        jnp.where(take_right, i2, i1),
    )

  _, pref_val, pref_idx = jax.lax.associative_scan(
      _combine, (has, vals, idxs), axis=0
  )
  highest_sim = jnp.concatenate([lowest[None], pref_val[:-1]])
  past_offset = jnp.concatenate(
      [jnp.asarray([-1], dtype=jnp.int32), pref_idx[:-1]]
  )
  return highest_sim, past_offset


def _dp_align_patches(
    ref_p: jax.Array,
    ref_idx: jax.Array,
    deg_patches: jax.Array,
    num_patches: jax.Array,
    n_deg: jax.Array,
    search_window_radius: int,
) -> tuple[jax.Array, jax.Array]:
  """Run monotonic DP and backtrace to `(matched_offsets, is_pkt_loss)`."""
  p_max, _, patch_size = ref_p.shape
  t_deg = deg_patches.shape[0]
  zero = jnp.float32(0.0)
  lowest = jnp.float32(-jnp.finfo(jnp.float32).max)
  s_all = jnp.arange(t_deg, dtype=jnp.int32)

  ref_stats = nsim.precompute_patch_stats(ref_p)
  deg_stats = nsim.precompute_patch_stats(deg_patches)

  sw = min(search_window_radius * patch_size, _INT32_MAX)
  sw_i32 = jnp.int32(sw)
  max_center = jnp.int32(_INT32_MAX - sw)
  win_len = min(t_deg, 2 * sw + 1)
  full_scan = win_len == t_deg
  last_index = jnp.maximum(num_patches - jnp.int32(1), jnp.int32(0))

  def _in_search_window(center: jax.Array) -> jax.Array:
    lo = jnp.maximum(center, sw_i32) - sw_i32
    hi = jnp.minimum(center, max_center) + sw_i32
    return (s_all >= lo) & (s_all <= hi) & (s_all < n_deg)

  def _dp_step(
      carry: tuple[jax.Array, jax.Array],
      inputs: tuple[jax.Array, nsim.PatchNsimStats, jax.Array, jax.Array],
  ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    prev_dp, last_dp = carry
    p, r_stat, r_idx, prev_r_idx = inputs
    r_stat_b = jax.tree_util.tree_map(lambda a: a[None, :, :], r_stat)

    if full_scan:
      sim_map = nsim.sim_map_from_stats(r_stat_b, deg_stats)
      sim_all = jnp.mean(jnp.mean(sim_map, axis=-1), axis=-1)
    else:
      win_start = jnp.minimum(
          jnp.maximum(r_idx, sw_i32) - sw_i32, jnp.int32(t_deg - win_len)
      )
      d_stat_win = jax.tree_util.tree_map(
          lambda a: jax.lax.dynamic_slice_in_dim(a, win_start, win_len, axis=0),
          deg_stats,
      )
      sim_map = nsim.sim_map_from_stats(r_stat_b, d_stat_win)
      sim_win = jnp.mean(jnp.mean(sim_map, axis=-1), axis=-1)
      sim_all = jax.lax.dynamic_update_slice(
          jnp.zeros((t_deg,), dtype=jnp.float32), sim_win, (win_start,)
      )

    lower_limit = jnp.maximum(prev_r_idx, sw_i32) - sw_i32
    highest_sim, past_offset = _prefix_max_right(prev_dp, lower_limit, lowest)
    cand = sim_all + highest_sim
    skip = prev_dp > cand
    dp_gt0 = jnp.where(skip, prev_dp, cand)
    bt_gt0 = jnp.where(skip, s_all, past_offset)

    is_first = p == jnp.int32(0)
    dp_cand = jnp.where(is_first, sim_all, dp_gt0)
    bt_cand = jnp.where(is_first, jnp.int32(-1), bt_gt0)

    write = (p < num_patches) & _in_search_window(r_idx)
    cur_dp = jnp.where(write, dp_cand, zero)
    cur_bt = jnp.where(write, bt_cand, jnp.int32(0))
    new_last_dp = jnp.where(p == last_index, cur_dp, last_dp)
    return (cur_dp, new_last_dp), cur_bt

  p_ids = jnp.arange(p_max, dtype=jnp.int32)
  prev_ref_idx = ref_idx[jnp.maximum(p_ids - jnp.int32(1), jnp.int32(0))]
  init_dp = jnp.zeros((t_deg,), dtype=jnp.float32)
  (_, last_dp), backtrace = jax.lax.scan(
      _dp_step, (init_dp, init_dp), (p_ids, ref_stats, ref_idx, prev_ref_idx)
  )

  # Ties favor smallest slide_offset; lax.argmax keeps int32 under x64.
  last_offset = jax.lax.argmax(
      jnp.where(
          _in_search_window(ref_idx[last_index]), last_dp, jnp.float32(-jnp.inf)
      ),
      0,
      jnp.int32,
  )

  def _backtrace_step(
      cur_offset: jax.Array, inputs: tuple[jax.Array, jax.Array]
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    p, bt_row = inputs
    active = p < num_patches
    safe_off = jnp.clip(cur_offset, jnp.int32(0), jnp.int32(t_deg - 1))
    prev_off = bt_row[safe_off]
    is_pkt_loss = active & (cur_offset == prev_off)
    next_offset = jnp.where(active, prev_off, cur_offset)
    return next_offset, (cur_offset, is_pkt_loss)

  _, (matched_offsets, is_pkt_loss) = jax.lax.scan(
      _backtrace_step, last_offset, (p_ids, backtrace), reverse=True
  )
  return matched_offsets, is_pkt_loss


def find_most_optimal_deg_patches(
    ref_patches: jax.typing.ArrayLike,
    ref_patch_indices: jax.typing.ArrayLike,
    deg_spectrogram: jax.typing.ArrayLike,
    frame_duration: float | jax.typing.ArrayLike,
    search_window_radius: int = DEFAULT_SEARCH_WINDOW_RADIUS,
    *,
    num_ref_patches: jax.typing.ArrayLike | None = None,
    deg_valid_cols: jax.typing.ArrayLike | None = None,
) -> PatchSimilarityResult:
  """Optimal monotonic degraded patch match (`FindMostOptimalDegPatches`)."""
  sw_rad = _check.check_nonneg_int(search_window_radius, "search_window_radius")
  fd = _as_positive_f32(frame_duration, "frame_duration")
  ref_p = _check.asarray_f32(ref_patches, "ref_patches")
  deg_spec = _check.asarray_f32(deg_spectrogram, "deg_spectrogram")
  if ref_p.ndim != 3 or min(ref_p.shape) == 0:
    raise ValueError(
        f"ref_patches must be a non-empty 3D array (P, B, W), got {ref_p.shape}"
    )
  _check.check_2d_nonempty(deg_spec, "deg_spectrogram")
  if ref_p.shape[1] != deg_spec.shape[0]:
    raise ValueError(
        f"band count mismatch: ref_patches has {ref_p.shape[1]}, "
        f"deg_spectrogram has {deg_spec.shape[0]}"
    )
  ref_idx = _check_ref_indices(ref_patch_indices, expected_len=ref_p.shape[0])

  p_max, _, patch_size = ref_p.shape
  t_deg = deg_spec.shape[1]
  n_deg = (
      jnp.int32(t_deg)
      if deg_valid_cols is None
      else _valid_len(deg_valid_cols, t_deg, "deg_valid_cols")
  )
  raw_num_patches = calc_max_num_patches(
      ref_idx, n_deg, patch_size, num_ref_patches=num_ref_patches
  )
  num_patches = jnp.where(
      n_deg > jnp.int32(0), raw_num_patches, jnp.int32(0)
  )

  zero = jnp.float32(0.0)
  s_all = jnp.arange(t_deg, dtype=jnp.int32)
  # Build all T_deg degraded patches zero-padded past n_deg.
  if deg_valid_cols is not None:
    deg_spec = jnp.where((s_all < n_deg)[None, :], deg_spec, zero)
  padded_deg = jnp.pad(
      deg_spec, ((0, 0), (0, patch_size - 1)), constant_values=zero
  )
  deg_patches = patches.create_patches_from_indices(
      padded_deg, s_all, patch_size
  )

  matched_offsets, is_pkt_loss = _dp_align_patches(
      ref_p, ref_idx, deg_patches, num_patches, n_deg, sw_rad
  )
  safe_matched = jnp.clip(matched_offsets, jnp.int32(0), jnp.int32(t_deg - 1))
  full_sim = nsim.measure_patch_similarity(ref_p, deg_patches[safe_matched])

  active = jnp.arange(p_max, dtype=jnp.int32) < num_patches
  matched_valid = active & (~is_pkt_loss)
  patch_dur = fd * jnp.float32(patch_size)
  ref_start = jnp.where(active, ref_idx.astype(jnp.float32) * fd, zero)
  deg_start = jnp.where(
      matched_valid, matched_offsets.astype(jnp.float32) * fd, zero
  )

  # Packet loss zeroes similarity and means while keeping stddevs and energy.
  return PatchSimilarityResult(
      similarity=jnp.where(matched_valid, full_sim.similarity, zero),
      freq_band_means=jnp.where(
          matched_valid[:, None], full_sim.freq_band_means, zero
      ),
      freq_band_stddevs=jnp.where(
          active[:, None], full_sim.freq_band_stddevs, zero
      ),
      freq_band_deg_energy=jnp.where(
          active[:, None], full_sim.freq_band_deg_energy, zero
      ),
      ref_patch_start_time=ref_start,
      ref_patch_end_time=jnp.where(active, ref_start + patch_dur, zero),
      deg_patch_start_time=deg_start,
      deg_patch_end_time=jnp.where(matched_valid, deg_start + patch_dur, zero),
      num_patches=num_patches,
  )


def slice_audio(
    signal: jax.typing.ArrayLike,
    start_time: float | jax.typing.ArrayLike,
    end_time: float | jax.typing.ArrayLike,
    sample_rate: float | jax.typing.ArrayLike,
    *,
    signal_len: jax.typing.ArrayLike | None = None,
    max_len: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Extract a time-domain slice (`ComparisonPatchesSelector::Slice`).

  Preserves C++ `end_index = min(N - 1, trunc(end_time * fs))` and prepends both
  `postsilence` and `presilence`. Pass `max_len` for static-shape `jax.jit`.

  Args:
    signal: Non-empty 1D time-domain waveform.
    start_time: Slice start time in seconds.
    end_time: Slice end time in seconds.
    sample_rate: Positive sample rate in Hz.
    signal_len: Optional active sample count for `signal`.
    max_len: Optional static output buffer length for `jax.jit`.

  Returns:
    Tuple `(sliced_signal, valid_len)`.
  """
  fs = _as_positive_f32(sample_rate, "sample_rate")
  t_start = _as_finite_f32(start_time, "start_time")
  t_end = _as_finite_f32(end_time, "end_time")
  x = _check.asarray_f32(signal, "signal")
  _check.check_1d_nonempty(x, "signal")
  n_buf = x.shape[0]
  n_sig = (
      jnp.int32(n_buf)
      if signal_len is None
      else _valid_len(signal_len, n_buf, "signal_len")
  )
  zero = jnp.float32(0.0)
  i0 = jnp.int32(0)
  imax = jnp.int32(_INT32_MAX)

  start_index = jnp.maximum(i0, _trunc_i32(t_start * fs))
  end_index = jnp.minimum(n_sig - jnp.int32(1), _trunc_i32(t_end * fs))
  core_len = jnp.maximum(end_index, start_index) - start_index

  end_diff = t_end * fs - n_sig.astype(jnp.float32)
  postsilence = jnp.where(end_diff > zero, _trunc_i32(end_diff), i0)
  presilence = jnp.where(t_start < zero, _trunc_i32(-t_start * fs), i0)
  # C++ JoinVertically prepends postsilence first, then presilence before that.
  pre_zeros = presilence + jnp.minimum(postsilence, imax - presilence)
  raw_valid_len = pre_zeros + jnp.minimum(core_len, imax - pre_zeros)

  if max_len is None:
    if isinstance(raw_valid_len, jax.core.Tracer):
      raise ValueError("max_len must be specified when slicing under jax.jit")
    out_len = int(raw_valid_len)
    valid_len = jnp.int32(out_len)
  else:
    out_len = _check.check_nonneg_int(max_len, "max_len")
    if out_len > _INT32_MAX:
      raise ValueError(
          f"max_len ({out_len}) exceeds int32 limit ({_INT32_MAX})"
      )
    valid_len = jnp.minimum(raw_valid_len, jnp.int32(out_len))

  out_idx = jnp.arange(out_len, dtype=jnp.int32)
  in_core = (out_idx >= pre_zeros) & (out_idx < valid_len)
  core_off = jnp.where(in_core, out_idx - pre_zeros, i0)
  src_idx = start_index + core_off
  safe_src = jnp.clip(src_idx, i0, jnp.int32(n_buf - 1))
  return jnp.where(in_core, x[safe_src], zero), valid_len


def finely_align_and_recreate_patches(
    sim_results: PatchSimilarityResult,
    ref_signal: jax.typing.ArrayLike,
    deg_signal: jax.typing.ArrayLike,
    sample_rate: int,
    *,
    use_speech_mode: bool = False,
    patch_size: int | None = None,
    ref_len: jax.typing.ArrayLike | None = None,
    deg_len: jax.typing.ArrayLike | None = None,
    num_bands: int | None = None,
    min_freq: float = spectrogram.MIN_FREQ,
    max_freq: float | None = None,
    overlap: float = spectrogram.DEFAULT_OVERLAP,
    window_duration: float = spectrogram.DEFAULT_WINDOW_DURATION,
) -> PatchSimilarityResult:
  """Fine time-domain patch realignment (`FinelyAlignAndRecreatePatches`)."""
  speech = _as_bool(use_speech_mode, "use_speech_mode")
  ref = _check.asarray_f32(ref_signal, "ref_signal")
  deg = _check.asarray_f32(deg_signal, "deg_signal")
  _check.check_1d_nonempty(ref, "ref_signal")
  _check.check_1d_nonempty(deg, "deg_signal")
  fs_int = _check.check_positive_int(sample_rate, "sample_rate")
  fs = _as_positive_f32(fs_int, "sample_rate")
  if patch_size is None:
    ps = patches.SPEECH_PATCH_SIZE if speech else patches.AUDIO_PATCH_SIZE
  else:
    ps = _check.check_positive_int(patch_size, "patch_size")
  n_ref = (
      None
      if ref_len is None
      else _valid_len(ref_len, ref.shape[0], "ref_len")
  )
  n_deg = (
      None
      if deg_len is None
      else _valid_len(deg_len, deg.shape[0], "deg_len")
  )

  if num_bands is None:
    eff_bands = (
        spectrogram.SPEECH_MODE_NUM_BANDS
        if speech
        else spectrogram.AUDIO_MODE_NUM_BANDS
    )
  else:
    eff_bands = _check.check_positive_int(num_bands, "num_bands")
  if max_freq is None:
    eff_max_freq = (
        spectrogram.SPEECH_MODE_MAX_FREQ if speech else fs_int / 2.0
    )
  else:
    eff_max_freq = max_freq
  spectrogram.make_erb_filters(fs_int, eff_bands, min_freq, eff_max_freq)

  window_size, hop_size, _ = spectrogram.analysis_window(
      fs_int, overlap=overlap, window_duration=window_duration
  )
  # +2 covers +-1 sample IEEE truncation drift in end_time*fs - start_time*fs.
  slice_len = ps * hop_size + 2
  if slice_len < window_size:
    raise ValueError(
        f"patch slice length ({slice_len}) smaller than window_size"
        f" ({window_size})"
    )
  if slice_len > _INT32_MAX:
    raise ValueError(
        f"patch slice length ({slice_len}) exceeds int32 limit ({_INT32_MAX})"
    )

  if (
      not isinstance(sim_results, PatchSimilarityResult)
      or len(sim_results) != 9
  ):
    raise ValueError(
        f"sim_results must be a PatchSimilarityResult, got {sim_results!r}"
    )
  fields = tuple(
      _check.asarray_f32(val, f"sim_results.{name}")
      for name, val in zip(PatchSimilarityResult._fields[:-1], sim_results[:-1])
  )
  sim_arr = fields[0]
  if sim_arr.ndim != 1:
    raise ValueError(
        f"sim_results.similarity must be 1D, got shape {sim_arr.shape}"
    )
  p_max = sim_arr.shape[0]
  for name, arr in zip(PatchSimilarityResult._fields[:-1], fields):
    want_shape = (
        (p_max, eff_bands) if name.startswith("freq_band_") else (p_max,)
    )
    if arr.shape != want_shape:
      raise ValueError(
          f"sim_results.{name} must have shape {want_shape}, got {arr.shape}"
      )
  n_patches = _valid_len(sim_results.num_patches, p_max, "num_patches")
  if p_max == 0:
    return PatchSimilarityResult(*fields, num_patches=jnp.int32(0))

  zero = jnp.float32(0.0)
  # pylint: disable-next=protected-access
  spec_fn = spectrogram._build_gammatone_spectrograms
  build_specs = functools.partial(
      spec_fn,
      sample_rate=fs_int,
      use_speech_mode=speech,
      num_bands=eff_bands,
      min_freq=min_freq,
      max_freq=eff_max_freq,
      overlap=overlap,
      window_duration=window_duration,
  )

  def _realign_one(
      p: jax.Array, old: tuple[jax.Array, ...]
  ) -> tuple[jax.Array, ...]:
    old_sim, _, _, _, old_ref_s, old_ref_e, old_deg_s, old_deg_e = old
    active = p < n_patches
    is_pkt_loss = (old_deg_s == zero) & (old_deg_e == zero)
    ref_s, ref_v = slice_audio(
        ref, old_ref_s, old_ref_e, fs_int, signal_len=n_ref, max_len=slice_len
    )
    deg_s, deg_v = slice_audio(
        deg, old_deg_s, old_deg_e, fs_int, signal_len=n_deg, max_len=slice_len
    )
    ref_aln, deg_aln, aln_len, lag_s = alignment.align_and_truncate(
        ref_s, deg_s, fs_int, ref_len=ref_v, deg_len=deg_v
    )
    (ref_spec, deg_spec), _ = build_specs((ref_aln, deg_aln))
    valid_cols = spectrogram.num_spectrogram_frames(
        aln_len, window_size, hop_size
    )
    prep_ref, prep_deg = spectrogram.prepare_spectrograms(
        ref_spec, deg_spec, ref_valid_cols=valid_cols, deg_valid_cols=valid_cols
    )
    new_res = nsim.measure_patch_similarity(
        prep_ref, prep_deg, valid_cols=valid_cols
    )

    use_new = (
        active
        & (~is_pkt_loss)
        & (valid_cols >= jnp.int32(1))
        & (new_res.similarity >= old_sim)
    )
    new_dur = aln_len.astype(jnp.float32) / fs
    cand_ref_s = jnp.where(lag_s > zero, old_ref_s + lag_s, old_ref_s)
    cand_deg_s = jnp.where(lag_s > zero, old_deg_s, old_deg_s - lag_s)
    cand = (
        new_res.similarity,
        new_res.freq_band_means,
        new_res.freq_band_stddevs,
        new_res.freq_band_deg_energy,
        cand_ref_s,
        cand_ref_s + new_dur,
        cand_deg_s,
        cand_deg_s + new_dur,
    )
    return tuple(
        jnp.where(active, jnp.where(use_new, n, o), zero)
        for n, o in zip(cand, old)
    )

  out = jax.vmap(_realign_one)(jnp.arange(p_max, dtype=jnp.int32), fields)
  return PatchSimilarityResult(*out, num_patches=n_patches)
