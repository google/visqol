"""Live differential float32 parity tests against C++ ViSQOL v3.3.3."""

import array
from collections.abc import Sequence
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import math
import os
import pathlib
import struct
import sys
from typing import NoReturn, Protocol, cast

import jax
import jax.numpy as jnp
import pytest

import visqol_jax


class _CppPatchSimilarity(Protocol):
  similarity: float
  freq_band_means: Sequence[float]
  ref_patch_start_time: float
  ref_patch_end_time: float
  deg_patch_start_time: float
  deg_patch_end_time: float


class _CppSimilarityResult(Protocol):
  moslqo: float
  vnsim: float
  fvnsim: Sequence[float]
  fvnsim10: Sequence[float]
  fstdnsim: Sequence[float]
  fvdegenergy: Sequence[float]
  alignment_lag_s: float
  center_freq_bands: Sequence[float]
  patch_sims: Sequence[_CppPatchSimilarity]


class _CppFilePath(Protocol):
  pass


# pylint: disable=invalid-name
class _CppVisqolManager(Protocol):

  def Init(
      self,
      model_path: _CppFilePath | None,
      use_speech: bool,
      use_unscaled_speech_mos_mapping: bool,
      search_window_radius: int,
      use_lattice_model: bool,
      disable_global_alignment: bool,
      disable_realignment: bool,
  ) -> None: ...

  def Run(
      self, ref_path: _CppFilePath, deg_path: _CppFilePath
  ) -> _CppSimilarityResult: ...


class _CppAudioConfig(Protocol):
  sample_rate: int


class _CppOptionsConfig(Protocol):
  use_speech_scoring: bool
  use_lattice_model: bool
  use_unscaled_speech_mos_mapping: bool
  allow_unsupported_sample_rates: bool
  skip_mos_mapping: bool
  search_window_radius: int
  svr_model_path: str


class _CppVisqolConfig(Protocol):
  audio: _CppAudioConfig
  options: _CppOptionsConfig


class _CppVisqolApi(Protocol):

  def Create(self, config: _CppVisqolConfig) -> None: ...

  def Measure(
      self, ref: Sequence[float], deg: Sequence[float]
  ) -> _CppSimilarityResult: ...


class _VisqolLibModule(Protocol):
  __doc__: str | None

  def FilePath(self, path: str) -> _CppFilePath: ...

  def VisqolManager(self) -> _CppVisqolManager: ...

  def VisqolApi(self) -> _CppVisqolApi: ...

  def ConformanceSpeechCA01TranscodedValue(self) -> float: ...


class _VisqolConfigModule(Protocol):

  def VisqolConfig(self) -> _CppVisqolConfig: ...
# pylint: enable=invalid-name


def _has_spec(mod_name: str) -> bool:
  try:
    return importlib.util.find_spec(mod_name) is not None
  except ModuleNotFoundError:
    return False


def _load_cpp_bindings(
    pkg_name: str | None = None,
) -> tuple[_VisqolLibModule, _VisqolConfigModule]:
  """Load C++ bindings (mandatory in monorepo; optional in OSS pure-Python)."""
  name = visqol_jax.__name__ if pkg_name is None else pkg_name
  root_pkg = name.rpartition(".")[0]
  if root_pkg:
    return (
        cast(
            _VisqolLibModule,
            importlib.import_module(f"{root_pkg}.python.visqol_lib_py"),
        ),
        cast(
            _VisqolConfigModule,
            importlib.import_module(f"{root_pkg}.src.proto.visqol_config_pb2"),
        ),
    )
  for lib_mod, cfg_mod, sim_mod in (
      (
          "visqol.visqol_lib_py",
          "visqol.pb2.visqol_config_pb2",
          "visqol.pb2.similarity_result_pb2",
      ),
      (
          "python.visqol_lib_py",
          "visqol_config_pb2",
          "similarity_result_pb2",
      ),
  ):
    if _has_spec(lib_mod):
      importlib.import_module(sim_mod)
      return (
          cast(_VisqolLibModule, importlib.import_module(lib_mod)),
          cast(_VisqolConfigModule, importlib.import_module(cfg_mod)),
      )
  pytest.skip(
      "C++ visqol_lib_py bindings are not installed",
      allow_module_level=True,
  )
  raise RuntimeError("unreachable")


visqol_lib_py, visqol_config_pb2 = _load_cpp_bindings()

_SPEECH_MODEL_REL = (
    "model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_"
    "ep2400_train1_7_raw.tflite"
)
_AUDIO_MODEL_REL = "model/libsvm_nu_svr_model.txt"
_CUSTOM_CPP_SVR_REL = "testdata/test_model/cpp_model.txt"
_NUM_PATCHES_PATH = (
    jax.tree_util.GetAttrKey("patch_sims"),
    jax.tree_util.GetAttrKey("num_patches"),
)


def _lcg_f32(
    seed: int, n: int, lo: float = -1.0, hi: float = 1.0
) -> jax.Array:
  state = seed & 0xFFFFFFFF
  vals = [0.0] * n
  scale = (hi - lo) / float(0x100000000)
  for i in range(n):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    vals[i] = lo + state * scale
  return jnp.asarray(vals, dtype=jnp.float32)


def _hann(n: int) -> jax.Array:
  idx = jnp.arange(n, dtype=jnp.float32)
  return jnp.float32(0.5) - jnp.float32(0.5) * jnp.cos(
      jnp.float32(2.0 * math.pi) * idx / jnp.float32(n - 1)
  )


def _assert_allclose(
    actual: jax.typing.ArrayLike,
    expected: (
        jax.typing.ArrayLike | Sequence[float] | Sequence[Sequence[float]]
    ),
    *,
    atol: float,
    rtol: float,
    label: str,
) -> None:
  a = jnp.asarray(actual, dtype=jnp.float32)
  b = jnp.asarray(expected, dtype=jnp.float32)
  assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
  assert bool(jnp.allclose(a, b, atol=atol, rtol=rtol)), (
      f"{label}: max abs diff {float(jnp.max(jnp.abs(a - b)))}"
  )


@functools.cache
def _find_repo_file(relpath: str) -> pathlib.Path:
  """Locate `model/...` or `testdata/...` across runfiles and repo layouts."""
  this_file = pathlib.Path(__file__)
  for base in (this_file, this_file.resolve()):
    for parent in base.parents:
      candidate = parent / relpath
      if candidate.is_file():
        return candidate
  for env_name in ("RUNFILES_DIR", "TEST_SRCDIR"):
    env_val = os.environ.get(env_name)
    if env_val:
      root = pathlib.Path(env_val)
      if root.is_dir():
        for match in root.rglob(relpath):
          if match.is_file():
            return match
  raise FileNotFoundError(f"Could not locate {relpath}")


@functools.cache
def _read_pcm16_wav_mono(path: pathlib.Path) -> tuple[jax.Array, int]:
  """Read a 16-bit PCM WAV file into mono `float32` matching `LoadAsMono`."""
  raw = path.read_bytes()
  riff_id, _, wave_id = struct.unpack_from("<4sI4s", raw, 0)
  if riff_id != b"RIFF" or wave_id != b"WAVE":
    raise ValueError(f"Not a RIFF/WAVE file: {path}")
  pos = 12
  num_channels = 0
  sample_rate = 0
  pcm_bytes = b""
  while pos + 8 <= len(raw):
    chunk_id, chunk_size = struct.unpack_from("<4sI", raw, pos)
    pos += 8
    if chunk_id == b"fmt ":
      fmt_tag, num_channels, sample_rate, _, _, bits = struct.unpack_from(
          "<HHIIHH", raw, pos
      )
      if fmt_tag not in (0x0001, 0xFFFE) or bits != 16:
        raise ValueError(
            f"Unsupported WAV format ({fmt_tag}, {bits}) in {path}"
        )
    elif chunk_id == b"data":
      pcm_bytes = raw[pos : pos + chunk_size]
      break
    pad = chunk_size & 1 if pos + chunk_size < len(raw) else 0
    pos += chunk_size + pad
  if not pcm_bytes or num_channels < 1:
    raise ValueError(f"Missing fmt/data chunk in {path}")
  usable = (len(pcm_bytes) // (2 * num_channels)) * num_channels
  buf = pcm_bytes[: usable * 2]
  if sys.byteorder != "little":
    swapped = array.array("h")
    swapped.frombytes(buf)
    swapped.byteswap()
    buf = swapped.tobytes()
  samples = jnp.frombuffer(buf, dtype=jnp.int16)
  norm = samples.astype(jnp.float32) / jnp.float32(32768.0)
  if num_channels > 1:
    norm = jnp.mean(norm.reshape(-1, num_channels), axis=-1)
  return norm, sample_rate


def _write_pcm16_wav_mono(
    path: pathlib.Path, samples: jax.Array, sample_rate: int
) -> jax.Array:
  """Write mono 16-bit PCM WAV and return exact quantized `float32` samples."""
  pcm = jnp.clip(
      jnp.round(samples * jnp.float32(32768.0)),
      jnp.float32(-32768.0),
      jnp.float32(32767.0),
  ).astype(jnp.int16)
  arr = array.array("h", pcm.tolist())
  if sys.byteorder != "little":
    arr.byteswap()
  data_bytes = arr.tobytes()
  header = struct.pack(
      "<4sI4s4sIHHIIHH4sI",
      b"RIFF",
      36 + len(data_bytes),
      b"WAVE",
      b"fmt ",
      16,
      1,
      1,
      sample_rate,
      sample_rate * 2,
      2,
      16,
      b"data",
      len(data_bytes),
  )
  path.write_bytes(header + data_bytes)
  return pcm.astype(jnp.float32) / jnp.float32(32768.0)


def _cpp_model_for_options(opts: visqol_jax.VisqolOptions) -> pathlib.Path:
  if opts.svr_model_path is not None:
    return pathlib.Path(opts.svr_model_path)
  if opts.use_speech_scoring:
    return _find_repo_file(_SPEECH_MODEL_REL)
  return _find_repo_file(_AUDIO_MODEL_REL)


def _run_cpp_manager(
    ref_wav: pathlib.Path,
    deg_wav: pathlib.Path,
    opts: visqol_jax.VisqolOptions,
) -> _CppSimilarityResult:
  mgr = visqol_lib_py.VisqolManager()
  model_fp = (
      None
      if opts.skip_mos_mapping
      else visqol_lib_py.FilePath(str(_cpp_model_for_options(opts)))
  )
  mgr.Init(
      model_fp,
      opts.use_speech_scoring,
      opts.use_unscaled_speech_mos_mapping,
      opts.search_window_radius,
      opts.use_lattice_model,
      opts.disable_global_alignment,
      opts.disable_realignment,
  )
  return mgr.Run(
      visqol_lib_py.FilePath(str(ref_wav)),
      visqol_lib_py.FilePath(str(deg_wav)),
  )


def _run_cpp_api(
    ref: jax.Array,
    deg: jax.Array,
    sample_rate: int,
    opts: visqol_jax.VisqolOptions,
) -> _CppSimilarityResult:
  cfg = visqol_config_pb2.VisqolConfig()
  cfg.audio.sample_rate = sample_rate
  cfg.options.use_speech_scoring = opts.use_speech_scoring
  cfg.options.use_lattice_model = opts.use_lattice_model
  cfg.options.use_unscaled_speech_mos_mapping = (
      opts.use_unscaled_speech_mos_mapping
  )
  cfg.options.allow_unsupported_sample_rates = (
      opts.allow_unsupported_sample_rates
  )
  cfg.options.skip_mos_mapping = opts.skip_mos_mapping
  if opts.search_window_radius > 0:
    cfg.options.search_window_radius = opts.search_window_radius
  if not opts.skip_mos_mapping:
    cfg.options.svr_model_path = str(_cpp_model_for_options(opts))
  api = visqol_lib_py.VisqolApi()
  api.Create(cfg)
  return api.Measure(ref.tolist(), deg.tolist())


def _assert_parity(
    jax_res: visqol_jax.VisqolResult,
    cpp_res: _CppSimilarityResult,
    opts: visqol_jax.VisqolOptions,
    sample_rate: int,
    label: str,
    *,
    audio_conformance: bool = False,
) -> None:
  """Assert float32 parity between JAX and C++ ViSQOL outputs."""
  for path, leaf in jax.tree_util.tree_leaves_with_path(jax_res):
    want_dt = jnp.int32 if path == _NUM_PATCHES_PATH else jnp.float32
    assert isinstance(leaf, jax.Array) and leaf.dtype == want_dt

  # Exact sample lag and float32-rounded seconds lag (within 1 float32 ULP).
  assert round(float(jax_res.alignment_lag_s) * sample_rate) == round(
      cpp_res.alignment_lag_s * sample_rate
  ), label
  _assert_allclose(
      jax_res.alignment_lag_s,
      cpp_res.alignment_lag_s,
      atol=1e-7,
      rtol=1e-6,
      label=f"{label}: alignment_lag_s",
  )
  _assert_allclose(
      jax_res.center_freq_bands,
      list(cpp_res.center_freq_bands),
      atol=0.0,
      rtol=1e-6,
      label=f"{label}: center_freq_bands",
  )

  score_tol = 2e-3 if audio_conformance else 1e-2
  if opts.skip_mos_mapping:
    assert math.isnan(float(jax_res.moslqo)), label
    assert cpp_res.moslqo == 0.0, label
  else:
    assert abs(float(jax_res.moslqo) - cpp_res.moslqo) <= score_tol, (
        f"{label}: moslqo {float(jax_res.moslqo)} vs {cpp_res.moslqo}"
    )

  assert abs(float(jax_res.vnsim) - cpp_res.vnsim) <= score_tol, (
      f"{label}: vnsim {float(jax_res.vnsim)} vs {cpp_res.vnsim}"
  )
  _assert_allclose(
      jax_res.fvnsim,
      list(cpp_res.fvnsim),
      atol=2e-2,
      rtol=2e-2,
      label=f"{label}: fvnsim",
  )
  _assert_allclose(
      jax_res.fvnsim10,
      list(cpp_res.fvnsim10),
      atol=0.1,
      rtol=0.1,
      label=f"{label}: fvnsim10",
  )
  _assert_allclose(
      jax_res.fstdnsim,
      list(cpp_res.fstdnsim),
      atol=5e-2,
      rtol=5e-2,
      label=f"{label}: fstdnsim",
  )
  _assert_allclose(
      jax_res.fvdegenergy,
      list(cpp_res.fvdegenergy),
      atol=0.25,
      rtol=5e-2,
      label=f"{label}: fvdegenergy",
  )

  ps = jax_res.patch_sims
  n_p = int(ps.num_patches)
  assert n_p == len(cpp_res.patch_sims), f"{label}: num_patches"
  _assert_allclose(
      ps.similarity[:n_p],
      [p.similarity for p in cpp_res.patch_sims],
      atol=5e-2,
      rtol=5e-2,
      label=f"{label}: patch similarity",
  )
  _assert_allclose(
      ps.freq_band_means[:n_p],
      [list(p.freq_band_means) for p in cpp_res.patch_sims],
      atol=0.1,
      rtol=0.1,
      label=f"{label}: patch freq_band_means",
  )
  time_tol = 0.025
  _assert_allclose(
      ps.ref_patch_start_time[:n_p],
      [p.ref_patch_start_time for p in cpp_res.patch_sims],
      atol=time_tol,
      rtol=1e-4,
      label=f"{label}: ref_patch_start_time",
  )
  _assert_allclose(
      ps.ref_patch_end_time[:n_p],
      [p.ref_patch_end_time for p in cpp_res.patch_sims],
      atol=time_tol,
      rtol=1e-4,
      label=f"{label}: ref_patch_end_time",
  )
  _assert_allclose(
      ps.deg_patch_start_time[:n_p],
      [p.deg_patch_start_time for p in cpp_res.patch_sims],
      atol=time_tol,
      rtol=1e-4,
      label=f"{label}: deg_patch_start_time",
  )
  _assert_allclose(
      ps.deg_patch_end_time[:n_p],
      [p.deg_patch_end_time for p in cpp_res.patch_sims],
      atol=time_tol,
      rtol=1e-4,
      label=f"{label}: deg_patch_end_time",
  )
  for arr in ps[:-1]:
    assert bool(jnp.all(arr[n_p:] == jnp.float32(0.0))), label


@dataclasses.dataclass(frozen=True)
class _WavCase:
  name: str
  ref_rel: str
  deg_rel: str
  options: visqol_jax.VisqolOptions
  custom_svr_rel: str | None = None
  check_api: bool = True
  audio_conformance: bool = False


_SUBSET_DIR = "testdata/conformance_testdata_subset"

_WAV_CASES: tuple[_WavCase, ...] = (
    _WavCase(
        "ca01_transcoded_speech_lattice",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/transcoded_CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, use_lattice_model=True
        ),
    ),
    _WavCase(
        "ca01_transcoded_speech_exp_scaled",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/transcoded_CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, use_lattice_model=False
        ),
    ),
    _WavCase(
        "ca01_transcoded_speech_exp_unscaled",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/transcoded_CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True,
            use_lattice_model=False,
            use_unscaled_speech_mos_mapping=True,
        ),
    ),
    _WavCase(
        "ca01_perfect_speech_lattice",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, use_lattice_model=True
        ),
    ),
    _WavCase(
        "ca01_perfect_speech_exp_unscaled",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True,
            use_lattice_model=False,
            use_unscaled_speech_mos_mapping=True,
        ),
    ),
    _WavCase(
        "ca01_transcoded_as_audio_svr",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/transcoded_CA01_01.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
    ),
    _WavCase(
        "ca01_speech_skip_mos_mapping",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/transcoded_CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, skip_mos_mapping=True
        ),
    ),
    _WavCase(
        "ca01_speech_disable_realignment",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/transcoded_CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, disable_realignment=True
        ),
        check_api=False,
    ),
    _WavCase(
        "ca01_audio_disable_realignment",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/transcoded_CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=False, disable_realignment=True
        ),
        check_api=False,
    ),
    _WavCase(
        "ca01_audio_search_window_10",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/transcoded_CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=False, search_window_radius=10
        ),
    ),
    _WavCase(
        "ca01_audio_search_window_0",
        "testdata/clean_speech/CA01_01.wav",
        "testdata/clean_speech/transcoded_CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=False, search_window_radius=0
        ),
        check_api=False,
    ),
    _WavCase(
        "strauss48_lp35",
        f"{_SUBSET_DIR}/strauss48_stereo.wav",
        f"{_SUBSET_DIR}/strauss48_stereo_lp35.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "steely48_lp7",
        f"{_SUBSET_DIR}/steely48_stereo.wav",
        f"{_SUBSET_DIR}/steely48_stereo_lp7.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "sopr48_256kbps_aac",
        f"{_SUBSET_DIR}/sopr48_stereo.wav",
        f"{_SUBSET_DIR}/sopr48_stereo_256kbps_aac.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "ravel48_128kbps_opus",
        f"{_SUBSET_DIR}/ravel48_stereo.wav",
        f"{_SUBSET_DIR}/ravel48_stereo_128kbps_opus.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "moonlight48_128kbps_aac",
        f"{_SUBSET_DIR}/moonlight48_stereo.wav",
        f"{_SUBSET_DIR}/moonlight48_stereo_128kbps_aac.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "harpsichord48_96kbps_mp3",
        f"{_SUBSET_DIR}/harpsichord48_stereo.wav",
        f"{_SUBSET_DIR}/harpsichord48_stereo_96kbps_mp3.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "glock48_48kbps_aac",
        f"{_SUBSET_DIR}/glock48_stereo.wav",
        f"{_SUBSET_DIR}/glock48_stereo_48kbps_aac.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "contrabassoon48_24kbps_aac",
        f"{_SUBSET_DIR}/contrabassoon48_stereo.wav",
        f"{_SUBSET_DIR}/contrabassoon48_stereo_24kbps_aac.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "contrabassoon48_skip_mos_mapping",
        f"{_SUBSET_DIR}/contrabassoon48_stereo.wav",
        f"{_SUBSET_DIR}/contrabassoon48_stereo_24kbps_aac.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=False, skip_mos_mapping=True
        ),
        audio_conformance=True,
    ),
    _WavCase(
        "guitar48_64kbps_aac_default_svr",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        f"{_SUBSET_DIR}/guitar48_stereo_64kbps_aac.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "guitar48_64kbps_aac_custom_cpp_model",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        f"{_SUBSET_DIR}/guitar48_stereo_64kbps_aac.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        custom_svr_rel=_CUSTOM_CPP_SVR_REL,
        audio_conformance=True,
    ),
    _WavCase(
        "castanets48_identity",
        f"{_SUBSET_DIR}/castanets48_stereo.wav",
        f"{_SUBSET_DIR}/castanets48_stereo.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "alignment_bad_degraded_speech_lattice",
        "testdata/alignment/reference.wav",
        "testdata/alignment/degraded.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, use_lattice_model=True
        ),
    ),
    _WavCase(
        "alignment_bad_degraded_speech_exp",
        "testdata/alignment/reference.wav",
        "testdata/alignment/degraded.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, use_lattice_model=False
        ),
    ),
    _WavCase(
        "alignment_bad_degraded_disable_global_alignment",
        "testdata/alignment/reference.wav",
        "testdata/alignment/degraded.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, disable_global_alignment=True
        ),
        check_api=False,
    ),
    _WavCase(
        "mismatched_duration_guitar_middle_50ms_cut",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        "testdata/mismatched_duration/guitar48_stereo_middle_50ms_cut.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
    ),
    _WavCase(
        "mismatched_duration_guitar_middle_2sec_cut_deg",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        "testdata/mismatched_duration/guitar48_stereo_middle_2sec_cut.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
    ),
    _WavCase(
        "mismatched_duration_guitar_middle_2sec_cut_ref",
        "testdata/mismatched_duration/guitar48_stereo_middle_2sec_cut.wav",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
    ),
    _WavCase(
        "filtered_freqs_guitar_10k",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        "testdata/filtered_freqs/guitar48_stereo_10k_filtered_freqs.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
    ),
    _WavCase(
        "guitar_short_1s_identity",
        "testdata/short_duration/1_second/guitar48_stereo_1_sec.wav",
        "testdata/short_duration/1_second/guitar48_stereo_1_sec.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
    ),
    _WavCase(
        "guitar_short_5s_degraded",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        "testdata/short_duration/5_second/guitar48_stereo_5_sec.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "guitar_short_5s_reference",
        "testdata/short_duration/5_second/guitar48_stereo_5_sec.wav",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        visqol_jax.VisqolOptions(use_speech_scoring=False),
        audio_conformance=True,
    ),
    _WavCase(
        "non_48k_guitar_44100hz_audio_override",
        "testdata/non_48k_sample_rate/guitar48_stereo_44100Hz.wav",
        "testdata/non_48k_sample_rate/guitar48_stereo_44100Hz.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=False, allow_unsupported_sample_rates=True
        ),
    ),
    _WavCase(
        "different_audios_guitar_vs_ca01_lattice",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        "testdata/clean_speech/CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, use_lattice_model=True
        ),
    ),
    _WavCase(
        "different_audios_guitar_vs_ca01_exp",
        f"{_SUBSET_DIR}/guitar48_stereo.wav",
        "testdata/clean_speech/CA01_01.wav",
        visqol_jax.VisqolOptions(
            use_speech_scoring=True, use_lattice_model=False
        ),
    ),
)


@pytest.mark.parametrize("case", _WAV_CASES, ids=lambda c: c.name)
def test_live_cpp_wav_parity(case: _WavCase):
  ref_wav = _find_repo_file(case.ref_rel)
  deg_wav = _find_repo_file(case.deg_rel)
  opts = case.options
  if case.custom_svr_rel is not None:
    opts = dataclasses.replace(
        opts, svr_model_path=_find_repo_file(case.custom_svr_rel)
    )
  ref_mono, ref_sr = _read_pcm16_wav_mono(ref_wav)
  deg_mono, deg_sr = _read_pcm16_wav_mono(deg_wav)
  assert ref_sr == deg_sr

  jax_res = visqol_jax.visqol(ref_mono, deg_mono, ref_sr, opts)
  mgr_res = _run_cpp_manager(ref_wav, deg_wav, opts)
  _assert_parity(
      jax_res,
      mgr_res,
      opts,
      ref_sr,
      f"{case.name}/VisqolManager",
      audio_conformance=case.audio_conformance,
  )

  if case.check_api:
    api_res = _run_cpp_api(ref_mono, deg_mono, ref_sr, opts)
    _assert_parity(
        jax_res,
        api_res,
        opts,
        ref_sr,
        f"{case.name}/VisqolApi",
        audio_conformance=case.audio_conformance,
    )

  if case.name == "ca01_transcoded_speech_lattice":
    conf_val = visqol_lib_py.ConformanceSpeechCA01TranscodedValue()
    assert abs(float(jax_res.moslqo) - conf_val) <= 1e-2
  if case.name == "filtered_freqs_guitar_10k":
    assert int(jnp.argmin(jax_res.fvnsim)) == 26


def test_live_cpp_synthetic_lags_flags_and_non_div4_sample_rate(
    tmp_path: pathlib.Path,
):
  two_pi = jnp.float32(2.0 * math.pi)

  # 16 kHz lagged speech across alignment/realignment/search_window flags.
  t16 = jnp.arange(32000, dtype=jnp.float32) / jnp.float32(16000.0)
  env16 = jnp.float32(0.3) + jnp.float32(0.7) * jnp.exp(
      jnp.float32(-0.5) * ((t16 - 0.85) / 0.22) ** 2
  )
  sp_base = env16 * (
      jnp.float32(0.32) * jnp.sin(two_pi * (190.0 * t16 + 100.0 * t16 * t16))
      + jnp.float32(0.16) * jnp.sin(two_pi * (650.0 * t16 + 70.0 * t16 * t16))
  )
  sp_delayed = jnp.concatenate(
      [jnp.zeros((800,), dtype=jnp.float32), sp_base[:-800]]
  )
  sp_noisy = sp_base + _lcg_f32(20261117, 32000, -0.03, 0.03)

  ref16_wav = tmp_path / "sp_ref16.wav"
  deg16_wav = tmp_path / "sp_deg16.wav"
  sp_ref_q = _write_pcm16_wav_mono(ref16_wav, sp_delayed, 16000)
  sp_deg_q = _write_pcm16_wav_mono(deg16_wav, sp_noisy, 16000)

  for tag, sp_opts in (
      ("sp16_default", visqol_jax.VisqolOptions(use_speech_scoring=True)),
      (
          "sp16_no_global",
          visqol_jax.VisqolOptions(
              use_speech_scoring=True, disable_global_alignment=True
          ),
      ),
      (
          "sp16_no_realign",
          visqol_jax.VisqolOptions(
              use_speech_scoring=True, disable_realignment=True
          ),
      ),
      (
          "sp16_sw10",
          visqol_jax.VisqolOptions(
              use_speech_scoring=True, search_window_radius=10
          ),
      ),
  ):
    jax_res = visqol_jax.visqol(sp_ref_q, sp_deg_q, 16000, sp_opts)
    mgr_res = _run_cpp_manager(ref16_wav, deg16_wav, sp_opts)
    _assert_parity(jax_res, mgr_res, sp_opts, 16000, tag)

  # 11025 Hz (window_size=882, hop_size=220 -> non-div4 CalcFrameDuration).
  t11 = jnp.arange(24000, dtype=jnp.float32) / jnp.float32(11025.0)
  s11_ref = jnp.float32(0.35) * jnp.sin(
      two_pi * (240.0 * t11 + 80.0 * t11 * t11)
  )
  s11_deg = s11_ref + _lcg_f32(20261118, 24000, -0.035, 0.035)
  for tag, opts11 in (
      ("sr11025_speech", visqol_jax.VisqolOptions(use_speech_scoring=True)),
      (
          "sr11025_audio_override",
          visqol_jax.VisqolOptions(
              use_speech_scoring=False, allow_unsupported_sample_rates=True
          ),
      ),
  ):
    jax_res = visqol_jax.visqol(s11_ref, s11_deg, 11025, opts11)
    api_res = _run_cpp_api(s11_ref, s11_deg, 11025, opts11)
    _assert_parity(jax_res, api_res, opts11, 11025, tag)

  # Gzipped custom SVR (.txt.gz) in JAX matches plain cpp_model.txt in C++.
  cpp_svr_txt = _find_repo_file(_CUSTOM_CPP_SVR_REL)
  cpp_svr_gz = tmp_path / "cpp_model.txt.gz"
  cpp_svr_gz.write_bytes(
      gzip.compress(cpp_svr_txt.read_bytes(), compresslevel=6, mtime=0)
  )
  g_ref, g_sr = _read_pcm16_wav_mono(
      _find_repo_file(f"{_SUBSET_DIR}/guitar48_stereo.wav")
  )
  g_deg, _ = _read_pcm16_wav_mono(
      _find_repo_file(f"{_SUBSET_DIR}/guitar48_stereo_64kbps_aac.wav")
  )
  jax_gz = visqol_jax.visqol_audio(
      g_ref, g_deg, g_sr, svr_model_path=cpp_svr_gz
  )
  cpp_plain = _run_cpp_api(
      g_ref,
      g_deg,
      g_sr,
      visqol_jax.VisqolOptions(
          use_speech_scoring=False, svr_model_path=cpp_svr_txt
      ),
  )
  _assert_parity(
      jax_gz,
      cpp_plain,
      visqol_jax.VisqolOptions(use_speech_scoring=False),
      g_sr,
      "custom_cpp_model_txt_gz",
      audio_conformance=True,
  )

  # Short delayed degraded burst (< window_size) grows past window_size via lag.
  burst_n = 800
  delay = 4000
  idx_b = jnp.arange(burst_n, dtype=jnp.float32)
  burst = (
      jnp.float32(0.5)
      * jnp.sin(
          two_pi
          * (450.0 + 300.0 * idx_b / jnp.float32(16000.0))
          * idx_b
          / jnp.float32(16000.0)
      )
      * _hann(burst_n)
  )
  ref_short = (
      jnp.zeros((24000,), dtype=jnp.float32)
      .at[delay : delay + burst_n]
      .set(burst)
  )
  ref_sb_wav = tmp_path / "short_burst_ref.wav"
  deg_sb_wav = tmp_path / "short_burst_deg.wav"
  ref_sb_q = _write_pcm16_wav_mono(ref_sb_wav, ref_short, 16000)
  deg_sb_q = _write_pcm16_wav_mono(deg_sb_wav, burst, 16000)
  sb_opts = visqol_jax.VisqolOptions(use_speech_scoring=True)
  jax_sb = visqol_jax.visqol(ref_sb_q, deg_sb_q, 16000, sb_opts)
  assert float(jax_sb.alignment_lag_s) > 0.0
  assert int(jax_sb.patch_sims.num_patches) > 0
  _assert_parity(
      jax_sb,
      _run_cpp_manager(ref_sb_wav, deg_sb_wav, sb_opts),
      sb_opts,
      16000,
      "short_delayed_burst/VisqolManager",
  )
  _assert_parity(
      jax_sb,
      _run_cpp_api(ref_sb_q, deg_sb_q, 16000, sb_opts),
      sb_opts,
      16000,
      "short_delayed_burst/VisqolApi",
  )
  sb_no_align = visqol_jax.VisqolOptions(
      use_speech_scoring=True, disable_global_alignment=True
  )
  with pytest.raises(ValueError, match="degraded signal"):
    visqol_jax.visqol(ref_sb_q, deg_sb_q, 16000, sb_no_align)
  with pytest.raises(Exception):
    _run_cpp_manager(ref_sb_wav, deg_sb_wav, sb_no_align)


def test_cpp_binding_loader_monorepo_mandatory_and_oss_skip(
    monkeypatch: pytest.MonkeyPatch,
):
  assert "ViSQOL" in (visqol_lib_py.__doc__ or "")
  # Monorepo packages never skip on missing C++ bindings; they fail loudly.
  with pytest.raises(ModuleNotFoundError):
    _load_cpp_bindings("nonexistent_monorepo_root.visqol.jax")

  # OSS pure-Python runs skip cleanly when parent package or binding is absent.
  def _missing_parent(mod_name: str) -> None:
    raise ModuleNotFoundError(f"No module named {mod_name!r}")

  monkeypatch.setattr(importlib.util, "find_spec", _missing_parent)
  with pytest.raises(pytest.skip.Exception, match="visqol_lib_py"):
    _load_cpp_bindings("visqol_jax")

  # Installed-but-broken OSS bindings raise ImportError instead of skipping.
  monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())

  def _broken_import(mod_name: str) -> NoReturn:
    raise ImportError(f"broken extension {mod_name}")

  monkeypatch.setattr(importlib, "import_module", _broken_import)
  with pytest.raises(ImportError, match="broken extension"):
    _load_cpp_bindings("visqol_jax")


def test_live_cpp_error_rejection_parity():
  # Too-short WAVs fail in both C++ (VisqolManager & VisqolApi) and visqol_jax.
  for short_rel in (
      "testdata/short_duration/1_sample/guitar48_stereo_1_sample.wav",
      "testdata/short_duration/10_sample/guitar48_stereo_10_sample.wav",
      "testdata/short_duration/100_sample/guitar48_stereo_100_sample.wav",
      "testdata/short_duration/1000_sample/guitar48_stereo_1000_sample.wav",
      "testdata/short_duration/10000_sample/guitar48_stereo_10000_sample.wav",
  ):
    short_wav = _find_repo_file(short_rel)
    mono, sr = _read_pcm16_wav_mono(short_wav)
    opts = visqol_jax.VisqolOptions(use_speech_scoring=False)
    with pytest.raises(ValueError):
      visqol_jax.visqol(mono, mono, sr, opts)
    with pytest.raises(Exception):
      _run_cpp_manager(short_wav, short_wav, opts)
    with pytest.raises(Exception):
      _run_cpp_api(mono, mono, sr, opts)

  # Post-alignment unscorable degraded clip fails in C++ and yields NaN in JAX.
  idx300 = jnp.arange(300, dtype=jnp.float32)
  burst = jnp.sin(
      jnp.float32(2.0 * math.pi) * 600.0 * idx300 / jnp.float32(16000.0)
  ) * _hann(300)
  ref_unsc = jnp.zeros((20000,), dtype=jnp.float32).at[100:400].set(burst)
  deg_unsc = jnp.zeros((1400,), dtype=jnp.float32).at[600:900].set(burst)
  sp_opts = visqol_jax.VisqolOptions(use_speech_scoring=True)
  jax_unsc = visqol_jax.visqol(ref_unsc, deg_unsc, 16000, sp_opts)
  assert math.isnan(float(jax_unsc.moslqo)) and int(
      jax_unsc.patch_sims.num_patches
  ) == 0
  with pytest.raises(Exception):
    _run_cpp_api(ref_unsc, deg_unsc, 16000, sp_opts)

  # Non-48 kHz audio without override fails in both VisqolApi and visqol_jax.
  wav_44k = _find_repo_file(
      "testdata/non_48k_sample_rate/guitar48_stereo_44100Hz.wav"
  )
  mono_44k, sr_44k = _read_pcm16_wav_mono(wav_44k)
  strict_audio_opts = visqol_jax.VisqolOptions(
      use_speech_scoring=False, allow_unsupported_sample_rates=False
  )
  with pytest.raises(ValueError, match="48000 Hz"):
    visqol_jax.visqol(mono_44k, mono_44k, sr_44k, strict_audio_opts)
  with pytest.raises(Exception):
    _run_cpp_api(mono_44k, mono_44k, sr_44k, strict_audio_opts)
