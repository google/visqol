"""Anti-cheating audit and unseen-seed C++ differential verification."""

import array
import ast
import functools
import gzip
import importlib
import importlib.resources
import importlib.util
import io
import math
import os
import pathlib
import struct
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import visqol_jax
from visqol_jax import lattice

_SPEECH_MODEL_REL = (
    "model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_"
    "ep2400_train1_7_raw.tflite"
)
_AUDIO_MODEL_REL = "model/libsvm_nu_svr_model.txt"
_FORBIDDEN_STDLIB = frozenset({
    "ctypes",
    "http",
    "marshal",
    "multiprocessing",
    "pickle",
    "socket",
    "subprocess",
    "urllib",
})
_FORBIDDEN_ATTRS = frozenset({
    "compile",
    "dlopen",
    "eval",
    "exec",
    "ffi_call",
    "host_callback",
    "io_callback",
    "pure_callback",
    "python_callback",
})


@functools.cache
def _find_repo_file(relpath: str) -> pathlib.Path:
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


def _load_cpp_bindings() -> tuple[Any, Any]:
  root_pkg = visqol_jax.__name__.rpartition(".")[0]
  if root_pkg:
    return (
        importlib.import_module(f"{root_pkg}.python.visqol_lib_py"),
        importlib.import_module(f"{root_pkg}.src.proto.visqol_config_pb2"),
    )
  for lib_mod, cfg_mod, sim_mod in (
      (
          "visqol.visqol_lib_py",
          "visqol.pb2.visqol_config_pb2",
          "visqol.pb2.similarity_result_pb2",
      ),
      ("python.visqol_lib_py", "visqol_config_pb2", "similarity_result_pb2"),
  ):
    try:
      found = importlib.util.find_spec(lib_mod) is not None
    except ModuleNotFoundError:
      found = False
    if found:
      importlib.import_module(sim_mod)
      return importlib.import_module(lib_mod), importlib.import_module(cfg_mod)
  pytest.skip("C++ visqol_lib_py bindings are not installed")
  raise RuntimeError("unreachable")


def _lcg_floats(
    seed: int, n: int, lo: float = -1.0, hi: float = 1.0
) -> list[float]:
  state = seed & 0xFFFFFFFF
  scale = (hi - lo) / float(0x100000000)
  out = [0.0] * n
  for i in range(n):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    out[i] = lo + state * scale
  return out


def _write_pcm16_wav_mono(
    path: pathlib.Path, samples: jax.Array, sample_rate: int
) -> jax.Array:
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


def _synth_pair(
    seed: int, sample_rate: int, dur_s: float
) -> tuple[jax.Array, jax.Array]:
  params = _lcg_floats(seed, 16, 0.0, 1.0)
  n = int(round(sample_rate * dur_s))
  t = jnp.arange(n, dtype=jnp.float32) / jnp.float32(sample_rate)
  two_pi = jnp.float32(2.0 * math.pi)
  f0 = jnp.float32(95.0 + 115.0 * params[0])
  phase_mod = jnp.float32(0.4 + 0.6 * params[1]) * jnp.sin(
      two_pi * jnp.float32(1.5 + 1.5 * params[2]) * t
  )
  # Broadband harmonic stack (h=1..20) + unvoiced excitation across ERB bands.
  harmonics = jnp.arange(1, 21, dtype=jnp.float32)[:, None]
  nyq_mask = (harmonics * f0 < jnp.float32(0.46 * sample_rate)).astype(
      jnp.float32
  )
  weights = nyq_mask / jnp.power(
      harmonics, jnp.float32(0.75 + 0.25 * params[3])
  )
  voiced = jnp.sum(
      weights
      * jnp.sin(two_pi * harmonics * f0 * t[None, :] + harmonics * phase_mod),
      axis=0,
  )
  voiced = voiced / jnp.maximum(jnp.max(jnp.abs(voiced)), jnp.float32(1e-6))
  unvoiced = jnp.asarray(
      _lcg_floats(seed ^ 0x517CC1B7, n, -0.08, 0.08), dtype=jnp.float32
  )
  carrier = jnp.float32(0.48) * voiced + unvoiced
  # Smooth syllable-like envelope so RMS-VAD and patch selection vary per seed.
  syllable = jnp.maximum(
      jnp.sin(
          two_pi * jnp.float32(1.1 + 1.2 * params[5]) * t
          + jnp.float32(params[6] * 3.0)
      ),
      jnp.float32(0.0),
  )
  gate = jnp.float32(0.015) + jnp.float32(0.75) * (syllable * syllable)
  ref = gate * carrier
  lag = int(round(-450 + 1100 * params[7]))
  gain = jnp.float32(0.45 + 1.1 * params[8])
  if lag >= 0:
    shifted = jnp.concatenate(
        [jnp.zeros((lag,), dtype=jnp.float32), ref[: -lag or None]]
    )
  else:
    shifted = jnp.concatenate(
        [ref[-lag:], jnp.zeros((-lag,), dtype=jnp.float32)]
    )
  # Comb filter + additive noise so band similarities vary realistically.
  comb = jnp.float32(0.84) * shifted + jnp.float32(0.16) * jnp.roll(shifted, 3)
  noise_amp = 0.008 + 0.03 * params[9]
  noise = jnp.asarray(
      _lcg_floats(seed ^ 0x9E3779B9, n, -noise_amp, noise_amp),
      dtype=jnp.float32,
  )
  deg = gain * comb + noise
  return jnp.clip(ref, -0.98, 0.98), jnp.clip(deg, -0.98, 0.98)


def test_no_cheating_imports_ast_weights_and_callback_free_hlo(
    monkeypatch: pytest.MonkeyPatch,
):
  pkg_root = importlib.resources.files(visqol_jax)
  allowed_roots = (sys.stdlib_module_names - _FORBIDDEN_STDLIB) | {
      "jax",
      "numpy",
      visqol_jax.__name__.split(".")[0],
  }
  runtime_files = [
      entry
      for entry in pkg_root.iterdir()
      if entry.name.endswith(".py") and not entry.name.endswith("_test.py")
  ]
  assert len(runtime_files) >= 9
  for entry in runtime_files:
    tree = ast.parse(entry.read_text(encoding="utf-8"), filename=entry.name)
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          assert (
              alias.name.split(".")[0] in allowed_roots
          ), f"{entry.name}: {alias.name}"
          assert "visqol_lib_py" not in alias.name
      elif isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        if mod:
          assert mod.split(".")[0] in allowed_roots, f"{entry.name}: {mod}"
        assert "visqol_lib_py" not in mod
        if node.module and not mod.startswith("jax"):
          for alias in node.names:
            assert alias.name != "visqol_lib_py"
      elif isinstance(node, ast.Name):
        assert node.id not in _FORBIDDEN_ATTRS, f"{entry.name}: {node.id}"
      elif isinstance(node, ast.Attribute):
        assert node.attr not in _FORBIDDEN_ATTRS, f"{entry.name}: {node.attr}"

  # Bundled SVR text is byte-identical to model/libsvm_nu_svr_model.txt.
  models_dir = pkg_root.joinpath("_model_data")
  svr_bytes = gzip.decompress(
      models_dir.joinpath("svr_model.txt.gz").read_bytes()
  )
  assert svr_bytes == _find_repo_file(_AUDIO_MODEL_REL).read_bytes()

  # Every float32 weight slice in speech_lattice.npz exists in .tflite.
  tflite_bytes = _find_repo_file(_SPEECH_MODEL_REL).read_bytes()
  with np.load(
      io.BytesIO(models_dir.joinpath("speech_lattice.npz").read_bytes()),
      allow_pickle=False,
  ) as lat:
    for key in ("in_keypoints", "in_inv_lengths", "lattice_weights"):
      for row in np.asarray(lat[key], dtype=np.float32):
        assert row.astype("<f4").tobytes() in tflite_bytes, key
    for row in np.asarray(lat["lattice_calibrator_weights"], dtype=np.float32):
      for cal in row:
        assert cal.astype("<f4").tobytes() in tflite_bytes
    for key in (
        "ensemble_weights",
        "out_keypoints",
        "out_inv_lengths",
        "out_weights",
    ):
      vals = np.asarray(lat[key], dtype=np.float32)
      assert vals.astype("<f4").tobytes() in tflite_bytes, key

  # Lowered StableHLO has zero host callbacks and runs with C++ blocked.
  r16, d16 = _synth_pair(7, 16000, 1.5)
  r48, d48 = _synth_pair(8, 48000, 1.5)
  for fn, r, d, sr in (
      (visqol_jax.visqol_speech, r16, d16, 16000),
      (visqol_jax.visqol_audio, r48, d48, 48000),
  ):
    hlo = (
        jax.jit(fn, static_argnames=("sample_rate",))
        .lower(r, d, sample_rate=sr)
        .as_text()
    )
    for forbidden in (
        "python_callback",
        "pure_callback",
        "io_callback",
        "host_callback",
        "ffi",
    ):
      assert forbidden not in hlo
  root_pkg = visqol_jax.__name__.rpartition(".")[0]
  for mod_name in (
      "visqol.visqol_lib_py",
      "python.visqol_lib_py",
      f"{root_pkg}.python.visqol_lib_py" if root_pkg else "visqol_lib_py",
  ):
    monkeypatch.setitem(sys.modules, mod_name, None)
  assert math.isfinite(float(visqol_jax.visqol_speech(r16, d16, 16000).moslqo))
  assert math.isfinite(float(visqol_jax.visqol_audio(r48, d48, 48000).moslqo))


_NUM_PATCHES_PATH = (
    jax.tree_util.GetAttrKey("patch_sims"),
    jax.tree_util.GetAttrKey("num_patches"),
)


def _assert_allclose(
    actual: jax.typing.ArrayLike,
    expected: Any,
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


def _assert_all_fields_parity(
    jax_res: visqol_jax.VisqolResult,
    cpp_res: Any,
    opts: visqol_jax.VisqolOptions,
    sample_rate: int,
    label: str,
    *,
    score_tol: float,
) -> None:
  """Check every VisqolResult field against C++, mirroring _assert_parity."""
  for path, leaf in jax.tree_util.tree_leaves_with_path(jax_res):
    want_dt = jnp.int32 if path == _NUM_PATCHES_PATH else jnp.float32
    assert isinstance(leaf, jax.Array) and leaf.dtype == want_dt

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
  if opts.use_speech_scoring and opts.use_lattice_model:
    # Direct TFLite-vs-JAX lattice check on identical features (<=2e-4).
    lat_mos = float(
        lattice.predict_lattice(
            list(cpp_res.fvnsim),
            list(cpp_res.fvnsim10),
            list(cpp_res.fstdnsim),
            list(cpp_res.fvdegenergy),
        )
    )
    if cpp_res.vnsim >= 0.15:
      assert abs(lat_mos - cpp_res.moslqo) <= 2e-4, label

  assert abs(float(jax_res.moslqo) - cpp_res.moslqo) <= score_tol, (
      f"{label}: moslqo {float(jax_res.moslqo)} vs {cpp_res.moslqo}"
  )
  assert abs(float(jax_res.vnsim) - cpp_res.vnsim) <= 1.5e-2, (
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


def test_live_cpp_random_unseen_waveforms_parity(tmp_path: pathlib.Path):
  visqol_lib_py, visqol_config_pb2 = _load_cpp_bindings()
  env_seed = os.environ.get("VISQOL_PARITY_SEED")
  seeds = (
      tuple(int(env_seed) + k for k in range(4))
      if env_seed is not None
      else (101, 202, 303, 404)
  )
  speech_model = str(_find_repo_file(_SPEECH_MODEL_REL))
  audio_model = str(_find_repo_file(_AUDIO_MODEL_REL))

  for seed in seeds:
    for sr, dur_s, opts_list, model_path in (
        (
            16000,
            2.4,
            (
                visqol_jax.VisqolOptions(
                    use_speech_scoring=True, use_lattice_model=True
                ),
                visqol_jax.VisqolOptions(
                    use_speech_scoring=True, use_lattice_model=False
                ),
            ),
            speech_model,
        ),
        (
            48000,
            2.2,
            (visqol_jax.VisqolOptions(use_speech_scoring=False),),
            audio_model,
        ),
    ):
      ref_raw, deg_raw = _synth_pair(seed * 31 + sr, sr, dur_s)
      ref_wav = tmp_path / f"ref_{seed}_{sr}.wav"
      deg_wav = tmp_path / f"deg_{seed}_{sr}.wav"
      ref_q = _write_pcm16_wav_mono(ref_wav, ref_raw, sr)
      deg_q = _write_pcm16_wav_mono(deg_wav, deg_raw, sr)

      for opts in opts_list:
        jax_q = visqol_jax.visqol(ref_q, deg_q, sr, opts)
        jax_raw = visqol_jax.visqol(ref_raw, deg_raw, sr, opts)
        mgr = visqol_lib_py.VisqolManager()
        mgr.Init(
            visqol_lib_py.FilePath(model_path),
            opts.use_speech_scoring,
            opts.use_unscaled_speech_mos_mapping,
            opts.search_window_radius,
            opts.use_lattice_model,
            opts.disable_global_alignment,
            opts.disable_realignment,
        )
        mgr_res = mgr.Run(
            visqol_lib_py.FilePath(str(ref_wav)),
            visqol_lib_py.FilePath(str(deg_wav)),
        )

        cfg = visqol_config_pb2.VisqolConfig()
        cfg.audio.sample_rate = sr
        cfg.options.use_speech_scoring = opts.use_speech_scoring
        cfg.options.use_lattice_model = opts.use_lattice_model
        cfg.options.svr_model_path = model_path
        api = visqol_lib_py.VisqolApi()
        api.Create(cfg)
        api_res = api.Measure(ref_raw.tolist(), deg_raw.tolist())

        # Direct mapper check <=2e-4; 4-patch f32 Slice trunc <=0.075.
        tol = 1e-2 if not opts.use_speech_scoring else 7.5e-2
        for tag, jax_res, cpp_res in (
            ("mgr", jax_q, mgr_res),
            ("api", jax_raw, api_res),
        ):
          label = f"seed={seed} sr={sr} lattice={opts.use_lattice_model} {tag}"
          _assert_all_fields_parity(
              jax_res, cpp_res, opts, sr, label, score_tol=tol
          )
