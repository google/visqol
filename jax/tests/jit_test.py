"""Static-shape JIT zero-recompilation, outer jax.jit, vmap, and x64 tests."""

from collections.abc import Callable, Sequence
import dataclasses
import gzip
import math
import os
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import visqol_jax
from visqol_jax import lattice
from visqol_jax import patches
from visqol_jax import spectrogram
from visqol_jax import svr
from visqol_jax.visqol import _visqol_jit

_NUM_PATCHES_PATH = (
    jax.tree_util.GetAttrKey("patch_sims"),
    jax.tree_util.GetAttrKey("num_patches"),
)
_FFT_LINE = re.compile(r"^\s*%\S+ = stablehlo\.fft .*?:(?P<types>.*)$")
_WIDE_FFT_TYPE = re.compile(r"(?:<|x|\b)(?:f64|c128)\b")


def _lcg_f32(seed: int, n: int, lo: float = -1.0, hi: float = 1.0) -> jax.Array:
  state = seed & 0xFFFFFFFF
  vals = [0.0] * n
  scale = (hi - lo) / float(0x100000000)
  for i in range(n):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    vals[i] = lo + state * scale
  return jnp.asarray(vals, dtype=jnp.float32)


def _assert_allclose(
    actual: jax.typing.ArrayLike,
    expected: jax.typing.ArrayLike,
    *,
    atol: float = 1e-5,
    err_msg: str = "",
) -> None:
  a = jnp.asarray(actual)
  b = jnp.asarray(expected, dtype=a.dtype)
  if b.ndim != 0:
    assert a.shape == b.shape, err_msg
  assert bool(
      jnp.allclose(a, b, rtol=1e-5, atol=atol)
  ), f"{err_msg}: max diff {float(jnp.max(jnp.abs(a - b)))}"


def _speech_clip_pairs() -> list[tuple[str, jax.Array, jax.Array]]:
  """Six same-length 16 kHz clips spanning distinct lags, VAD, and silence."""
  sr, n = 16000, 24064
  t = jnp.arange(n, dtype=jnp.float32) / jnp.float32(sr)
  two_pi = jnp.float32(2.0 * math.pi)
  env = jnp.exp(jnp.float32(-0.5) * ((t - 0.55) / 0.25) ** 2) + jnp.float32(
      0.55
  ) * jnp.exp(jnp.float32(-0.5) * ((t - 1.15) / 0.14) ** 2)
  base = env * (
      jnp.float32(0.45) * jnp.sin(two_pi * (180.0 * t + 140.0 * t * t))
      + jnp.float32(0.2) * jnp.sin(two_pi * (620.0 * t + 90.0 * t * t))
  )
  pos_lag = jnp.concatenate([base[800:], jnp.zeros((800,), jnp.float32)])
  neg_lag = jnp.concatenate([jnp.zeros((800,), jnp.float32), base[:-800]])
  sparse = jnp.where(t < jnp.float32(0.45), base, jnp.float32(0.0))
  silence = jnp.zeros((n,), dtype=jnp.float32)
  return [
      ("noisy", base, base + _lcg_f32(101, n, -0.09, 0.09)),
      ("pos_lag", base, pos_lag),
      ("neg_lag", base, neg_lag),
      ("sparse_vad", sparse, sparse + _lcg_f32(102, n, -0.03, 0.03)),
      ("both_silence", silence, silence),
      ("deg_silence", base, silence),
  ]


def _audio_clip_pairs() -> list[tuple[str, jax.Array, jax.Array]]:
  """Six same-length 48 kHz clips spanning distinct lags, noise, and silence."""
  sr, n = 48000, 48128
  t = jnp.arange(n, dtype=jnp.float32) / jnp.float32(sr)
  two_pi = jnp.float32(2.0 * math.pi)
  env = jnp.exp(jnp.float32(-0.5) * ((t - 0.35) / 0.16) ** 2) + jnp.float32(
      0.65
  ) * jnp.exp(jnp.float32(-0.5) * ((t - 0.72) / 0.10) ** 2)
  base = env * (
      jnp.float32(0.45) * jnp.sin(two_pi * (300.0 * t + 450.0 * t * t))
      + jnp.float32(0.25) * jnp.sin(two_pi * (1100.0 * t + 300.0 * t * t))
      + _lcg_f32(202, n, -0.15, 0.15)
  )
  pos_lag = jnp.concatenate([base[2400:], jnp.zeros((2400,), jnp.float32)])
  neg_lag = jnp.concatenate([jnp.zeros((2400,), jnp.float32), base[:-2400]])
  silence = jnp.zeros((n,), dtype=jnp.float32)
  return [
      ("identical", base, base),
      ("pos_lag", base, pos_lag),
      ("neg_lag", base, neg_lag),
      ("noisy", base, base + _lcg_f32(203, n, -0.12, 0.12)),
      ("both_silence", silence, silence),
      ("deg_silence", base, silence),
  ]


def _assert_f32_finite(result: visqol_jax.VisqolResult) -> None:
  for path, leaf in jax.tree_util.tree_leaves_with_path(result):
    assert isinstance(leaf, jax.Array)
    want_dt = jnp.int32 if path == _NUM_PATCHES_PATH else jnp.float32
    assert leaf.dtype == want_dt
    assert bool(jnp.all(jnp.isfinite(leaf)))


def _jit_cache_size(fn: jax.stages.Wrapped) -> int:
  return int(getattr(fn, "_cache_size")())


def _run_and_assert_single_compile(
    pairs: Sequence[tuple[str, jax.Array, jax.Array]],
    call: Callable[
        [jax.typing.ArrayLike, jax.typing.ArrayLike], visqol_jax.VisqolResult
    ],
) -> dict[str, visqol_jax.VisqolResult]:
  before = _jit_cache_size(_visqol_jit)
  results: dict[str, visqol_jax.VisqolResult] = {}
  for idx, (name, ref, deg) in enumerate(pairs):
    res = call(ref, deg)
    jax.block_until_ready(res)
    _assert_f32_finite(res)
    results[name] = res
    after = _jit_cache_size(_visqol_jit)
    if idx == 0:
      assert after == before + 1
    else:
      assert after == before + 1, f"recompiled on {name}"
  return results


def _make_caller(mode: str, use_lattice_model: bool) -> tuple[
    list[tuple[str, jax.Array, jax.Array]],
    Callable[
        [jax.typing.ArrayLike, jax.typing.ArrayLike], visqol_jax.VisqolResult
    ],
]:
  if mode == "speech":
    return _speech_clip_pairs(), lambda r, d: visqol_jax.visqol_speech(
        r, d, 16000, use_lattice_model=use_lattice_model
    )
  return _audio_clip_pairs(), lambda r, d: visqol_jax.visqol_audio(r, d, 48000)


@pytest.mark.parametrize("use_lattice_model", [True, False])
def test_visqol_speech_zero_recompilations_across_same_length_clips(
    use_lattice_model: bool,
):
  sr = 16000
  pairs, call = _make_caller("speech", use_lattice_model)

  refs_by_name = {name: ref for name, ref, _ in pairs}
  ws, hop, _ = spectrogram.analysis_window(sr)
  spec_len = int(
      spectrogram.num_spectrogram_frames(
          refs_by_name["noisy"].shape[0], ws, hop
      )
  )
  _, n_full = patches.create_vad_patch_indices(
      spec_len, refs_by_name["noisy"], sr
  )
  _, n_sparse = patches.create_vad_patch_indices(
      spec_len, refs_by_name["sparse_vad"], sr
  )
  assert int(n_sparse) < int(n_full)

  results = _run_and_assert_single_compile(pairs, call)
  assert float(results["pos_lag"].alignment_lag_s) > 0.02
  assert float(results["neg_lag"].alignment_lag_s) < -0.02
  assert float(results["noisy"].alignment_lag_s) == 0.0
  assert float(results["deg_silence"].moslqo) == pytest.approx(1.0, abs=1e-6)


def test_visqol_audio_zero_recompilations_across_same_length_clips():
  pairs, call = _make_caller("audio", False)
  results = _run_and_assert_single_compile(pairs, call)
  assert float(results["pos_lag"].alignment_lag_s) > 0.02
  assert float(results["neg_lag"].alignment_lag_s) < -0.02
  assert float(results["identical"].moslqo) > float(results["noisy"].moslqo)
  assert float(results["deg_silence"].moslqo) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("mode", "use_lattice_model"),
    [("speech", True), ("audio", False)],
)
def test_outer_jit_and_vmap_trace_once_and_match_direct_calls(
    mode: str, use_lattice_model: bool
):
  pairs, call = _make_caller(mode, use_lattice_model)
  mos_atol = 5e-4 if (mode == "speech" and use_lattice_model) else 1e-5

  outer_traces = 0

  @jax.jit
  def outer(r: jax.Array, d: jax.Array) -> visqol_jax.VisqolResult:
    nonlocal outer_traces
    outer_traces += 1
    return call(r, d)

  subset = [pairs[0], pairs[1], pairs[2], pairs[-1]]
  wants: dict[str, visqol_jax.VisqolResult] = {}
  for name, ref, deg in subset:
    got = outer(ref, deg)
    want = call(ref, deg)
    wants[name] = want
    assert outer_traces == 1, f"outer jit retraced on {name}"
    for (path, got_leaf), (want_path, want_leaf) in zip(
        jax.tree_util.tree_leaves_with_path(got),
        jax.tree_util.tree_leaves_with_path(want),
        strict=True,
    ):
      assert path == want_path
      assert got_leaf.dtype == want_leaf.dtype
      key = jax.tree_util.keystr(path)
      if path == _NUM_PATCHES_PATH:
        assert int(got_leaf) == int(want_leaf), f"outer/{mode}/{name}{key}"
      else:
        _assert_allclose(
            got_leaf, want_leaf, atol=1e-6, err_msg=f"outer/{mode}/{name}{key}"
        )

  vmap_traces = 0

  @jax.jit
  def batched_fn(refs: jax.Array, degs: jax.Array) -> visqol_jax.VisqolResult:
    nonlocal vmap_traces
    vmap_traces += 1
    return jax.vmap(call)(refs, degs)

  for batch in (subset[:2], subset[2:]):
    refs = jnp.stack([r for _, r, _ in batch])
    degs = jnp.stack([d for _, _, d in batch])
    got = batched_fn(refs, degs)
    assert vmap_traces == 1
    for i, (name, _, _) in enumerate(batch):
      for (path, got_leaf), (want_path, want_leaf) in zip(
          jax.tree_util.tree_leaves_with_path(got),
          jax.tree_util.tree_leaves_with_path(wants[name]),
          strict=True,
      ):
        assert path == want_path
        assert got_leaf.dtype == want_leaf.dtype
        key = jax.tree_util.keystr(path)
        if path == _NUM_PATCHES_PATH:
          assert int(got_leaf[i]) == int(want_leaf), f"vmap/{mode}/{name}{key}"
        else:
          # Near-zero per-patch variance sqrt(O(1e-10)) and 60-lattice slopes.
          tol_map = {
              ".moslqo": mos_atol,
              ".fstdnsim": 2e-4,
              ".patch_sims.freq_band_stddevs": 5e-4,
          }
          _assert_allclose(
              got_leaf[i],
              want_leaf,
              atol=tol_map.get(key, 1e-5),
              err_msg=f"vmap/{mode}/{name}{key}",
          )


def test_custom_models_live_reload_zero_recompile_vmap_and_tracer_safety(
    tmp_path: pathlib.Path,
):
  pairs = _speech_clip_pairs()
  nf = 4 * spectrogram.SPEECH_MODE_NUM_BANDS + 1
  lat_path = tmp_path / "custom_lat.npz"
  svr_path = tmp_path / "custom_svr.txt.gz"

  def write_const_lat(val: float) -> None:
    prev = lat_path.stat().st_mtime_ns if lat_path.exists() else None
    with lat_path.open("wb") as fh:
      np.savez_compressed(
          fh,
          in_keypoints=np.zeros((nf, 1), dtype=np.float32),
          in_inv_lengths=np.ones((nf, 1), dtype=np.float32),
          in_has_missing=np.zeros((nf,), dtype=np.bool_),
          in_missing_value=np.zeros((nf,), dtype=np.float32),
          lattice_feature_indices=np.zeros((1, 1), dtype=np.int32),
          lattice_calibrator_weights=np.zeros((1, 1, 2), dtype=np.float32),
          lattice_calibrator_missing_output=np.zeros((1, 1), dtype=np.float32),
          lattice_weights=np.zeros((1, 2), dtype=np.float32),
          ensemble_weights=np.zeros((1,), dtype=np.float32),
          out_keypoints=np.zeros((1,), dtype=np.float32),
          out_inv_lengths=np.ones((1,), dtype=np.float32),
          out_weights=np.array([val, 0.0], dtype=np.float32),
      )
    if prev is not None:
      os.utime(lat_path, ns=(prev + 1_000_000, prev + 1_000_000))

  feats = " ".join(f"{i + 1}:0.5" for i in range(32))
  svr_raw = (
      "svm_type nu_svr\nkernel_type rbf\ngamma 0.01\nnr_class 2\n"
      f"total_sv 1\nrho -2.75\nSV\n0.0 {feats}\n"
  ).encode("utf-8")
  svr_path.write_bytes(gzip.compress(svr_raw, compresslevel=6, mtime=0))

  _, ref0, deg0 = pairs[0]
  _, ref1, deg1 = pairs[1]
  write_const_lat(2.5)
  before = _jit_cache_size(_visqol_jit)
  r1 = visqol_jax.visqol_speech(ref0, deg0, 16000, svr_model_path=lat_path)
  assert _jit_cache_size(_visqol_jit) == before + 1
  assert float(r1.moslqo) == pytest.approx(2.5, abs=1e-6)

  # Same-shape overwrite reloads values with zero _visqol_jit recompilation.
  write_const_lat(3.75)
  r2 = visqol_jax.visqol_speech(ref0, deg0, 16000, svr_model_path=lat_path)
  assert _jit_cache_size(_visqol_jit) == before + 1
  assert float(r2.moslqo) == pytest.approx(3.75, abs=1e-6)

  refs = jnp.stack([ref0, ref1])
  degs = jnp.stack([deg0, deg1])
  batched = jax.jit(
      jax.vmap(
          lambda r, d: visqol_jax.visqol_speech(
              r, d, 16000, svr_model_path=lat_path
          )
      )
  )(refs, degs)
  _assert_allclose(batched.moslqo, [3.75, 3.75], atol=1e-6)

  _, au_ref, au_deg = _audio_clip_pairs()[0]
  au_res = jax.jit(
      lambda r, d: visqol_jax.visqol_audio(r, d, 48000, svr_model_path=svr_path)
  )(au_ref, au_deg)
  assert float(au_res.moslqo) == pytest.approx(2.75, abs=1e-6)

  for obj in (
      lattice.load_lattice_model(),
      lattice.load_lattice_model(lat_path),
      svr.load_svr_model(),
      svr.load_svr_model(svr_path),
  ):
    for field in dataclasses.fields(obj):
      val = getattr(obj, field.name)
      assert not isinstance(val, (jax.Array, jax.core.Tracer))


def test_x64_float64_inputs_produce_float32_leaves_and_zero_f64_ffts():
  def _check_fft_ir(hlo: str) -> tuple[int, bool]:
    fft_types = [
        m.group("types")
        for line in hlo.splitlines()
        if (m := _FFT_LINE.match(line)) is not None
    ]
    has_wide = any(_WIDE_FFT_TYPE.search(t) is not None for t in fft_types)
    return len(fft_types), has_wide

  with jax.enable_x64(True):
    # Sanity-check that the detector flags an actual float64/complex128 FFT.
    s64 = jax.ShapeDtypeStruct((64,), jnp.float64)
    bad_hlo = jax.jit(lambda a: jnp.fft.rfft(a, 64)).lower(s64).as_text()
    bad_count, bad_wide = _check_fft_ir(bad_hlo)
    assert bad_count == 1 and bad_wide

    r16 = jnp.ones((12800,), dtype=jnp.float64)
    r48 = jnp.ones((48000,), dtype=jnp.float64)
    assert r16.dtype == jnp.float64 and r48.dtype == jnp.float64

    for fn, sig in (
        (lambda r, d: visqol_jax.visqol_speech(r, d, 16000), r16),
        (
            lambda r, d: visqol_jax.visqol_speech(
                r, d, 16000, use_lattice_model=False
            ),
            r16,
        ),
        (lambda r, d: visqol_jax.visqol_audio(r, d, 48000), r48),
    ):
      res = fn(sig, sig)
      _assert_f32_finite(res)
      hlo = jax.jit(fn).lower(sig, sig).as_text()
      fft_count, has_wide = _check_fft_ir(hlo)
      assert fft_count > 0 and not has_wide
