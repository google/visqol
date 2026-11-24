"""Unit tests for visqol.py public API, aggregation helpers, and validation."""

import gzip
import importlib
import logging
import math
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import visqol_jax
from visqol_jax import selector

visqol_mod = importlib.import_module(f"{visqol_jax.__name__}.visqol")
alter_for_similarity_extremes = visqol_mod.alter_for_similarity_extremes
calc_per_patch_freq_band_quantile = visqol_mod.calc_per_patch_freq_band_quantile
calc_per_patch_mean_freq_band_stddevs = (
    visqol_mod.calc_per_patch_mean_freq_band_stddevs
)

_NUM_PATCHES_PATH = (
    jax.tree_util.GetAttrKey("patch_sims"),
    jax.tree_util.GetAttrKey("num_patches"),
)


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
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
  a = jnp.asarray(actual)
  b = jnp.asarray(expected, dtype=a.dtype)
  if b.ndim != 0:
    assert a.shape == b.shape
  assert bool(
      jnp.allclose(a, b, rtol=rtol, atol=atol)
  ), f"max diff {float(jnp.max(jnp.abs(a - b)))}"


def _assert_array_equal(
    actual: jax.typing.ArrayLike, expected: jax.typing.ArrayLike
) -> None:
  a = jnp.asarray(actual)
  b = jnp.asarray(expected, dtype=a.dtype)
  if b.ndim == 0:
    assert bool(jnp.all(a == b))
  else:
    assert a.shape == b.shape and bool(jnp.array_equal(a, b))


def _hann(n: int) -> jax.Array:
  idx = jnp.arange(n, dtype=jnp.float32)
  return jnp.float32(0.5) - jnp.float32(0.5) * jnp.cos(
      jnp.float32(2.0 * math.pi) * idx / jnp.float32(n - 1)
  )


def _synth_speech_pair(
    n: int = 12800, seed: int = 1001
) -> tuple[jax.Array, jax.Array]:
  """Deterministic 16 kHz speech-band reference and noisy degraded waveforms."""
  t = jnp.arange(n, dtype=jnp.float32) / jnp.float32(16000.0)
  two_pi = jnp.float32(2.0 * math.pi)
  env = jnp.float32(0.35) + jnp.float32(0.65) * jnp.exp(
      jnp.float32(-0.5) * ((t - jnp.float32(0.35)) / jnp.float32(0.16)) ** 2
  )
  ref = env * (
      jnp.float32(0.30) * jnp.sin(two_pi * (210.0 * t + 110.0 * t * t))
      + jnp.float32(0.18) * jnp.sin(two_pi * (680.0 * t + 75.0 * t * t))
      + jnp.float32(0.08) * jnp.sin(two_pi * 1900.0 * t)
  )
  deg = ref + _lcg_f32(seed, n, -0.04, 0.04)
  return ref, deg


def _synth_audio_pair(
    n: int = 48000, seed: int = 2002
) -> tuple[jax.Array, jax.Array]:
  """Deterministic 48 kHz audio reference and noisy degraded waveforms."""
  t = jnp.arange(n, dtype=jnp.float32) / jnp.float32(48000.0)
  two_pi = jnp.float32(2.0 * math.pi)
  env = jnp.float32(0.4) + jnp.float32(0.6) * jnp.sin(two_pi * 2.5 * t) ** 2
  ref = env * (
      jnp.float32(0.22) * jnp.sin(two_pi * 440.0 * t)
      + jnp.float32(0.15) * jnp.sin(two_pi * 1320.0 * t)
      + jnp.float32(0.10) * jnp.sin(two_pi * 5500.0 * t)
  ) + _lcg_f32(seed, n, -0.02, 0.02)
  deg = ref + _lcg_f32(seed + 1, n, -0.045, 0.045)
  return ref, deg


def _write_constant_lattice(
    path: pathlib.Path,
    value: float,
    *,
    num_bands: int = 21,
    num_lattices: int = 1,
    compress: bool = True,
) -> pathlib.Path:
  nf = 4 * num_bands + 1
  prev_mtime = path.stat().st_mtime_ns if path.exists() else None
  arrays = {
      "in_keypoints": np.zeros((nf, 1), dtype=np.float32),
      "in_inv_lengths": np.ones((nf, 1), dtype=np.float32),
      "in_has_missing": np.zeros((nf,), dtype=np.bool_),
      "in_missing_value": np.zeros((nf,), dtype=np.float32),
      "lattice_feature_indices": np.zeros((num_lattices, 1), dtype=np.int32),
      "lattice_calibrator_weights": np.zeros(
          (num_lattices, 1, 2), dtype=np.float32
      ),
      "lattice_calibrator_missing_output": np.zeros(
          (num_lattices, 1), dtype=np.float32
      ),
      "lattice_weights": np.zeros((num_lattices, 2), dtype=np.float32),
      "ensemble_weights": np.zeros((num_lattices,), dtype=np.float32),
      "out_keypoints": np.zeros((1,), dtype=np.float32),
      "out_inv_lengths": np.ones((1,), dtype=np.float32),
      "out_weights": np.array([value, 0.0], dtype=np.float32),
  }
  with path.open("wb") as fh:
    if compress:
      np.savez_compressed(fh, **arrays)
    else:
      np.savez(fh, **arrays)
  if prev_mtime is not None:
    os.utime(path, ns=(prev_mtime + 1_000_000, prev_mtime + 1_000_000))
  return path


def _write_constant_svr(
    path: pathlib.Path, value: float, n_feat: int
) -> pathlib.Path:
  prev_mtime = path.stat().st_mtime_ns if path.exists() else None
  feats = " ".join(f"{i + 1}:0.0" for i in range(n_feat))
  raw = (
      "svm_type nu_svr\nkernel_type rbf\ngamma 0.01\nnr_class 2\n"
      f"total_sv 1\nrho {-value!r}\nSV\n0.0 {feats}\n"
  ).encode("utf-8")
  if path.suffix == ".gz":
    raw = gzip.compress(raw, compresslevel=6, mtime=0)
  path.write_bytes(raw)
  if prev_mtime is not None:
    os.utime(path, ns=(prev_mtime + 1_000_000, prev_mtime + 1_000_000))
  return path


def test_aggregation_helpers_quantile_stddev_and_extremes():
  # Exact reference test from tests/visqol_test.cc (100 patches, q=0.10).
  n, q = 100, 0.10
  expected = jnp.array([5.5, 15.5], dtype=jnp.float32)
  fbm = jnp.stack(
      [
          jnp.arange(n, 0, -1, dtype=jnp.float32),
          jnp.arange(11, n + 11, dtype=jnp.float32),
      ],
      axis=1,
  )
  _assert_allclose(
      calc_per_patch_freq_band_quantile(fbm, quantile=q), expected, atol=1e-6
  )
  padded = jnp.concatenate([fbm, jnp.full((30, 2), jnp.nan)], axis=0)
  got_pad = jax.jit(
      calc_per_patch_freq_band_quantile,
      static_argnames=("quantile",),
  )(padded, num_patches=jnp.int32(n), quantile=q)
  _assert_allclose(got_pad, expected, atol=1e-6)
  _assert_allclose(
      calc_per_patch_freq_band_quantile(
          padded, num_patches=jnp.int32(1), quantile=q
      ),
      fbm[0],
      atol=1e-6,
  )
  _assert_array_equal(
      calc_per_patch_freq_band_quantile(
          padded, num_patches=jnp.int32(0), quantile=q
      ),
      [0.0, 0.0],
  )
  # Clamping num_patches > p_max avoids dividing a P-row sum by > P.
  _assert_allclose(
      calc_per_patch_freq_band_quantile(
          fbm[:5], num_patches=jnp.int32(100), quantile=1.0
      ),
      jnp.mean(fbm[:5], axis=0),
      atol=1e-6,
  )
  # Active NaNs propagate as NaN rather than sorting behind +inf padding.
  nan_q = calc_per_patch_freq_band_quantile(
      jnp.array([[jnp.nan, 1.0], [jnp.nan, 3.0], [0.0, 0.0]], jnp.float32),
      num_patches=jnp.int32(2),
      quantile=0.5,
  )
  assert math.isnan(float(nan_q[0])) and float(nan_q[1]) == 1.0
  for bad_q in (0.0, 1.1, float("nan"), True, 1.0 + 0.0j):
    with pytest.raises(ValueError, match="quantile"):
      calc_per_patch_freq_band_quantile(
          fbm, quantile=bad_q  # type: ignore[arg-type]
      )
  for bad_np in (-1, True, 1.5):
    with pytest.raises(ValueError, match="num_patches"):
      calc_per_patch_freq_band_quantile(
          fbm, num_patches=bad_np  # type: ignore[arg-type]
      )
  with pytest.raises(ValueError, match="bool"):
    calc_per_patch_freq_band_quantile(jnp.ones((4, 2), dtype=jnp.bool_))
  for bad_fbm in (
      jnp.ones((4,), dtype=jnp.float32),
      jnp.zeros((0, 2), dtype=jnp.float32),
      jnp.zeros((2, 0), dtype=jnp.float32),
  ):
    with pytest.raises(ValueError, match="freq_band_means"):
      calc_per_patch_freq_band_quantile(bad_fbm)

  sim = selector.PatchSimilarityResult(
      similarity=jnp.array([0.3, 0.3, 0.0], dtype=jnp.float32),
      freq_band_means=jnp.array([[0.3, 0.7], [0.3, 0.7], [jnp.nan, jnp.nan]]),
      freq_band_stddevs=jnp.array([[0.0, 0.0], [0.0, 0.0], [jnp.nan, jnp.nan]]),
      freq_band_deg_energy=jnp.zeros((3, 2), dtype=jnp.float32),
      ref_patch_start_time=jnp.array([0.0, 0.4, jnp.nan], dtype=jnp.float32),
      ref_patch_end_time=jnp.array([0.4, 0.8, jnp.nan], dtype=jnp.float32),
      deg_patch_start_time=jnp.zeros((3,), dtype=jnp.float32),
      deg_patch_end_time=jnp.zeros((3,), dtype=jnp.float32),
      num_patches=jnp.int32(2),
  )
  _assert_array_equal(
      calc_per_patch_mean_freq_band_stddevs(sim, 0.02), [0.0, 0.0]
  )
  _assert_array_equal(
      calc_per_patch_mean_freq_band_stddevs(
          sim._replace(num_patches=jnp.int32(0)), 0.02
      ),
      [0.0, 0.0],
  )
  # Constant 50-patch bands with mixed 20/21-frame durations avoid cancellation.
  p50 = 50
  starts50 = jnp.arange(p50, dtype=jnp.float32) * jnp.float32(0.4)
  ends50 = starts50 + jnp.where(
      (jnp.arange(p50) % 3) == 0, jnp.float32(0.42), jnp.float32(0.4)
  )
  sim50 = selector.PatchSimilarityResult(
      similarity=jnp.full((p50,), 0.7, dtype=jnp.float32),
      freq_band_means=jnp.broadcast_to(
          jnp.array([[0.7, 0.9999]], dtype=jnp.float32), (p50, 2)
      ),
      freq_band_stddevs=jnp.zeros((p50, 2), dtype=jnp.float32),
      freq_band_deg_energy=jnp.zeros((p50, 2), dtype=jnp.float32),
      ref_patch_start_time=starts50,
      ref_patch_end_time=ends50,
      deg_patch_start_time=starts50,
      deg_patch_end_time=ends50,
      num_patches=jnp.int32(p50),
  )
  _assert_allclose(
      calc_per_patch_mean_freq_band_stddevs(sim50, 0.02), [0.0, 0.0], atol=1e-6
  )
  # Unequal patch frame counts (20 vs 21) match C++ unweighted-fvnsim formula.
  sim_uneq = selector.PatchSimilarityResult(
      similarity=jnp.array([10.0, 0.0], dtype=jnp.float32),
      freq_band_means=jnp.array([[10.0], [0.0]], dtype=jnp.float32),
      freq_band_stddevs=jnp.zeros((2, 1), dtype=jnp.float32),
      freq_band_deg_energy=jnp.zeros((2, 1), dtype=jnp.float32),
      ref_patch_start_time=jnp.array([0.0, 0.4], dtype=jnp.float32),
      ref_patch_end_time=jnp.array([0.4, 0.82], dtype=jnp.float32),
      deg_patch_start_time=jnp.zeros((2,), dtype=jnp.float32),
      deg_patch_end_time=jnp.zeros((2,), dtype=jnp.float32),
      num_patches=jnp.int32(2),
  )
  _assert_allclose(
      calc_per_patch_mean_freq_band_stddevs(sim_uneq, 0.02),
      [math.sqrt(975.0 / 40.0)],
      atol=1e-5,
  )
  # Active NaNs propagate instead of turning into 0.0.
  nan_std = calc_per_patch_mean_freq_band_stddevs(
      sim_uneq._replace(
          freq_band_means=jnp.array([[jnp.nan], [0.5]], dtype=jnp.float32)
      ),
      0.02,
  )
  assert math.isnan(float(nan_std[0]))
  # Non-trivial pooled sample stddev matches explicit sample std across frames.
  p0 = jnp.array([0.2, 0.4, 0.6, 0.8], dtype=jnp.float32)
  p1 = jnp.array([0.5, 0.7, 0.9, 1.0], dtype=jnp.float32)
  sim_var = selector.PatchSimilarityResult(
      similarity=jnp.array([float(jnp.mean(p0)), float(jnp.mean(p1))]),
      freq_band_means=jnp.array([[jnp.mean(p0)], [jnp.mean(p1)]]),
      freq_band_stddevs=jnp.array(
          [[jnp.std(p0, ddof=1)], [jnp.std(p1, ddof=1)]]
      ),
      freq_band_deg_energy=jnp.zeros((2, 1), dtype=jnp.float32),
      ref_patch_start_time=jnp.array([0.0, 0.25], dtype=jnp.float32),
      ref_patch_end_time=jnp.array([0.25, 0.50], dtype=jnp.float32),
      deg_patch_start_time=jnp.zeros((2,), dtype=jnp.float32),
      deg_patch_end_time=jnp.zeros((2,), dtype=jnp.float32),
      num_patches=jnp.int32(2),
  )
  want_std = float(jnp.std(jnp.concatenate([p0, p1]), ddof=1))
  _assert_allclose(
      calc_per_patch_mean_freq_band_stddevs(sim_var, 0.0625),
      [want_std],
      atol=1e-6,
  )
  for bad_fd in (0.0, -0.02, float("nan"), True):
    with pytest.raises(ValueError, match="frame_duration"):
      calc_per_patch_mean_freq_band_stddevs(sim_var, bad_fd)
  with pytest.raises(ValueError, match="PatchSimilarityResult"):
    calc_per_patch_mean_freq_band_stddevs(None, 0.02)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="bool"):
    calc_per_patch_mean_freq_band_stddevs(
        sim_var._replace(similarity=jnp.array([True, False])), 0.02
    )
  with pytest.raises(ValueError, match="real-valued"):
    calc_per_patch_mean_freq_band_stddevs(
        sim_var._replace(similarity=jnp.zeros((2,), dtype=jnp.complex64)), 0.02
    )
  for field_name, bad_sim in (
      ("similarity", sim_var._replace(similarity=jnp.zeros((3,), jnp.float32))),
      (
          "freq_band_stddevs",
          sim_var._replace(freq_band_stddevs=jnp.zeros((2, 2), jnp.float32)),
      ),
      (
          "ref_patch_start_time",
          sim_var._replace(ref_patch_start_time=jnp.zeros((3,), jnp.float32)),
      ),
      (
          "deg_patch_end_time",
          sim_var._replace(deg_patch_end_time=jnp.zeros((3,), jnp.float32)),
      ),
  ):
    with pytest.raises(ValueError, match=rf"sim_results\.{field_name}"):
      calc_per_patch_mean_freq_band_stddevs(bad_sim, 0.02)

  low_ext = alter_for_similarity_extremes(jnp.float32(0.1499), jnp.float32(3.5))
  high_ext = alter_for_similarity_extremes(jnp.float32(0.15), jnp.float32(3.5))
  assert float(low_ext) == 1.0
  assert float(high_ext) == 3.5
  traced_ext = jax.jit(lambda v, m: alter_for_similarity_extremes([v], [m]))(
      jnp.float32(0.14), jnp.float32(3.5)
  )
  _assert_array_equal(traced_ext, [1.0])
  with pytest.raises(ValueError, match="bool"):
    alter_for_similarity_extremes(True, 3.5)


def test_public_api_pytree_modes_and_options():
  assert {
      "PatchSimilarityResult",
      "VisqolOptions",
      "VisqolResult",
      "__version__",
      "visqol",
      "visqol_audio",
      "visqol_speech",
  }.issubset(visqol_jax.__all__)
  assert visqol_jax.PatchSimilarityResult is selector.PatchSimilarityResult

  sp_ref, sp_deg = _synth_speech_pair()
  au_ref, _ = _synth_audio_pair()

  sp_res = visqol_jax.visqol_speech(sp_ref, sp_deg, 16000)
  for path, leaf in jax.tree_util.tree_leaves_with_path(sp_res):
    want_dtype = jnp.int32 if path == _NUM_PATCHES_PATH else jnp.float32
    assert isinstance(leaf, jax.Array) and leaf.dtype == want_dtype
    assert bool(jnp.all(jnp.isfinite(leaf)))
  leaves, treedef = jax.tree_util.tree_flatten(sp_res)
  assert len(leaves) == 17
  unflat = jax.tree_util.tree_unflatten(treedef, leaves)
  assert isinstance(unflat, visqol_jax.VisqolResult)
  assert isinstance(unflat.patch_sims, visqol_jax.PatchSimilarityResult)

  # Identical clips match conformance.h constants in speech exp and audio SVR.
  sp_id_scaled = visqol_jax.visqol_speech(
      sp_ref, sp_ref, 16000, use_lattice_model=False
  )
  assert float(sp_id_scaled.vnsim) == pytest.approx(1.0, abs=1e-6)
  assert float(sp_id_scaled.moslqo) == pytest.approx(5.0, abs=1e-5)
  sp_id_unscaled = visqol_jax.visqol_speech(
      sp_ref,
      sp_ref,
      16000,
      use_lattice_model=False,
      use_unscaled_speech_mos_mapping=True,
  )
  assert float(sp_id_unscaled.moslqo) == pytest.approx(
      4.015861169223797, abs=1e-5
  )
  au_id = visqol_jax.visqol_audio(au_ref, au_ref, 48000)
  assert float(au_id.vnsim) == pytest.approx(1.0, abs=1e-6)
  assert float(au_id.moslqo) == pytest.approx(4.732101253042348, abs=2e-5)
  au_default = visqol_jax.visqol(au_ref, au_ref, 48000)
  assert float(au_default.moslqo) == pytest.approx(
      float(au_id.moslqo), abs=1e-6
  )

  # Dissimilar clips (vnsim < 0.15) clamp to 1.0 unless skip_mos_mapping=True.
  t16 = jnp.arange(12800, dtype=jnp.float32) / jnp.float32(16000.0)
  two_pi = jnp.float32(2.0 * math.pi)
  low_tone = jnp.float32(0.4) * jnp.sin(two_pi * 120.0 * t16)
  gate = (jnp.sin(two_pi * 3.0 * t16) > jnp.float32(0.0)).astype(jnp.float32)
  high_chop = jnp.float32(0.4) * jnp.sin(two_pi * 7200.0 * t16) * gate
  dis_res = visqol_jax.visqol_speech(low_tone, high_chop, 16000)
  assert float(dis_res.vnsim) < 0.15
  assert float(dis_res.moslqo) == 1.0
  dis_skip = visqol_jax.visqol_speech(
      low_tone, high_chop, 16000, skip_mos_mapping=True
  )
  assert math.isnan(float(dis_skip.moslqo))
  assert float(dis_skip.vnsim) == pytest.approx(float(dis_res.vnsim), abs=1e-6)

  # Alignment and search window options take effect as expected.
  sp_delayed = jnp.concatenate([jnp.zeros((800,), jnp.float32), sp_ref[:-800]])
  lagged = visqol_jax.visqol_speech(sp_ref, sp_delayed, 16000)
  assert float(lagged.alignment_lag_s) == pytest.approx(-0.05, abs=1e-3)
  no_glob = visqol_jax.visqol_speech(
      sp_ref,
      sp_delayed,
      16000,
      disable_global_alignment=True,
      disable_realignment=True,
      search_window_radius=0,
  )
  assert float(no_glob.alignment_lag_s) == 0.0
  assert math.isfinite(float(no_glob.moslqo))


def test_custom_svr_and_lattice_live_reload_and_ignored_modes(
    tmp_path: pathlib.Path,
):
  sp_ref, sp_deg = _synth_speech_pair()
  au_ref, au_deg = _synth_audio_pair()

  # Overwriting .txt and .txt.gz SVR files updates visqol_audio cleanly.
  svr_path = _write_constant_svr(tmp_path / "live_svr.txt", 2.5, n_feat=31)
  r1 = visqol_jax.visqol_audio(au_ref, au_deg, 48000, svr_model_path=svr_path)
  assert float(r1.moslqo) == pytest.approx(2.5, abs=1e-6)

  _write_constant_svr(svr_path, 3.75, n_feat=32)
  opts = visqol_jax.VisqolOptions(
      use_speech_scoring=False, svr_model_path=svr_path
  )
  r2 = visqol_jax.visqol(au_ref, au_deg, 48000, opts)
  assert float(r2.moslqo) == pytest.approx(3.75, abs=1e-6)

  svr_gz = _write_constant_svr(tmp_path / "live_svr.txt.gz", 4.125, n_feat=32)
  r_gz = visqol_jax.visqol_audio(au_ref, au_deg, 48000, svr_model_path=svr_gz)
  assert float(r_gz.moslqo) == pytest.approx(4.125, abs=1e-6)

  # 0-indexed legacy SVR support vectors (as in cpp_model.txt) match libsvm.
  zero_svr = tmp_path / "zero_idx_svr.txt"
  zero_svr.write_text(
      "svm_type nu_svr\nkernel_type rbf\ngamma 0.5\nnr_class 2\n"
      "total_sv 1\nrho -2.0\nSV\n1.0 0:2.0 1:1.0\n",
      encoding="utf-8",
  )
  r_zero = visqol_jax.visqol_audio(
      au_ref, au_ref, 48000, svr_model_path=zero_svr
  )
  # Identical fvnsim == 1.0 gives sum = 2^2 + (1 - 1)^2 + 31 * 1^2 = 35.0.
  want_zero_mos = math.exp(-0.5 * 35.0) + 2.0
  assert float(r_zero.moslqo) == pytest.approx(want_zero_mos, abs=1e-6)

  # Custom speech lattice reload across uncompressed and compressed .npz files.
  lat_path = _write_constant_lattice(
      tmp_path / "live_lat.npz", 2.25, compress=False
  )
  l1 = visqol_jax.visqol_speech(sp_ref, sp_deg, 16000, svr_model_path=lat_path)
  assert float(l1.moslqo) == pytest.approx(2.25, abs=1e-6)

  _write_constant_lattice(lat_path, 3.875, compress=False)
  lat_opts = visqol_jax.VisqolOptions(
      use_speech_scoring=True, svr_model_path=lat_path
  )
  l2 = visqol_jax.visqol(sp_ref, sp_deg, 16000, lat_opts)
  assert float(l2.moslqo) == pytest.approx(3.875, abs=1e-6)

  _write_constant_lattice(lat_path, 4.25, num_lattices=2, compress=False)
  l3 = visqol_jax.visqol_speech(sp_ref, sp_deg, 16000, svr_model_path=lat_path)
  assert float(l3.moslqo) == pytest.approx(4.25, abs=1e-6)

  lat_cmp = _write_constant_lattice(
      tmp_path / "live_lat_cmp.npz", 3.125, compress=True
  )
  l_cmp = visqol_jax.visqol_speech(
      sp_ref, sp_deg, 16000, svr_model_path=lat_cmp
  )
  assert float(l_cmp.moslqo) == pytest.approx(3.125, abs=1e-6)

  bad_bands = _write_constant_lattice(
      tmp_path / "bad_bands.npz", 3.0, num_bands=4
  )
  with pytest.raises(ValueError, match="21 bands"):
    visqol_jax.visqol_speech(sp_ref, sp_deg, 16000, svr_model_path=bad_bands)

  fake_tflite = tmp_path / "other.tflite"
  fake_tflite.write_bytes(b"TFL3-not-supported")
  with pytest.raises(ValueError, match=r"export_lattice_npz\.py"):
    visqol_jax.visqol_speech(sp_ref, sp_deg, 16000, svr_model_path=fake_tflite)

  # Speech exp mode and skip_mos_mapping ignore nonexistent svr_model_path.
  missing_path = tmp_path / "does_not_exist.npz"
  exp_ref = visqol_jax.visqol_speech(
      sp_ref, sp_deg, 16000, use_lattice_model=False
  )
  exp_ign = visqol_jax.visqol_speech(
      sp_ref,
      sp_deg,
      16000,
      use_lattice_model=False,
      svr_model_path=missing_path,
  )
  assert float(exp_ign.moslqo) == pytest.approx(float(exp_ref.moslqo), abs=1e-6)
  skip_sp = visqol_jax.visqol_speech(
      sp_ref, sp_deg, 16000, skip_mos_mapping=True, svr_model_path=missing_path
  )
  assert math.isnan(float(skip_sp.moslqo))
  skip_au = visqol_jax.visqol_audio(
      au_ref, au_deg, 48000, skip_mos_mapping=True, svr_model_path=missing_path
  )
  assert math.isnan(float(skip_au.moslqo))


def test_unscorable_clips_return_nan_scores_not_fake_one():
  # Negative global lag shrinks aligned degraded below window_size (0 cols).
  two_pi = jnp.float32(2.0 * math.pi)
  for sr, mode_fn, ref_len, deg_len, shift in (
      (16000, visqol_jax.visqol_speech, 20000, 1400, 500),
      (48000, visqol_jax.visqol_audio, 60000, 4200, 1500),
  ):
    burst_n = 300 if sr == 16000 else 900
    idx = jnp.arange(burst_n, dtype=jnp.float32)
    burst = jnp.sin(two_pi * 600.0 * idx / jnp.float32(sr)) * _hann(burst_n)
    ref = (
        jnp.zeros((ref_len,), dtype=jnp.float32)
        .at[100 : 100 + burst_n]
        .set(burst)
    )
    deg = (
        jnp.zeros((deg_len,), dtype=jnp.float32)
        .at[100 + shift : 100 + shift + burst_n]
        .set(burst)
    )
    res = mode_fn(ref, deg, sr)
    assert math.isfinite(float(res.alignment_lag_s))
    assert float(res.alignment_lag_s) < 0.0
    assert bool(jnp.all(jnp.isfinite(res.center_freq_bands)))
    assert math.isnan(float(res.moslqo)) and math.isnan(float(res.vnsim))
    assert bool(jnp.all(jnp.isnan(res.fvnsim)))
    assert bool(jnp.all(jnp.isnan(res.fvnsim10)))
    assert bool(jnp.all(jnp.isnan(res.fstdnsim)))
    assert bool(jnp.all(jnp.isnan(res.fvdegenergy)))
    assert res.patch_sims.num_patches.dtype == jnp.int32
    assert int(res.patch_sims.num_patches) == 0
    for arr in res.patch_sims[:-1]:
      assert arr.dtype == jnp.float32
      assert bool(jnp.all(jnp.isfinite(arr)))
      _assert_array_equal(arr, 0.0)

  # Mixed [scorable, unscorable] batch under jit(vmap(...)) isolates NaNs.
  sp_ref, sp_deg = _synth_speech_pair(n=20000)
  idx300 = jnp.arange(300, dtype=jnp.float32)
  burst = jnp.sin(two_pi * 600.0 * idx300 / jnp.float32(16000.0)) * _hann(300)
  unsc_ref = jnp.zeros((20000,), dtype=jnp.float32).at[100:400].set(burst)
  sp_deg_short = sp_deg[:1400]
  unsc_deg_short = jnp.zeros((1400,), dtype=jnp.float32).at[600:900].set(burst)
  want_ok = visqol_jax.visqol_speech(sp_ref, sp_deg_short, 16000)
  batched = jax.jit(
      jax.vmap(lambda r, d: visqol_jax.visqol_speech(r, d, 16000))
  )(
      jnp.stack([sp_ref, unsc_ref]),
      jnp.stack([sp_deg_short, unsc_deg_short]),
  )
  assert float(batched.moslqo[0]) == pytest.approx(
      float(want_ok.moslqo), abs=1e-5
  )
  assert int(batched.patch_sims.num_patches[0]) > 0
  assert math.isnan(float(batched.moslqo[1])) and math.isnan(
      float(batched.vnsim[1])
  )
  assert int(batched.patch_sims.num_patches[1]) == 0
  _assert_array_equal(batched.patch_sims.similarity[1], 0.0)

  # Short degraded (< window_size) with positive lag grows past window_size.
  burst_n = 800
  idx800 = jnp.arange(burst_n, dtype=jnp.float32)
  burst800 = jnp.sin(
      two_pi * (450.0 + 300.0 * idx800 / jnp.float32(16000.0)) * idx800
      / jnp.float32(16000.0)
  ) * _hann(burst_n)
  delay = 4000
  ref_delayed = (
      jnp.zeros((24000,), dtype=jnp.float32)
      .at[delay : delay + burst_n]
      .set(burst800)
  )
  res_short_pos = visqol_jax.visqol_speech(ref_delayed, burst800, 16000)
  assert float(res_short_pos.alignment_lag_s) == pytest.approx(
      delay / 16000.0, abs=1e-3
  )
  assert int(res_short_pos.patch_sims.num_patches) > 0
  assert math.isfinite(float(res_short_pos.moslqo))
  # Short degraded staying < window_size after alignment returns NaN scores.
  res_short_zero = visqol_jax.visqol_speech(
      sp_ref, jnp.zeros((500,), dtype=jnp.float32), 16000
  )
  assert math.isnan(float(res_short_zero.moslqo))
  assert int(res_short_zero.patch_sims.num_patches) == 0


def _assert_nonfinite_result(res: visqol_jax.VisqolResult) -> None:
  assert math.isnan(float(res.moslqo)) and math.isnan(float(res.vnsim))
  assert math.isnan(float(res.alignment_lag_s))
  assert bool(jnp.all(jnp.isnan(res.fvnsim)))
  assert bool(jnp.all(jnp.isnan(res.fvnsim10)))
  assert bool(jnp.all(jnp.isnan(res.fstdnsim)))
  assert bool(jnp.all(jnp.isnan(res.fvdegenergy)))
  assert bool(jnp.all(jnp.isfinite(res.center_freq_bands)))
  assert int(res.patch_sims.num_patches) == 0
  for arr in res.patch_sims[:-1]:
    _assert_array_equal(arr, 0.0)


def test_nonfinite_inputs_return_nan_scores_and_lag():
  sp_ref, sp_deg = _synth_speech_pair()
  au_ref, au_deg = _synth_audio_pair()

  sp_all_nan = jnp.full_like(sp_ref, jnp.nan)
  sp_one_nan = sp_deg.at[123].set(jnp.nan)
  sp_inf = sp_deg.at[456].set(jnp.inf)
  sp_ninf = sp_ref.at[789].set(-jnp.inf)
  for r, d in (
      (sp_ref, sp_all_nan),
      (sp_all_nan, sp_all_nan),
      (sp_ref, sp_one_nan),
      (sp_one_nan, sp_deg),
      (sp_ref, sp_inf),
      (sp_ninf, sp_deg),
  ):
    _assert_nonfinite_result(visqol_jax.visqol_speech(r, d, 16000))
    _assert_nonfinite_result(
        visqol_jax.visqol_speech(r, d, 16000, use_lattice_model=False)
    )

  au_all_nan = jnp.full_like(au_ref, jnp.nan)
  au_one_nan = au_deg.at[321].set(jnp.nan)
  au_inf = au_deg.at[654].set(jnp.inf)
  for r, d in (
      (au_ref, au_all_nan),
      (au_all_nan, au_all_nan),
      (au_ref, au_one_nan),
      (au_ref, au_inf),
  ):
    _assert_nonfinite_result(visqol_jax.visqol_audio(r, d, 48000))

  # jit and vmap isolate non-finite batch elements to NaN scores and lag.
  for fn, ref, deg, bad_deg in (
      (
          lambda r, d: visqol_jax.visqol_speech(r, d, 16000),
          sp_ref,
          sp_deg,
          sp_one_nan,
      ),
      (
          lambda r, d: visqol_jax.visqol_audio(r, d, 48000),
          au_ref,
          au_deg,
          au_inf,
      ),
  ):
    want = fn(ref, deg)
    jit_bad = jax.jit(fn)(ref, bad_deg)
    _assert_nonfinite_result(jit_bad)
    batched = jax.jit(jax.vmap(fn))(
        jnp.stack([ref, ref]), jnp.stack([deg, bad_deg])
    )
    assert float(batched.moslqo[0]) == pytest.approx(
        float(want.moslqo), abs=1e-5
    )
    assert float(batched.alignment_lag_s[0]) == pytest.approx(
        float(want.alignment_lag_s), abs=1e-6
    )
    lane1 = jax.tree_util.tree_map(lambda a: a[1], batched)
    _assert_nonfinite_result(lane1)


def test_scalar_coercion_non_div4_sr_and_validation():
  sp_ref, _ = _synth_speech_pair()
  int_t = jnp.dtype(jnp.int32).type
  bool_t = jnp.dtype(jnp.bool_).type

  res_i32 = visqol_jax.visqol_speech(
      sp_ref,
      sp_ref,
      sample_rate=int_t(16000),  # type: ignore[arg-type]
      search_window_radius=int_t(60),  # type: ignore[arg-type]
  )
  assert math.isfinite(float(res_i32.moslqo))
  opts_bool = visqol_jax.VisqolOptions(
      use_speech_scoring=bool_t(True),  # type: ignore[arg-type]
      use_lattice_model=bool_t(False),  # type: ignore[arg-type]
      search_window_radius=int_t(60),  # type: ignore[arg-type]
      svr_model_path=pathlib.Path("/tmp/custom_model.txt"),
  )
  # pylint: disable-next=unidiomatic-typecheck
  assert type(opts_bool.use_speech_scoring) is bool
  assert opts_bool.use_speech_scoring
  assert not opts_bool.use_lattice_model
  # pylint: disable-next=unidiomatic-typecheck
  assert type(opts_bool.search_window_radius) is int
  assert opts_bool.search_window_radius == 60
  # pylint: disable-next=unidiomatic-typecheck
  assert type(opts_bool.svr_model_path) is str
  assert opts_bool.svr_model_path == "/tmp/custom_model.txt"

  # Non-multiple-of-4 window_size (11025 Hz -> ws=882, hop=220) runs cleanly.
  t11 = jnp.arange(22050, dtype=jnp.float32) / jnp.float32(11025.0)
  s11 = jnp.float32(0.3) * jnp.sin(jnp.float32(2.0 * math.pi) * 440.0 * t11)
  res11 = visqol_jax.visqol_speech(s11, s11, sample_rate=11025)
  assert math.isfinite(float(res11.moslqo))

  # Unsupported sample rate rejection and override.
  with pytest.raises(ValueError, match="48000 Hz"):
    visqol_jax.visqol_audio(sp_ref, sp_ref, sample_rate=16000)
  sig_16k_long = jnp.tile(sp_ref, 2)[:20000]
  res_override = visqol_jax.visqol_audio(
      sig_16k_long,
      sig_16k_long,
      sample_rate=16000,
      allow_unsupported_sample_rates=True,
  )
  assert res_override.fvnsim.shape == (32,) and math.isfinite(
      float(res_override.moslqo)
  )

  # Eager input and option validation.
  with pytest.raises(ValueError, match="non-empty 1D"):
    visqol_jax.visqol_speech(jnp.zeros((2, 16000), jnp.float32), sp_ref, 16000)
  with pytest.raises(ValueError, match="non-empty 1D"):
    visqol_jax.visqol_speech(sp_ref, jnp.zeros((0,), jnp.float32), 16000)
  with pytest.raises(ValueError, match="numeric"):
    visqol_jax.visqol_speech(jnp.ones((16000,), dtype=jnp.bool_), sp_ref, 16000)
  with pytest.raises(ValueError, match="numeric"):
    visqol_jax.visqol_speech("invalid", sp_ref, 16000)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="real-valued"):
    visqol_jax.visqol_speech(sp_ref.astype(jnp.complex64), sp_ref, 16000)
  with pytest.raises(ValueError, match="sample_rate"):
    visqol_jax.visqol_speech(sp_ref, sp_ref, 0)
  with pytest.raises(ValueError, match="sample_rate"):
    visqol_jax.visqol_speech(sp_ref, sp_ref, True)
  with pytest.raises(ValueError, match="at least 100 Hz"):
    visqol_jax.visqol_speech(sp_ref, sp_ref, 50)
  with pytest.raises(ValueError, match="at least 100 Hz"):
    visqol_jax.visqol_audio(
        sp_ref, sp_ref, 50, allow_unsupported_sample_rates=True
    )
  with pytest.raises(ValueError, match="degraded signal"):
    visqol_jax.visqol_speech(
        sp_ref,
        jnp.zeros((500,), jnp.float32),
        16000,
        disable_global_alignment=True,
    )
  # 9920 samples at 16 kHz yields 28 frames (min_frames - 1 = 29 - 1).
  with pytest.raises(
      ValueError,
      match=r"Reference spectrum size \(28\) smaller than minimum patch size"
      r" \(29\)",
  ):
    visqol_jax.visqol_speech(
        jnp.zeros((9920,), jnp.float32), jnp.zeros((9920,), jnp.float32), 16000
    )
  with pytest.raises(ValueError, match="minimum patch size"):
    visqol_jax.visqol_audio(
        jnp.zeros((10000,), jnp.float32),
        jnp.zeros((10000,), jnp.float32),
        48000,
    )
  # At fs=120 (hop=2, 2.5 samples/VAD frame), 426 samples needs 509 VAD samples.
  with pytest.raises(
      ValueError,
      match=r"Too few samples \(426\) in reference signal for VAD "
      r"\(must be at least 509\)",
  ):
    visqol_jax.visqol_speech(
        jnp.zeros((426,), jnp.float32), jnp.zeros((426,), jnp.float32), 120
    )
  with pytest.raises(ValueError, match="search_window_radius"):
    visqol_jax.VisqolOptions(search_window_radius=-1)
  with pytest.raises(ValueError, match="search_window_radius"):
    visqol_jax.VisqolOptions(search_window_radius=True)
  with pytest.raises(ValueError, match="bool"):
    visqol_jax.VisqolOptions(
        use_speech_scoring="false"  # type: ignore[arg-type]
    )
  with pytest.raises(ValueError, match="non-empty"):
    visqol_jax.VisqolOptions(use_speech_scoring=False, svr_model_path="")
  with pytest.raises(ValueError, match="non-empty"):
    visqol_jax.VisqolOptions(svr_model_path="bad\x00path")
  with pytest.raises(ValueError, match="non-empty"):
    visqol_jax.VisqolOptions(
        use_speech_scoring=True,
        svr_model_path=b"/tmp/a",  # type: ignore[arg-type]
    )
  with pytest.raises(ValueError, match="non-empty"):
    visqol_jax.VisqolOptions(svr_model_path=123)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="VisqolOptions"):
    visqol_jax.visqol(
        sp_ref, sp_ref, 16000, "invalid"  # type: ignore[arg-type]
    )


def test_validation_warnings_via_logging(caplog: pytest.LogCaptureFixture):
  sp_ref, _ = _synth_speech_pair()
  au_ref, au_deg = _synth_audio_pair()
  visqol_logger = visqol_jax.visqol.__module__

  # Duration mismatch > 1.0 s warns; exact 1.0 s boundary does not warn.
  with caplog.at_level(logging.WARNING, logger=visqol_logger):
    caplog.clear()
    visqol_jax.visqol_speech(
        jnp.concatenate([sp_ref, jnp.zeros((16000,), dtype=jnp.float32)]),
        sp_ref,
        sample_rate=16000,
    )
    assert not caplog.records

    visqol_jax.visqol_speech(
        jnp.concatenate([sp_ref, jnp.zeros((16800,), dtype=jnp.float32)]),
        sp_ref,
        sample_rate=16000,
    )
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "Mismatch in duration" in caplog.records[0].getMessage()

  # Speech mode with fs > 16000 warns; fs <= 16000 does not warn.
  with caplog.at_level(logging.WARNING, logger=visqol_logger):
    caplog.clear()
    sig_24k = jnp.tile(sp_ref, 2)[:24000]
    visqol_jax.visqol_speech(sig_24k, sig_24k, sample_rate=24000)
    assert len(caplog.records) == 1
    assert "above 16 kHz" in caplog.records[0].getMessage()

    caplog.clear()
    visqol_jax.visqol_speech(sp_ref, sp_ref, sample_rate=8000)
    assert not caplog.records

  # Audio mode with fs != 48000 and allow_unsupported_sample_rates=True warns.
  with caplog.at_level(logging.WARNING, logger=visqol_logger):
    caplog.clear()
    sig_16k_long = jnp.tile(sp_ref, 2)[:20000]
    visqol_jax.visqol_audio(
        sig_16k_long,
        sig_16k_long,
        sample_rate=16000,
        allow_unsupported_sample_rates=True,
    )
    assert len(caplog.records) == 1
    assert "48 kHz" in caplog.records[0].getMessage()

    caplog.clear()
    visqol_jax.visqol_audio(au_ref, au_deg, sample_rate=48000)
    assert not caplog.records
    with pytest.raises(ValueError, match="48000 Hz"):
      visqol_jax.visqol_audio(sig_16k_long, sig_16k_long, sample_rate=16000)
    assert not caplog.records
