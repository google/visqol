"""Tests for nsim.py against C++ ViSQOL v3.3.3."""

import math
import re

import jax
import jax.numpy as jnp
import pytest

from visqol_jax import nsim

_CPP_CONV_TOLERANCE = 0.001
_FLOAT_WIDTH = re.compile(r"(?:<|x|\b)f(?P<width>16|32|64)\b")
_NP_BOOL_TRUE = jnp.dtype(jnp.bool_).type(True)


def _col_major_2d(
    vals: tuple[float, ...], rows: int, cols: int
) -> tuple[tuple[float, ...], ...]:
  return tuple(
      tuple(vals[c * rows + r] for c in range(cols)) for r in range(rows)
  )


# Column-major AMatrix vectors from tests/convolution_2d_test.cc.
_CPP_WINDOW = _col_major_2d(
    (
        0.0113033910173052, 0.0838251475442633, 0.0113033910173052,
        0.0838251475442633, 0.619485845753726, 0.0838251475442633,
        0.0113033910173052, 0.0838251475442633, 0.0113033910173052,
    ),
    3,
    3,
)
_CPP_INPUT = _col_major_2d(
    (
        40.0392, 43.3409, 39.5270, 41.1731, 41.3591,
        42.6852, 45.2083, 45.7769, 39.9689, 43.6190,
        41.0119, 40.4244, 41.5932, 43.6027, 42.6204,
        43.0624, 42.2610, 42.4725, 43.4258, 42.9079,
    ),
    5,
    4,
)
_CPP_EXPECTED = _col_major_2d(
    (
        40.6634, 42.8407, 40.6395, 41.0129, 41.5407,
        42.4677, 44.2760, 44.2031, 41.2263, 42.9752,
        41.3784, 41.2656, 42.1388, 43.0366, 42.8042,
        42.7613, 42.1817, 42.4590, 43.2709, 42.9377,
    ),
    5,
    4,
)


def _assert_allclose(
    actual: jax.typing.ArrayLike,
    expected: jax.typing.ArrayLike,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
  a = jnp.asarray(actual)
  b = jnp.asarray(expected, dtype=a.dtype)
  if b.ndim != 0:
    assert a.shape == b.shape
  assert bool(jnp.allclose(a, b, rtol=rtol, atol=atol)), (
      f"max diff {float(jnp.max(jnp.abs(a - b)))}"
  )


def _assert_array_equal(
    actual: jax.typing.ArrayLike, expected: jax.typing.ArrayLike
) -> None:
  a = jnp.asarray(actual)
  b = jnp.asarray(expected, dtype=a.dtype)
  if b.ndim == 0:
    assert bool(jnp.all(a == b))
  else:
    assert a.shape == b.shape and bool(jnp.array_equal(a, b))


def _to_2d_f64(mat: jax.typing.ArrayLike) -> list[list[float]]:
  if isinstance(mat, jax.Array):
    return mat.tolist()
  return [[float(v) for v in row] for row in mat]


def _cpp_valid_2d_conv_with_boundary_ref(
    fir_filter: jax.typing.ArrayLike, input_matrix: jax.typing.ArrayLike
) -> list[list[float]]:
  """64-bit Python port of C++ Convolution2D::Valid2DConvWithBoundary."""
  filt = _to_2d_f64(fir_filter)
  inp = _to_2d_f64(input_matrix)
  rows, cols = len(inp), len(inp[0])
  padded = [[inp[0][0], *inp[0], inp[0][-1]]]
  for r in inp:
    padded.append([r[0], *r, r[-1]])
  padded.append([inp[-1][0], *inp[-1], inp[-1][-1]])

  i_r_c, i_c_c = rows + 2, cols + 2
  f_r_c, f_c_c = len(filt), len(filt[0])
  o_r_c = max(0, i_r_c - f_r_c + 1)
  o_c_c = max(0, i_c_c - f_c_c + 1)
  out = [[0.0] * o_c_c for _ in range(o_r_c)]
  for o_row in range(o_r_c):
    for o_col in range(o_c_c):
      acc = 0.0
      filt_idx = f_r_c * f_c_c - 1
      for f_col in range(f_c_c):
        for f_row in range(f_r_c):
          fr, fc = filt_idx % f_r_c, filt_idx // f_r_c
          acc += padded[f_row + o_row][f_col + o_col] * filt[fr][fc]
          filt_idx -= 1
      out[o_row][o_col] = acc
  return out


def _cpp_nsim_ref(
    ref_patch: jax.typing.ArrayLike,
    deg_patch: jax.typing.ArrayLike,
    intensity_range: float = 1.0,
) -> tuple[float, list[float], list[float], list[float]]:
  """64-bit Python port of C++ MeasurePatchSimilarity."""
  ref = _to_2d_f64(ref_patch)
  deg = _to_2d_f64(deg_patch)
  h, w_cols = len(ref), len(ref[0])
  w = nsim.GAUSSIAN_WINDOW
  c1 = (0.01 * intensity_range) ** 2
  c3 = ((0.03 * intensity_range) ** 2) / 2.0

  mu_r = _cpp_valid_2d_conv_with_boundary_ref(w, ref)
  mu_d = _cpp_valid_2d_conv_with_boundary_ref(w, deg)
  ref_sq = [[ref[i][j] * ref[i][j] for j in range(w_cols)] for i in range(h)]
  deg_sq = [[deg[i][j] * deg[i][j] for j in range(w_cols)] for i in range(h)]
  ref_deg = [[ref[i][j] * deg[i][j] for j in range(w_cols)] for i in range(h)]
  conv_r_sq = _cpp_valid_2d_conv_with_boundary_ref(w, ref_sq)
  conv_d_sq = _cpp_valid_2d_conv_with_boundary_ref(w, deg_sq)
  conv_r_d = _cpp_valid_2d_conv_with_boundary_ref(w, ref_deg)

  fbm = [0.0] * h
  fstd = [0.0] * h
  fdeg = [0.0] * h
  for i in range(h):
    sim_row = [0.0] * w_cols
    for j in range(w_cols):
      mr, md = mu_r[i][j], mu_d[i][j]
      sr_sq = conv_r_sq[i][j] - mr * mr
      sd_sq = conv_d_sq[i][j] - md * md
      srd = conv_r_d[i][j] - mr * md
      inten = (2.0 * mr * md + c1) / (mr * mr + md * md + c1)
      prod = sr_sq * sd_sq
      denom = c3 if prod < 0.0 else math.sqrt(prod) + c3
      sim_row[j] = inten * ((srd + c3) / denom)
    fdeg[i] = sum(deg[i]) / w_cols
    row_mean = sum(sim_row) / w_cols
    fbm[i] = row_mean
    if w_cols >= 2:
      var = sum((v - row_mean) ** 2 for v in sim_row) / (w_cols - 1)
      fstd[i] = math.sqrt(max(var, 0.0))
  return sum(fbm) / h, fbm, fstd, fdeg


def test_valid_2d_conv_with_boundary_cpp_golden_flip_and_shapes():
  assert _CPP_WINDOW == nsim.GAUSSIAN_WINDOW
  got = nsim.valid_2d_conv_with_boundary(_CPP_WINDOW, _CPP_INPUT)
  assert got.dtype == jnp.float32
  assert got.shape == (5, 4)
  _assert_allclose(got, _CPP_EXPECTED, atol=_CPP_CONV_TOLERANCE, rtol=0.0)
  _assert_allclose(
      got,
      _cpp_valid_2d_conv_with_boundary_ref(_CPP_WINDOW, _CPP_INPUT),
      atol=1e-4,
      rtol=1e-5,
  )

  # Asymmetric kernel verifies 180-degree convolution flip on a center impulse.
  asym = jnp.array(
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float32
  )
  impulse = jnp.zeros((3, 3), dtype=jnp.float32).at[1, 1].set(1.0)
  _assert_allclose(
      nsim.valid_2d_conv_with_boundary(asym, impulse), asym, atol=1e-6
  )

  # Non-square <= 9 (2x3) and > 9 (3x4, _conv2d_general fallback) filters.
  k_in, k1, k2, k3 = jax.random.split(jax.random.key(19), 4)
  inp = jax.random.uniform(
      k_in, (6, 7), dtype=jnp.float32, minval=-5.0, maxval=25.0
  )
  for key, shape in ((k1, (2, 3)), (k2, (3, 4)), (k3, (4, 4))):
    filt = jax.random.normal(key, shape, dtype=jnp.float32)
    _assert_allclose(
        nsim.valid_2d_conv_with_boundary(filt, inp),
        _cpp_valid_2d_conv_with_boundary_ref(filt, inp),
        atol=1e-4,
        rtol=1e-5,
    )


def test_conv2d_stencil_general_parity_and_batched_valid_cols():
  asym = jnp.array(
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float32
  )
  batch = jax.random.normal(jax.random.key(7), (3, 5, 8), dtype=jnp.float32)

  for vc in (1, 2, 5, 8):
    dirty = batch.at[:, :, vc:].set(999.0)
    masked = nsim.valid_2d_conv_with_boundary(asym, dirty, valid_cols=vc)
    unmasked = nsim.valid_2d_conv_with_boundary(asym, batch[:, :, :vc])
    _assert_allclose(masked[:, :, :vc], unmasked, atol=1e-5, rtol=1e-5)

  # Out-of-range valid_cols (including wide uint32 and huge int) clamps safely.
  want_full = nsim.valid_2d_conv_with_boundary(asym, batch)
  for hi_vc in (99, 10**30, jnp.uint32(0xFFFFFFFF)):
    _assert_allclose(
        nsim.valid_2d_conv_with_boundary(asym, batch, valid_cols=hi_vc),
        want_full,
        atol=1e-6,
    )
  clamped_lo = nsim.valid_2d_conv_with_boundary(asym, batch, valid_cols=0)
  assert clamped_lo.shape == batch.shape and bool(
      jnp.all(jnp.isfinite(clamped_lo))
  )

  # Direct float32 CPU parity between _conv2d_stencil and _conv2d_general.
  for kern in (asym, asym[:2, :]):
    k_flipped = kern[::-1, ::-1]
    for arr in (batch[0], batch):
      # pylint: disable=protected-access
      padded = nsim._edge_pad_2d(arr)
      _assert_allclose(
          nsim._conv2d_stencil(padded, k_flipped),
          nsim._conv2d_general(padded, k_flipped),
          rtol=1e-5,
          atol=1e-5,
      )
      # pylint: enable=protected-access


def test_measure_patch_similarity_cpp_ref_masked_stddev_and_constants():
  k_ref, k_noise = jax.random.split(jax.random.key(123))
  ref = jax.random.uniform(
      k_ref, (21, 17), dtype=jnp.float32, minval=0.0, maxval=45.0
  )
  deg = ref + 2.5 * jax.random.normal(k_noise, (21, 17), dtype=jnp.float32)

  want_sim, want_fbm, want_fstd, want_fdeg = _cpp_nsim_ref(ref, deg)
  full = nsim.measure_patch_similarity(ref, deg)
  assert full.similarity.dtype == jnp.float32
  assert float(full.similarity) == pytest.approx(want_sim, abs=5e-4)
  _assert_allclose(full.freq_band_means, want_fbm, atol=5e-4, rtol=1e-4)
  _assert_allclose(full.freq_band_stddevs, want_fstd, atol=5e-4, rtol=1e-4)
  _assert_allclose(full.freq_band_deg_energy, want_fdeg, atol=1e-5, rtol=1e-5)

  # Unmasked single-column (W=1) and masked valid_cols=1 both yield 0.0 stddev.
  unmasked_1 = nsim.measure_patch_similarity(ref[:, :1], deg[:, :1])
  _assert_array_equal(unmasked_1.freq_band_stddevs, 0.0)

  for vc in (1, 2, 7, 17):
    ref_dirty = ref.at[:, vc:].set(-777.0)
    deg_dirty = deg.at[:, vc:].set(888.0)
    masked = nsim.measure_patch_similarity(ref_dirty, deg_dirty, valid_cols=vc)
    unmasked = nsim.measure_patch_similarity(ref[:, :vc], deg[:, :vc])
    assert float(masked.similarity) == pytest.approx(
        float(unmasked.similarity), abs=1e-6
    )
    _assert_allclose(
        masked.freq_band_means, unmasked.freq_band_means, atol=1e-6, rtol=1e-6
    )
    _assert_allclose(
        masked.freq_band_stddevs,
        unmasked.freq_band_stddevs,
        atol=1e-6,
        rtol=1e-6,
    )
    _assert_allclose(
        masked.freq_band_deg_energy,
        unmasked.freq_band_deg_energy,
        atol=1e-5,
        rtol=1e-6,
    )
    if vc == 1:
      _assert_array_equal(masked.freq_band_stddevs, 0.0)

  # valid_cols=0 yields finite zeros; valid_cols > W clamps to W.
  zero_vc = nsim.measure_patch_similarity(ref, deg, valid_cols=0)
  assert float(zero_vc.similarity) == 0.0
  _assert_array_equal(zero_vc.freq_band_means, 0.0)
  _assert_array_equal(zero_vc.freq_band_stddevs, 0.0)
  _assert_array_equal(zero_vc.freq_band_deg_energy, 0.0)

  over_vc = nsim.measure_patch_similarity(ref, deg, valid_cols=50)
  assert float(over_vc.similarity) == pytest.approx(
      float(full.similarity), abs=1e-6
  )

  # Constant patches across [0, 95] dB return 1.0 without f32 cancellation.
  for fill in (0.0, 12.34, 37.72, 51.64, 93.19):
    const = jnp.full((8, 6), fill, dtype=jnp.float32)
    res_const = nsim.measure_patch_similarity(const, const)
    assert float(res_const.similarity) == pytest.approx(1.0, abs=1e-6)
    _assert_allclose(res_const.freq_band_means, 1.0, atol=1e-6)
    _assert_allclose(res_const.freq_band_stddevs, 0.0, atol=1e-6)

  # High-contrast bands (60 dB at band 0, 0 dB at quiet bands) stay 1.0 on id.
  mixed_bands = jnp.concatenate(
      [
          jnp.full((2, 6), 60.0, dtype=jnp.float32),
          jnp.zeros((6, 6), dtype=jnp.float32),
      ],
      axis=0,
  )
  res_mixed = nsim.measure_patch_similarity(mixed_bands, mixed_bands)
  assert float(res_mixed.similarity) == pytest.approx(1.0, abs=1e-6)
  _assert_allclose(res_mixed.freq_band_means, 1.0, atol=1e-6)

  # Distinct constant dB levels and near-constant high-dB patches match f64.
  c_hi = jnp.full((8, 6), 51.64, dtype=jnp.float32)
  c_lo = jnp.full((8, 6), 37.72, dtype=jnp.float32)
  # pylint: disable=protected-access
  win_f32 = jnp.asarray(nsim.GAUSSIAN_WINDOW, dtype=jnp.float32)
  pad_hi = nsim._edge_pad_2d(c_hi)
  mu_sten = nsim._conv2d_stencil(pad_hi, win_f32)
  anc = pad_hi[..., :1, :1]
  pad_c = pad_hi - anc
  mu_c = mu_sten - anc
  sig_sten = jnp.maximum(
      nsim._conv2d_stencil(pad_c * pad_c, win_f32) - mu_c * mu_c,
      jnp.float32(0.0),
  )
  st_sten = nsim.PatchNsimStats(pad_hi, mu_sten, mu_sten * mu_sten, sig_sten)
  _assert_allclose(nsim.sim_map_from_stats(st_sten, st_sten), 1.0, atol=1e-5)
  # pylint: enable=protected-access
  want_dc, _, _, _ = _cpp_nsim_ref(c_hi, c_lo)
  assert float(nsim.measure_patch_similarity(c_hi, c_lo).similarity) == (
      pytest.approx(want_dc, abs=1e-5)
  )
  kr1, kr2 = jax.random.split(jax.random.key(99))
  near_r = jnp.float32(50.0) + jnp.float32(1e-2) * jax.random.normal(
      kr1, (8, 6), dtype=jnp.float32
  )
  near_d = jnp.float32(50.0) + jnp.float32(1e-2) * jax.random.normal(
      kr2, (8, 6), dtype=jnp.float32
  )
  want_near, want_nfbm, want_nfstd, _ = _cpp_nsim_ref(near_r, near_d)
  got_near = nsim.measure_patch_similarity(near_r, near_d)
  assert float(got_near.similarity) == pytest.approx(want_near, abs=5e-5)
  _assert_allclose(got_near.freq_band_means, want_nfbm, atol=1e-4, rtol=1e-4)
  _assert_allclose(got_near.freq_band_stddevs, want_nfstd, atol=1e-4, rtol=1e-4)

  # Negative variance roundoff (one or both < 0) clamps denominator to c3.
  st_pos = nsim.precompute_patch_stats(jnp.ones((4, 5), dtype=jnp.float32))
  st_neg = st_pos._replace(sigma_sq=jnp.full((4, 5), -1e-3, dtype=jnp.float32))
  st_unit = st_pos._replace(sigma_sq=jnp.ones((4, 5), dtype=jnp.float32))
  for smap in (
      nsim.sim_map_from_stats(st_neg, st_unit),
      nsim.sim_map_from_stats(st_neg, st_neg),
  ):
    assert bool(jnp.all(jnp.isfinite(smap)))
    _assert_allclose(smap, 1.0, atol=1e-5)


def test_nsim_jit_vmap_grad_x64_hlo_and_validation():
  k_ref, k_noise = jax.random.split(jax.random.key(42))
  ref_b = jax.random.uniform(
      k_ref, (4, 8, 10), dtype=jnp.float32, minval=0.0, maxval=30.0
  )
  deg_b = ref_b + 1.5 * jax.random.normal(
      k_noise, (4, 8, 10), dtype=jnp.float32
  )

  batched = jax.jit(nsim.measure_patch_similarity)(ref_b, deg_b)
  vmapped = jax.jit(
      jax.vmap(
          lambda r, d, vc: nsim.measure_patch_similarity(r, d, valid_cols=vc)
      )
  )(ref_b, deg_b, jnp.array([1, 4, 10, 10], dtype=jnp.int32))
  assert batched.similarity.shape == (4,)
  empty_batch = nsim.measure_patch_similarity(ref_b[:0], deg_b[:0])
  assert empty_batch.similarity.shape == (0,)
  assert empty_batch.freq_band_means.shape == (0, 8)
  _assert_allclose(vmapped.similarity[2:], batched.similarity[2:], atol=1e-6)
  _assert_array_equal(vmapped.freq_band_stddevs[0], 0.0)

  # Traced intensity_range inside jit matches eager custom intensity_range.
  want_ir, _, _, _ = _cpp_nsim_ref(ref_b[0], deg_b[0], intensity_range=2.5)
  eager_ir = nsim.measure_patch_similarity(
      ref_b[0], deg_b[0], intensity_range=2.5
  )
  got_ir = jax.jit(
      lambda r, d, ir: nsim.measure_patch_similarity(r, d, intensity_range=ir)
  )(ref_b[0], deg_b[0], jnp.float32(2.5))
  assert float(got_ir.similarity) == pytest.approx(
      float(eager_ir.similarity), abs=1e-7
  )
  assert float(got_ir.similarity) == pytest.approx(want_ir, abs=5e-4)

  # Gradients stay finite across noisy, silent, constant, and masked patches.
  def loss_fn(d: jax.Array, r: jax.Array, vc: int | None = None) -> jax.Array:
    res = nsim.measure_patch_similarity(r, d, valid_cols=vc)
    return res.similarity + jnp.sum(res.freq_band_stddevs)

  grad_noisy = jax.jit(jax.grad(loss_fn))(deg_b[0], ref_b[0])
  assert grad_noisy.dtype == jnp.float32 and bool(
      jnp.all(jnp.isfinite(grad_noisy))
  )
  for fill in (0.0, 12.34):
    c_patch = jnp.full((4, 5), fill, dtype=jnp.float32)
    g_const = jax.grad(loss_fn)(c_patch, c_patch)
    assert bool(jnp.all(jnp.isfinite(g_const)))
  for vc in (0, 1):
    g_vc = jax.grad(lambda d, c=vc: loss_fn(d, ref_b[0], vc=c))(deg_b[0])
    assert bool(jnp.all(jnp.isfinite(g_vc)))

  # Coerces float64 inputs to float32 and emits only f32 ops under x64=True.
  with jax.enable_x64(True):
    r64 = ref_b[0].astype(jnp.float64)
    d64 = deg_b[0].astype(jnp.float64)
    out_x64 = nsim.measure_patch_similarity(r64, d64, intensity_range=2.5)
    assert out_x64.similarity.dtype == jnp.float32
    assert out_x64.freq_band_means.dtype == jnp.float32
    assert out_x64.freq_band_stddevs.dtype == jnp.float32
    assert out_x64.freq_band_deg_energy.dtype == jnp.float32
    conv_x64 = nsim.valid_2d_conv_with_boundary(
        jnp.ones((3, 3), dtype=jnp.float64), r64
    )
    assert conv_x64.dtype == jnp.float32
    p2d = jax.ShapeDtypeStruct((8, 10), jnp.float32)
    i32 = jax.ShapeDtypeStruct((), jnp.int32)
    for hlo in (
        jax.jit(nsim.measure_patch_similarity).lower(p2d, p2d).as_text(),
        jax.jit(
            lambda r, d, vc: nsim.measure_patch_similarity(r, d, valid_cols=vc)
        )
        .lower(p2d, p2d, i32)
        .as_text(),
    ):
      assert {int(w) for w in _FLOAT_WIDTH.findall(hlo)} == {32}

  ones3 = jnp.ones((3, 3), dtype=jnp.float32)
  ones34 = jnp.ones((3, 4), dtype=jnp.float32)
  for bad_filt in (jnp.ones((3,)), jnp.zeros((0, 3)), jnp.zeros((3, 0))):
    with pytest.raises(ValueError, match="fir_filter"):
      nsim.valid_2d_conv_with_boundary(bad_filt, ones3)
  for bad_in in (jnp.ones((4,)), jnp.zeros((0, 3)), jnp.zeros((3, 0))):
    with pytest.raises(ValueError, match="at least 2D"):
      nsim.valid_2d_conv_with_boundary(ones3, bad_in)
  with pytest.raises(ValueError, match="smaller than filter"):
    nsim.valid_2d_conv_with_boundary(jnp.ones((5, 5)), jnp.ones((2, 2)))
  with pytest.raises(ValueError, match="ref_patch shape"):
    nsim.measure_patch_similarity(ones34, jnp.ones((3, 5)))
  with pytest.raises(ValueError, match="spatial shape mismatch"):
    nsim.sim_map_from_stats(
        nsim.precompute_patch_stats(ones34),
        nsim.precompute_patch_stats(jnp.ones((3, 5))),
    )
  good_st = nsim.precompute_patch_stats(ones34)
  for bad_st in (
      good_st._replace(padded=jnp.ones((6, 6))),
      good_st._replace(mu_sq=jnp.ones((3, 5))),
      good_st._replace(sigma_sq=jnp.ones((2, 4))),
  ):
    with pytest.raises(ValueError, match="spatial shape mismatch"):
      nsim.sim_map_from_stats(bad_st, good_st)
  with pytest.raises(ValueError, match="PatchNsimStats"):
    nsim.sim_map_from_stats((1, 2, 3, 4), good_st)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="at least 2D"):
    nsim.sim_map_from_stats(good_st._replace(padded=jnp.ones((5,))), good_st)
  with pytest.raises(ValueError, match="real-valued"):
    nsim.sim_map_from_stats(
        good_st._replace(sigma_sq=jnp.ones((3, 4), dtype=jnp.complex64)),
        good_st,
    )
  for bad_ir in (
      0.0,
      -1.0,
      True,
      _NP_BOOL_TRUE,
      float("nan"),
      float("inf"),
      1e300,
      1e-300,
      jnp.array(0.0),
      jnp.array(jnp.nan),
      jnp.array([1.0, 2.0]),
  ):
    with pytest.raises(ValueError, match="intensity_range"):
      nsim.measure_patch_similarity(ones34, ones34, intensity_range=bad_ir)
  for bad_vc in (True, _NP_BOOL_TRUE, 1.5, jnp.array([1, 2])):
    with pytest.raises(ValueError, match="valid_cols"):
      nsim.measure_patch_similarity(ones34, ones34, valid_cols=bad_vc)
  with pytest.raises(ValueError, match="real-valued"):
    nsim.measure_patch_similarity(
        jnp.ones((3, 4), dtype=jnp.complex64),
        jnp.ones((3, 4), dtype=jnp.complex64),
    )
