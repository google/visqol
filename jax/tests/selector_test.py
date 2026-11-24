"""Tests for selector.py against C++ ViSQOL v3.3.3."""

import math
import re

import jax
import jax.numpy as jnp
import pytest

from visqol_jax import alignment
from visqol_jax import nsim
from visqol_jax import patches
from visqol_jax import selector
from visqol_jax import spectrogram

_FFT_LINE = re.compile(r"^\s*%\S+ = stablehlo\.fft .*?:(?P<types>.*)$")
_FLOAT_WIDTH = re.compile(r"(?:<|x|\b)f(?P<width>16|32|64)\b")
_NP_BOOL_TRUE = jnp.dtype(jnp.bool_).type(True)

_JIT_SPEC = jax.jit(
    spectrogram.build_gammatone_spectrogram,
    static_argnums=(1,),
    static_argnames=("use_speech_mode", "num_bands"),
)


def _lcg_f32(
    seed: int, shape: tuple[int, ...], lo: float = -1.0, hi: float = 1.0
) -> jax.Array:
  n = math.prod(shape)
  state = seed & 0xFFFFFFFF
  vals = [0.0] * n
  scale = (hi - lo) / float(0x100000000)
  for i in range(n):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    vals[i] = lo + state * scale
  return jnp.asarray(vals, dtype=jnp.float32).reshape(shape)


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


def _py_conv3x3(x: list[list[float]]) -> list[list[float]]:
  h, cols = len(x), len(x[0])
  padded = [[x[0][0], *x[0], x[0][-1]]]
  for r in x:
    padded.append([r[0], *r, r[-1]])
  padded.append([x[-1][0], *x[-1], x[-1][-1]])
  w = nsim.GAUSSIAN_WINDOW
  out = [[0.0] * cols for _ in range(h)]
  for i in range(h):
    for j in range(cols):
      acc = 0.0
      for di in range(3):
        for dj in range(3):
          acc += w[2 - di][2 - dj] * padded[i + di][j + dj]
      out[i][j] = acc
  return out


def _py_patch_nsim(
    ref_p: list[list[float]], deg_p: list[list[float]]
) -> float:
  h, cols = len(ref_p), len(ref_p[0])
  c1, c3 = 0.01**2, (0.03**2) / 2.0
  mu_r = _py_conv3x3(ref_p)
  mu_d = _py_conv3x3(deg_p)
  sig_r_sq = _py_conv3x3(
      [[ref_p[i][j] ** 2 for j in range(cols)] for i in range(h)]
  )
  sig_d_sq = _py_conv3x3(
      [[deg_p[i][j] ** 2 for j in range(cols)] for i in range(h)]
  )
  sig_r_d = _py_conv3x3(
      [[ref_p[i][j] * deg_p[i][j] for j in range(cols)] for i in range(h)]
  )
  row_means = [0.0] * h
  for i in range(h):
    s_row = 0.0
    for j in range(cols):
      mr, md = mu_r[i][j], mu_d[i][j]
      sr2 = sig_r_sq[i][j] - mr * mr
      sd2 = sig_d_sq[i][j] - md * md
      srd = sig_r_d[i][j] - mr * md
      inten = (2.0 * mr * md + c1) / (mr * mr + md * md + c1)
      prod = sr2 * sd2
      denom = c3 if prod < 0.0 else math.sqrt(prod) + c3
      s_row += inten * ((srd + c3) / denom)
    row_means[i] = s_row / cols
  return sum(row_means) / h


def _cpp_dp_selector_ref(
    ref_patches_arr: jax.Array,
    ref_patch_indices: list[int],
    deg_spec_arr: jax.Array,
    frame_duration: float,
    search_window_radius: int,
) -> tuple[list[int], list[bool], list[float]]:
  """64-bit Python port of C++ FindMostOptimalDegPatches."""
  ref_patches_py: list[list[list[float]]] = ref_patches_arr.tolist()
  deg_spec_py: list[list[float]] = deg_spec_arr.tolist()
  patch_size = len(ref_patches_py[0][0])
  t_deg = len(deg_spec_py[0])
  sw = search_window_radius * patch_size
  num_patches = len(ref_patch_indices)
  while (
      num_patches > 0
      and ref_patch_indices[num_patches - 1] - (patch_size // 2) > t_deg
  ):
    num_patches -= 1
  if num_patches == 0:
    return [], [], []

  padded_deg = [row + [0.0] * (patch_size - 1) for row in deg_spec_py]
  deg_patches = [
      [row[s : s + patch_size] for row in padded_deg] for s in range(t_deg)
  ]
  dp = [[0.0] * t_deg for _ in range(len(ref_patch_indices))]
  bt = [[0] * t_deg for _ in range(len(ref_patch_indices))]
  lowest = -1.7976931348623157e308

  for p in range(num_patches):
    r_idx = ref_patch_indices[p]
    for s in range(max(0, r_idx - sw), min(t_deg, r_idx + sw + 1)):
      sim = _py_patch_nsim(ref_patches_py[p], deg_patches[s])
      past_offset = -1
      if p > 0:
        highest_sim = lowest
        lower_limit = max(ref_patch_indices[p - 1] - sw, 0)
        for back in range(s - 1, lower_limit - 1, -1):
          if dp[p - 1][back] > highest_sim:
            highest_sim = dp[p - 1][back]
            past_offset = back
        sim += highest_sim
        if dp[p - 1][s] > sim:
          sim = dp[p - 1][s]
          past_offset = s
      dp[p][s] = sim
      bt[p][s] = past_offset

  last_idx = num_patches - 1
  max_score = lowest
  last_offset = 0
  for s in range(
      max(0, ref_patch_indices[last_idx] - sw),
      min(t_deg, ref_patch_indices[last_idx] + sw + 1),
  ):
    if dp[last_idx][s] > max_score:
      max_score = dp[last_idx][s]
      last_offset = s

  offsets = [0] * num_patches
  pkt_loss = [False] * num_patches
  deg_starts = [0.0] * num_patches
  for p in range(num_patches - 1, -1, -1):
    offsets[p] = last_offset
    pkt_loss[p] = last_offset == bt[p][last_offset]
    deg_starts[p] = 0.0 if pkt_loss[p] else last_offset * frame_duration
    last_offset = bt[p][last_offset]
  return offsets, pkt_loss, deg_starts


def test_calc_max_num_patches_and_build_degraded_patch():
  # Exact EndPatches check from tests/comparison_patches_selector_test.cc.
  cpp_indices = [0, 15, 30, 45, 60]
  assert int(selector.calc_max_num_patches(cpp_indices, 45, 30)) == 5
  assert int(selector.calc_max_num_patches(cpp_indices, 44, 30)) == 4
  u32_max = jnp.uint32(0xFFFFFFFF)
  assert int(selector.calc_max_num_patches(cpp_indices, u32_max, 30)) == 5
  assert int(selector.calc_max_num_patches(cpp_indices, 10**30, 30)) == 5

  # Signed subtraction quirk: 9 - 10 = -1 <= 0 retains the initial VAD patch.
  assert (
      int(selector.calc_max_num_patches(jnp.array([9], jnp.int32), 0, 20)) == 1
  )
  assert (
      int(selector.calc_max_num_patches(jnp.array([30, 60], jnp.int32), 5, 20))
      == 0
  )
  assert (
      int(
          selector.calc_max_num_patches(
              jnp.array([9, 29, 99, 0], jnp.int32), 30, 20, num_ref_patches=2
          )
      )
      == 2
  )
  assert int(selector.calc_max_num_patches([], 30, 20)) == 0
  assert (
      int(selector.calc_max_num_patches(jnp.zeros((0,), jnp.int32), 30, 20))
      == 0
  )

  for bad_ps in (0, -1, True, _NP_BOOL_TRUE, 2.5):
    with pytest.raises(ValueError, match="patch_size"):
      selector.calc_max_num_patches(jnp.array([0], jnp.int32), 10, bad_ps)
  for bad_idx in (
      jnp.array([0.5]),
      jnp.array([True]),
      [True, 1],
      [jnp.array(True), 1],
      jnp.zeros((0,), dtype=jnp.bool_),
      jnp.zeros((0,), dtype=jnp.float32),
  ):
    with pytest.raises(ValueError, match="integer dtype"):
      selector.calc_max_num_patches(bad_idx, 10, 20)
  for bad_neg in (jnp.array([-1], jnp.int32), [-1], [1 << 40]):
    with pytest.raises(ValueError, match="non-negative"):
      selector.calc_max_num_patches(bad_neg, 10, 20)
  for bad_nrp in (True, _NP_BOOL_TRUE, 1.5, jnp.array([1])):
    with pytest.raises(ValueError, match="num_ref_patches"):
      selector.calc_max_num_patches([], 10, 20, num_ref_patches=bad_nrp)
  for bad_df in (-1, True, _NP_BOOL_TRUE, 4.5):
    with pytest.raises(ValueError, match="deg_num_frames"):
      selector.calc_max_num_patches([0], bad_df, 20)

  spec = jnp.arange(1, 13, dtype=jnp.float32).reshape(3, 4)
  p_neg = selector.build_degraded_patch(spec, -2, 5)
  assert p_neg.dtype == jnp.float32
  _assert_array_equal(p_neg[:, :2], 0.0)
  _assert_array_equal(p_neg[:, 2:], spec[:, :3])

  spec_dirty = spec.at[:, 3].set(999.0)
  p_tail = selector.build_degraded_patch(spec_dirty, 1, 5, deg_valid_cols=3)
  _assert_array_equal(p_tail[:, :2], spec[:, 1:3])
  _assert_array_equal(p_tail[:, 2:], 0.0)
  for far_start in (10, -5, -10**30, jnp.uint32(0xFFFFFFFF)):
    _assert_array_equal(selector.build_degraded_patch(spec, far_start, 3), 0.0)


def test_selector_matches_all_cpp_unit_tests():
  # OptimalPatches from tests/comparison_patches_selector_test.cc.
  opt_ref = jnp.array(
      [
          [1, 1, 1, 2, 2, 2, 2, 2, 3, 3],
          [0, 1, 0, 2, 1, 1, 2, 3, 1, 2],
          [0, 1, 0, 2, 1, 1, 2, 3, 1, 2],
      ],
      dtype=jnp.float32,
  )
  opt_deg = jnp.array(
      [
          [
              0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 2,
              3, 2, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0,
          ],
          [
              0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 2, 1,
              2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0,
          ],
          [
              0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1,
              2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0,
          ],
      ],
      dtype=jnp.float32,
  )
  opt_idx = jnp.arange(10, dtype=jnp.int32)
  opt_res = selector.find_most_optimal_deg_patches(
      patches.create_patches_from_indices(opt_ref, opt_idx, 1),
      opt_idx,
      opt_deg,
      1.0,
      8,
  )
  assert int(opt_res.num_patches) == 10
  assert float(opt_res.deg_patch_start_time[3]) == 0.0
  assert float(opt_res.deg_patch_start_time[4]) == 7.0
  assert float(opt_res.deg_patch_start_time[5]) == 8.0

  # OutOfOrderMatches: patch 1 experiences packet loss (deg start/end == 0.0).
  ooo_ref = jnp.array(
      [[1, 100, 3, 4], [0, 0, 0, 0], [1, 100, 3, 4]], dtype=jnp.float32
  )
  ooo_deg = jnp.array(
      [[100, 1, 3, 4], [0, 0, 0, 0], [100, 1, 3, 4]], dtype=jnp.float32
  )
  ooo_idx = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
  ooo_res = selector.find_most_optimal_deg_patches(
      patches.create_patches_from_indices(ooo_ref, ooo_idx, 1),
      ooo_idx,
      ooo_deg,
      1.0,
      60,
  )
  _assert_array_equal(ooo_res.deg_patch_start_time, [1.0, 0.0, 2.0, 3.0])
  _assert_array_equal(ooo_res.deg_patch_end_time, [2.0, 0.0, 3.0, 4.0])
  _assert_array_equal(ooo_res.ref_patch_start_time, [0.0, 1.0, 2.0, 3.0])
  _assert_array_equal(ooo_res.ref_patch_end_time, [1.0, 2.0, 3.0, 4.0])
  assert float(ooo_res.similarity[1]) == 0.0
  _assert_array_equal(ooo_res.freq_band_means[1], 0.0)
  _assert_array_equal(ooo_res.freq_band_deg_energy[1], [1.0, 0.0, 1.0])

  # DifferentResults from tests/comparison_patches_selector_test.cc.
  diff_ref = jnp.array([[1], [1], [0]], dtype=jnp.float32)
  diff_deg = jnp.array(
      [
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      ],
      dtype=jnp.float32,
  )
  diff_idx = jnp.array([0], dtype=jnp.int32)
  diff_res = selector.find_most_optimal_deg_patches(
      patches.create_patches_from_indices(diff_ref, diff_idx, 1),
      diff_idx,
      diff_deg,
      1.0,
      60,
  )
  _assert_array_equal(diff_res.deg_patch_start_time, [6.0])

  # BigExample from tests/comparison_patches_selector_test.cc.
  big_ref = jnp.array(
      [
          [
              0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2,
              0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0,
          ],
          [
              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 2, 3,
              0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0,
          ],
          [
              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 2, 3,
              0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0,
          ],
      ],
      dtype=jnp.float32,
  )
  big_deg = jnp.array(
      [
          [
              0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 1, 2,
              3, 2, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0,
          ],
          [
              0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1,
              2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0,
          ],
          [
              0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1,
              2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0,
          ],
      ],
      dtype=jnp.float32,
  )
  big_idx = jnp.array([4, 6, 10, 12, 14, 22], dtype=jnp.int32)
  big_res = selector.find_most_optimal_deg_patches(
      patches.create_patches_from_indices(big_ref, big_idx, 2),
      big_idx,
      big_deg,
      1.0,
      60,
  )
  _assert_array_equal(
      big_res.deg_patch_start_time, [6.0, 8.0, 12.0, 14.0, 16.0, 22.0]
  )


def test_dp_monotonic_backtrace_ties_windowed_scan_and_masking():
  # Compare associative-scan DP against sequential C++ reference across regimes.
  for idx_case, (sw_rad, t_deg, r_list) in enumerate((
      (2, 25, [1, 5, 8, 15, 20]),
      (1, 18, [0, 4, 9, 14]),
      (20, 14, [0, 3, 6, 10]),
  )):
    ref_spec = _lcg_f32(100 + 2 * idx_case, (5, 24), 0.0, 20.0)
    deg_spec = _lcg_f32(101 + 2 * idx_case, (5, t_deg), 0.0, 20.0)
    r_idx = jnp.array(r_list, dtype=jnp.int32)
    r_p = patches.create_patches_from_indices(ref_spec, r_idx, 2)
    got = selector.find_most_optimal_deg_patches(
        r_p, r_idx, deg_spec, 0.02, search_window_radius=sw_rad
    )
    n_p = int(got.num_patches)
    want_offsets, want_loss, want_deg_s = _cpp_dp_selector_ref(
        r_p, r_list, deg_spec, 0.02, sw_rad
    )
    assert n_p == len(want_offsets)
    _assert_allclose(got.deg_patch_start_time[:n_p], want_deg_s, atol=1e-6)
    # Matched offsets are non-decreasing and strictly increasing when matched.
    for i in range(n_p - 1):
      if want_loss[i + 1]:
        assert want_offsets[i] == want_offsets[i + 1]
      else:
        assert want_offsets[i] < want_offsets[i + 1]

  # Ties favor smallest last offset (1) and largest back offset (0) -> [0, 1].
  tie_spec = jnp.ones((3, 6), dtype=jnp.float32)
  tie_idx = jnp.array([0, 2], dtype=jnp.int32)
  tie_p = patches.create_patches_from_indices(tie_spec, tie_idx, 2)
  tie_got = selector.find_most_optimal_deg_patches(
      tie_p, tie_idx, tie_spec, 1.0, 2
  )
  _assert_array_equal(tie_got.deg_patch_start_time, [0.0, 1.0])

  # Padded ref_patches + deg_spectrogram match trimmed unpadded call.
  ref_s = _lcg_f32(201, (5, 24), 0.0, 20.0)
  deg_s = (
      _lcg_f32(202, (5, 25), 0.0, 20.0)
      .at[:, 2:4]
      .set(ref_s[:, 1:3])
      .at[:, 9:11]
      .set(ref_s[:, 8:10])
  )
  r_idx = jnp.array([1, 5, 8], dtype=jnp.int32)
  r_p = patches.create_patches_from_indices(ref_s, r_idx, 2)
  unpadded = selector.find_most_optimal_deg_patches(
      r_p, r_idx, deg_s, 0.02, search_window_radius=2
  )
  assert float(unpadded.deg_patch_start_time[0]) == pytest.approx(0.04)
  assert float(unpadded.deg_patch_start_time[2]) == pytest.approx(0.18)

  r_p_pad = jnp.concatenate(
      [r_p, jnp.full((2, 5, 2), -50.0, dtype=jnp.float32)], axis=0
  )
  r_idx_pad = jnp.array([1, 5, 8, 0, 0], dtype=jnp.int32)
  deg_pad = jnp.concatenate(
      [deg_s, jnp.full((5, 7), 99.0, dtype=jnp.float32)], axis=1
  )
  masked = selector.find_most_optimal_deg_patches(
      r_p_pad,
      r_idx_pad,
      deg_pad,
      0.02,
      search_window_radius=2,
      num_ref_patches=3,
      deg_valid_cols=25,
  )
  assert int(masked.num_patches) == 3
  for field in selector.PatchSimilarityResult._fields[:-1]:
    _assert_allclose(
        getattr(masked, field)[:3], getattr(unpadded, field), atol=1e-6
    )
    _assert_array_equal(getattr(masked, field)[3:], 0.0)

  # num_patches == 0 stays finite and zeroed under jax.jit (including n_deg==0).
  empty = jax.jit(selector.find_most_optimal_deg_patches, static_argnums=(4,))(
      r_p_pad, r_idx_pad, deg_pad, 0.02, 2, num_ref_patches=0, deg_valid_cols=25
  )
  assert int(empty.num_patches) == 0
  for field in selector.PatchSimilarityResult._fields[:-1]:
    _assert_array_equal(getattr(empty, field), 0.0)

  vad_idx = jnp.array([9, 29], dtype=jnp.int32)
  vad_p = jnp.ones((2, 5, 20), dtype=jnp.float32)
  empty_deg = jax.jit(
      selector.find_most_optimal_deg_patches, static_argnums=(4,)
  )(
      vad_p,
      vad_idx,
      jnp.ones((5, 30), dtype=jnp.float32),
      0.02,
      2,
      deg_valid_cols=0,
  )
  assert int(empty_deg.num_patches) == 0
  for field in selector.PatchSimilarityResult._fields[:-1]:
    _assert_array_equal(getattr(empty_deg, field), 0.0)

  # Exact prev_dp == cand tie (0.5 + 0.5 == 1.0) does not skip (> vs >=).
  tie_skip_ref = jnp.stack(
      [
          jnp.full((3, 1), 0.01, dtype=jnp.float32),
          jnp.zeros((3, 1), dtype=jnp.float32),
      ],
      axis=0,
  )
  tie_skip_deg = jnp.broadcast_to(
      jnp.array([[0.0, 0.01]], dtype=jnp.float32), (3, 2)
  )
  no_skip = selector.find_most_optimal_deg_patches(
      tie_skip_ref, jnp.array([0, 1], dtype=jnp.int32), tie_skip_deg, 1.0, 2
  )
  _assert_array_equal(no_skip.deg_patch_start_time, [0.0, 1.0])
  _assert_array_equal(no_skip.deg_patch_end_time, [1.0, 2.0])


def test_slice_audio_cpp_unit_test_and_quirks():
  # Exact Slice check from tests/comparison_patches_selector_test.cc.
  sig = jnp.zeros((16000 * 3,), dtype=jnp.float32).at[16000].set(1.0)
  sliced, vlen = selector.slice_audio(sig, 0.5, 2.5, 16000)
  assert sliced.dtype == jnp.float32
  assert int(vlen) == 32000
  assert float(sliced[7999]) == 0.0
  assert float(sliced[8000]) == 1.0
  assert float(sliced[8001]) == 0.0

  # Quirk 1: end_time * fs == N clamps end_index to N-1, dropping sample N-1.
  ramp = jnp.arange(1.0, 101.0, dtype=jnp.float32)
  s_end, l_end = selector.slice_audio(ramp, 0.8, 1.0, 100)
  assert int(l_end) == 19
  _assert_array_equal(s_end, ramp[80:99])

  # Quirk 2: end_time * fs > N drops sample N-1 and PREPENDS postsilence zeros.
  s1, l1 = selector.slice_audio(ramp, 0.8, 1.25, 100)
  assert int(l1) == 25 + 19
  _assert_array_equal(s1[:25], 0.0)
  _assert_array_equal(s1[25:], ramp[80:99])

  # Fractional 0 < end_diff < 1 truncates postsilence to 0 zeros.
  s_frac, l_frac = selector.slice_audio(ramp, 0.8, 1.005, 100)
  assert int(l_frac) == 19
  _assert_array_equal(s_frac, ramp[80:99])

  # Negative start_time + end_time > duration prepends both silences.
  padded = jnp.concatenate([ramp, jnp.full((30,), -999.0, dtype=jnp.float32)])
  j_slice = jax.jit(selector.slice_audio, static_argnames=("max_len",))
  s3, l3 = j_slice(padded, -0.25, 1.25, 100, signal_len=100, max_len=160)
  assert s3.shape == (160,) and int(l3) == 50 + 99
  _assert_array_equal(s3[:50], 0.0)
  _assert_array_equal(s3[50:149], ramp[:99])
  _assert_array_equal(s3[149:], 0.0)

  # max_len smaller than raw_valid_len truncates valid_len to max_len.
  s_cap, l_cap = j_slice(ramp, 0.0, 0.5, 100, max_len=12)
  assert s_cap.shape == (12,) and int(l_cap) == 12
  _assert_array_equal(s_cap, ramp[:12])

  # Huge time bounds (forward and reversed) never wrap signed int32.
  s_huge, l_huge = j_slice(ramp, -1e6, 1e6, 48000, max_len=16)
  assert s_huge.shape == (16,) and int(l_huge) == 16
  _assert_array_equal(s_huge, 0.0)
  s_rev, l_rev = j_slice(ramp, 1e6, -1e6, 48000, max_len=16)
  assert s_rev.shape == (16,) and int(l_rev) == 0
  _assert_array_equal(s_rev, 0.0)
  s_rev_eager, l_rev_eager = selector.slice_audio(
      ramp, 30000.0, -20000.0, 48000
  )
  assert s_rev_eager.shape == (0,) and int(l_rev_eager) == 0

  # Empty core range, signal_len=0, and max_len=0 stay bounded and exact.
  s_empty, l_empty = selector.slice_audio(ramp, 0.6, 0.4, 100)
  assert s_empty.shape == (0,) and int(l_empty) == 0
  s_zsig, l_zsig = selector.slice_audio(ramp, 0.0, 0.05, 100, signal_len=0)
  assert s_zsig.shape == (5,) and int(l_zsig) == 5
  _assert_array_equal(s_zsig, 0.0)
  s_zmax, l_zmax = selector.slice_audio(ramp, 0.0, 0.5, 100, max_len=0)
  assert s_zmax.shape == (0,) and int(l_zmax) == 0

  # 48 kHz float32 slice lengths stay consistent across eager and jit within +2.
  def slice_48k_lens(sig48: jax.Array) -> tuple[jax.Array, jax.Array]:
    fd48 = jnp.float32(960.0 / 48000.0)
    pd48 = fd48 * jnp.float32(30.0)
    st29 = jnp.int32(29).astype(jnp.float32) * fd48
    _, v29 = selector.slice_audio(
        sig48, st29, st29 + pd48, 48000, max_len=28802
    )
    st194 = jnp.int32(194).astype(jnp.float32) * fd48
    _, v194 = selector.slice_audio(
        sig48, st194, st194 + pd48, 48000, max_len=28802
    )
    return v29, v194

  sig48 = jnp.zeros((240000,), dtype=jnp.float32)
  e29, e194 = slice_48k_lens(sig48)
  j29, j194 = jax.jit(slice_48k_lens)(sig48)
  assert int(e29) == int(j29) and 28798 <= int(j29) <= 28802
  assert int(e194) == int(j194) and 28798 <= int(j194) <= 28802

  with pytest.raises(ValueError, match="max_len"):
    jax.jit(selector.slice_audio)(ramp, 0.1, 0.5, 100)


def test_finely_align_tie_adoption_lags_packet_loss_and_masking():
  sr = 2000
  nb = 6
  ps = 6
  ws, hop, _ = spectrogram.analysis_window(sr)
  fd = hop / float(sr)
  n = ws + 24 * hop
  pad_len = 40

  @jax.jit
  def realign_fn(
      sim_res: selector.PatchSimilarityResult,
      r_sig: jax.Array,
      d_sig: jax.Array,
      r_len: jax.Array,
      d_len: jax.Array,
  ) -> selector.PatchSimilarityResult:
    return selector.finely_align_and_recreate_patches(
        sim_res,
        r_sig,
        d_sig,
        sr,
        use_speech_mode=True,
        patch_size=ps,
        num_bands=nb,
        ref_len=r_len,
        deg_len=d_len,
    )

  # Ties (new_sim == old_sim) adopt new_res; new_sim < old_sim retains old.
  const_sig = jnp.zeros((n + pad_len,), dtype=jnp.float32)
  old_tie = selector.PatchSimilarityResult(
      similarity=jnp.array([1.0, 2.0, 0.0], dtype=jnp.float32),
      freq_band_means=jnp.ones((3, nb), dtype=jnp.float32),
      freq_band_stddevs=jnp.full((3, nb), 77.0, dtype=jnp.float32),
      freq_band_deg_energy=jnp.full((3, nb), 88.0, dtype=jnp.float32),
      ref_patch_start_time=jnp.array([0.0, 0.1, 0.2], dtype=jnp.float32),
      ref_patch_end_time=jnp.array([0.12, 0.22, 0.32], dtype=jnp.float32),
      deg_patch_start_time=jnp.array([0.0, 0.1, 0.0], dtype=jnp.float32),
      deg_patch_end_time=jnp.array([0.12, 0.22, 0.0], dtype=jnp.float32),
      num_patches=jnp.int32(3),
  )
  realigned_tie = realign_fn(
      old_tie, const_sig, const_sig, jnp.int32(n), jnp.int32(n)
  )
  assert float(realigned_tie.similarity[0]) == pytest.approx(1.0, abs=1e-6)
  _assert_array_equal(realigned_tie.freq_band_stddevs[0], 0.0)
  _assert_array_equal(realigned_tie.freq_band_deg_energy[0], 0.0)
  assert float(realigned_tie.similarity[1]) == 2.0
  _assert_array_equal(realigned_tie.freq_band_stddevs[1], 77.0)
  _assert_array_equal(realigned_tie.freq_band_stddevs[2], 77.0)

  # Positive and negative local shifts raise similarity and adjust start times.
  t = jnp.arange(n, dtype=jnp.float32) / jnp.float32(sr)
  env = jnp.exp(-0.5 * ((t - 0.28) / 0.08) ** 2)
  two_pi = jnp.float32(2.0 * math.pi)
  ref = env * (
      0.5 * jnp.sin(two_pi * (140.0 * t + 110.0 * t * t))
      + 0.25 * jnp.sin(two_pi * 410.0 * t)
  )
  shift = 8
  deg_pos = jnp.concatenate([ref[shift:], jnp.zeros((shift,), jnp.float32)])
  deg_neg = jnp.concatenate([jnp.zeros((shift,), jnp.float32), ref[:-shift]])

  ref_spec, _ = _JIT_SPEC(ref, sr, use_speech_mode=True, num_bands=nb)
  r_idx = jnp.array([8, 0, 0], dtype=jnp.int32)
  ref_pad0 = jnp.concatenate([ref, jnp.zeros((pad_len,), dtype=jnp.float32)])

  for deg, is_pos in ((deg_pos, True), (deg_neg, False)):
    deg_spec, _ = _JIT_SPEC(deg, sr, use_speech_mode=True, num_bands=nb)
    prep_r, prep_d = spectrogram.prepare_spectrograms(ref_spec, deg_spec)
    r_p = patches.create_patches_from_indices(prep_r, r_idx, ps)
    coarse = selector.find_most_optimal_deg_patches(
        r_p, r_idx, prep_d, fd, search_window_radius=3, num_ref_patches=1
    )
    deg_pad0 = jnp.concatenate([deg, jnp.zeros((pad_len,), dtype=jnp.float32)])
    fine = realign_fn(coarse, ref_pad0, deg_pad0, jnp.int32(n), jnp.int32(n))
    assert float(fine.similarity[0]) > float(coarse.similarity[0])
    if is_pos:
      assert float(fine.ref_patch_start_time[0]) == pytest.approx(
          float(coarse.ref_patch_start_time[0]) + shift / sr, abs=2.0 / sr
      )
      assert float(fine.deg_patch_start_time[0]) == float(
          coarse.deg_patch_start_time[0]
      )
    else:
      assert float(fine.ref_patch_start_time[0]) == float(
          coarse.ref_patch_start_time[0]
      )
      assert float(fine.deg_patch_start_time[0]) == pytest.approx(
          float(coarse.deg_patch_start_time[0]) + shift / sr, abs=2.0 / sr
      )

    # Masked ref_len and deg_len ignore trailing junk samples.
    ref_pad = jnp.concatenate(
        [ref, jnp.full((pad_len,), 999.0, dtype=jnp.float32)]
    )
    deg_pad = jnp.concatenate(
        [deg, jnp.full((pad_len,), -999.0, dtype=jnp.float32)]
    )
    fine_masked = realign_fn(
        coarse, ref_pad, deg_pad, jnp.int32(n), jnp.int32(n)
    )
    for field in selector.PatchSimilarityResult._fields[:-1]:
      _assert_allclose(
          getattr(fine_masked, field), getattr(fine, field), atol=1e-6
      )

    # Unpadded step-by-step C++ FinelyAlignAndRecreatePatches parity on patch 0.
    r_sl, _ = selector.slice_audio(
        ref,
        float(coarse.ref_patch_start_time[0]),
        float(coarse.ref_patch_end_time[0]),
        sr,
    )
    d_sl, _ = selector.slice_audio(
        deg,
        float(coarse.deg_patch_start_time[0]),
        float(coarse.deg_patch_end_time[0]),
        sr,
    )
    r_al, d_al, al_n, lag_s = alignment.align_and_truncate(r_sl, d_sl, sr)
    r_sp, _ = _JIT_SPEC(
        r_al[: int(al_n)], sr, use_speech_mode=True, num_bands=nb
    )
    d_sp, _ = _JIT_SPEC(
        d_al[: int(al_n)], sr, use_speech_mode=True, num_bands=nb
    )
    pr, pd = spectrogram.prepare_spectrograms(r_sp, d_sp)
    want_nsim = nsim.measure_patch_similarity(pr, pd)
    assert float(fine.similarity[0]) == pytest.approx(
        float(want_nsim.similarity), abs=1e-5
    )
    _assert_allclose(fine.freq_band_means[0], want_nsim.freq_band_means)
    _assert_allclose(fine.freq_band_stddevs[0], want_nsim.freq_band_stddevs)
    _assert_allclose(
        fine.freq_band_deg_energy[0], want_nsim.freq_band_deg_energy
    )
    assert float(lag_s) != 0.0

  # Boundary-straddling short deg_len matches unpadded truncated deg.
  deg_short_len = 8 * hop + ws + hop
  deg_short = deg_pos[:deg_short_len]
  deg_short_spec, _ = _JIT_SPEC(
      deg_short, sr, use_speech_mode=True, num_bands=nb
  )
  pr_s, pd_s = spectrogram.prepare_spectrograms(ref_spec, deg_short_spec)
  coarse_s = selector.find_most_optimal_deg_patches(
      patches.create_patches_from_indices(pr_s, r_idx, ps),
      r_idx,
      pd_s,
      fd,
      search_window_radius=3,
      num_ref_patches=1,
  )
  r_sl_s, _ = selector.slice_audio(
      ref,
      float(coarse_s.ref_patch_start_time[0]),
      float(coarse_s.ref_patch_end_time[0]),
      sr,
  )
  d_sl_s, _ = selector.slice_audio(
      deg_short,
      float(coarse_s.deg_patch_start_time[0]),
      float(coarse_s.deg_patch_end_time[0]),
      sr,
  )
  r_al_s, d_al_s, al_n_s, _ = alignment.align_and_truncate(r_sl_s, d_sl_s, sr)
  r_sp_s, _ = _JIT_SPEC(
      r_al_s[: int(al_n_s)], sr, use_speech_mode=True, num_bands=nb
  )
  d_sp_s, _ = _JIT_SPEC(
      d_al_s[: int(al_n_s)], sr, use_speech_mode=True, num_bands=nb
  )
  want_short = nsim.measure_patch_similarity(
      *spectrogram.prepare_spectrograms(r_sp_s, d_sp_s)
  )
  deg_short_clean = jnp.concatenate([
      deg_short,
      jnp.zeros((n + pad_len - deg_short_len,), dtype=jnp.float32),
  ])
  deg_short_dirty = jnp.concatenate([
      deg_short,
      jnp.full((n + pad_len - deg_short_len,), 555.0, dtype=jnp.float32),
  ])
  n_short = jnp.int32(deg_short_len)
  fine_short_clean = realign_fn(
      coarse_s, ref_pad0, deg_short_clean, jnp.int32(n), n_short
  )
  fine_short_masked = realign_fn(
      coarse_s, ref_pad0, deg_short_dirty, jnp.int32(n), n_short
  )
  for field in selector.PatchSimilarityResult._fields[:-1]:
    _assert_allclose(
        getattr(fine_short_masked, field),
        getattr(fine_short_clean, field),
        atol=1e-6,
    )
  assert float(fine_short_masked.similarity[0]) == pytest.approx(
      max(float(want_short.similarity), float(coarse_s.similarity[0])), abs=1e-5
  )

  # Sub-window deg slice keeps coarse; num_patches==0 zeros inactive slots.
  sub_win_res = old_tie._replace(
      similarity=jnp.array([0.42, 9.0, 9.0], dtype=jnp.float32),
      ref_patch_start_time=jnp.array([0.20, 0.1, 0.2], dtype=jnp.float32),
      ref_patch_end_time=jnp.array([0.25, 0.2, 0.3], dtype=jnp.float32),
      deg_patch_start_time=jnp.array([0.20, 0.1, 0.2], dtype=jnp.float32),
      deg_patch_end_time=jnp.array([0.25, 0.2, 0.3], dtype=jnp.float32),
      num_patches=jnp.int32(1),
  )
  kept = realign_fn(
      sub_win_res, const_sig, const_sig, jnp.int32(n), jnp.int32(10)
  )
  assert float(kept.similarity[0]) == pytest.approx(0.42, abs=1e-6)
  for field in selector.PatchSimilarityResult._fields[:-1]:
    _assert_array_equal(getattr(kept, field)[1:], 0.0)

  zero_active = realign_fn(
      sub_win_res._replace(num_patches=jnp.int32(0)),
      const_sig,
      const_sig,
      jnp.int32(n),
      jnp.int32(n),
  )
  assert int(zero_active.num_patches) == 0
  for field in selector.PatchSimilarityResult._fields[:-1]:
    _assert_array_equal(getattr(zero_active, field), 0.0)


def test_selector_static_shape_jit_x64_hlo_grad_and_validation():
  traces = 0
  sr = 2000
  nb = 6
  ps = 6
  ws, hop, _ = spectrogram.analysis_window(sr)
  n = ws + 14 * hop
  fd = hop / float(sr)

  @jax.jit
  def run_step(
      ref: jax.Array,
      deg: jax.Array,
      ref_p: jax.Array,
      ref_idx: jax.Array,
      deg_spec: jax.Array,
      n_ref_p: jax.Array,
      deg_vcols: jax.Array,
      deg_len: jax.Array,
  ):
    nonlocal traces
    traces += 1
    coarse = selector.find_most_optimal_deg_patches(
        ref_p,
        ref_idx,
        deg_spec,
        fd,
        search_window_radius=4,
        num_ref_patches=n_ref_p,
        deg_valid_cols=deg_vcols,
    )
    return selector.finely_align_and_recreate_patches(
        coarse,
        ref,
        deg,
        sr,
        use_speech_mode=True,
        patch_size=ps,
        num_bands=nb,
        deg_len=deg_len,
    )

  ref = _lcg_f32(11, (n,), -0.1, 0.1)
  deg = _lcg_f32(12, (n,), -0.1, 0.1)
  ref_spec, _ = _JIT_SPEC(ref, sr, use_speech_mode=True, num_bands=nb)
  deg_spec, _ = _JIT_SPEC(deg, sr, use_speech_mode=True, num_bands=nb)
  prep_ref, prep_deg = spectrogram.prepare_spectrograms(ref_spec, deg_spec)
  r_idx = jnp.array([2, 6], dtype=jnp.int32)
  r_p = patches.create_patches_from_indices(prep_ref, r_idx, ps)

  for n_rp, dv, dl in [
      (1, 11, ws + 10 * hop),
      (2, 15, n),
      (0, 9, ws + 8 * hop),
  ]:
    out = run_step(
        ref,
        deg,
        r_p,
        r_idx,
        prep_deg,
        jnp.int32(n_rp),
        jnp.int32(dv),
        jnp.int32(dl),
    )
    assert out.similarity.dtype == jnp.float32
    assert out.similarity.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(out.similarity)))
  assert traces == 1

  # Gradients through coarse DP stay finite on silence, active, and vc=0.
  dp_grad_fn = jax.jit(
      jax.grad(
          lambda ds, dv: jnp.sum(
              selector.find_most_optimal_deg_patches(
                  jnp.ones((1, 3, 4), dtype=jnp.float32),
                  jnp.array([0], dtype=jnp.int32),
                  ds,
                  0.02,
                  search_window_radius=2,
                  deg_valid_cols=dv,
              ).similarity
          )
      )
  )
  i6 = jnp.int32(6)
  assert bool(jnp.all(jnp.isfinite(dp_grad_fn(jnp.zeros((3, 6)), i6))))
  assert bool(jnp.all(jnp.isfinite(dp_grad_fn(prep_deg[:3, :6], i6))))
  _assert_array_equal(dp_grad_fn(prep_deg[:3, :6], jnp.int32(0)), 0.0)

  # Extreme search/index bounds and batched vmap over slice/max-patches.
  bound_dp = jax.jit(
      lambda idx: selector.find_most_optimal_deg_patches(
          jnp.ones((2, 3, 2), dtype=jnp.float32),
          idx,
          jnp.ones((3, 6), dtype=jnp.float32),
          0.02,
          search_window_radius=10**12,
      )
  )
  hi_res = bound_dp(jnp.array([0x7FFFFFFF, 1], dtype=jnp.int32))
  assert int(hi_res.num_patches) == 2
  assert bool(jnp.all(jnp.isfinite(hi_res.similarity)))
  neg_traced = bound_dp(jnp.array([-5, 1], dtype=jnp.int32))
  assert int(neg_traced.num_patches) == 2

  v_slices, v_lens = jax.vmap(
      lambda st: selector.slice_audio(ref, st, st + 0.1, sr, max_len=210)
  )(jnp.array([0.0, 0.05], dtype=jnp.float32))
  assert v_slices.shape == (2, 210) and v_lens.shape == (2,)
  v_cnts = jax.vmap(
      lambda df: selector.calc_max_num_patches(r_idx, df, ps)
  )(jnp.array([0, 15], dtype=jnp.int32))
  _assert_array_equal(v_cnts, [1, 2])

  # Under x64=True, float64 inputs coerce to float32 and f32 HLO has zero f64.
  with jax.enable_x64(True):
    coarse_x64 = selector.find_most_optimal_deg_patches(
        r_p.astype(jnp.float64),
        r_idx,
        prep_deg.astype(jnp.float64),
        fd,
        4,
    )
    assert coarse_x64.similarity.dtype == jnp.float32
    assert selector.build_degraded_patch(
        prep_deg.astype(jnp.float64), 0, 4
    ).dtype == jnp.float32
    assert selector.slice_audio(
        ref.astype(jnp.float64), 0.0, 0.1, sr
    )[0].dtype == jnp.float32

    hlo_f32 = run_step.lower(
        jax.ShapeDtypeStruct(ref.shape, jnp.float32),
        jax.ShapeDtypeStruct(deg.shape, jnp.float32),
        jax.ShapeDtypeStruct(r_p.shape, jnp.float32),
        jax.ShapeDtypeStruct(r_idx.shape, jnp.int32),
        jax.ShapeDtypeStruct(prep_deg.shape, jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    ).as_text()
    assert {int(w) for w in _FLOAT_WIDTH.findall(hlo_f32)} == {32}
    hlo_win = (
        jax.jit(
            lambda rp, ri, ds: selector.find_most_optimal_deg_patches(
                rp, ri, ds, fd, search_window_radius=1
            )
        )
        .lower(
            jax.ShapeDtypeStruct((2, 6, 2), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((6, 15), jnp.float32),
        )
        .as_text()
    )
    assert {int(w) for w in _FLOAT_WIDTH.findall(hlo_win)} == {32}

    lowered_f64 = run_step.lower(
        ref.astype(jnp.float64),
        deg.astype(jnp.float64),
        r_p.astype(jnp.float64),
        r_idx,
        prep_deg.astype(jnp.float64),
        jnp.int32(2),
        jnp.int32(15),
        jnp.int32(n),
    )
    for out_aval in jax.tree.leaves(lowered_f64.out_info):
      assert out_aval.dtype in (jnp.float32, jnp.int32)
    hlo_f64_in = lowered_f64.as_text()
    fft_widths = []
    for line in hlo_f64_in.splitlines():
      m = _FFT_LINE.match(line)
      if m is not None:
        found = [int(w) for w in _FLOAT_WIDTH.findall(m.group("types"))]
        assert found
        fft_widths.extend(found)
    assert fft_widths and max(fft_widths) <= 32

  # Zero-patch (P=0) input to finely_align_and_recreate_patches returns empty.
  empty_sr = selector.PatchSimilarityResult(
      similarity=jnp.zeros((0,), dtype=jnp.float32),
      freq_band_means=jnp.zeros((0, 21), dtype=jnp.float32),
      freq_band_stddevs=jnp.zeros((0, 21), dtype=jnp.float32),
      freq_band_deg_energy=jnp.zeros((0, 21), dtype=jnp.float32),
      ref_patch_start_time=jnp.zeros((0,), dtype=jnp.float32),
      ref_patch_end_time=jnp.zeros((0,), dtype=jnp.float32),
      deg_patch_start_time=jnp.zeros((0,), dtype=jnp.float32),
      deg_patch_end_time=jnp.zeros((0,), dtype=jnp.float32),
      num_patches=jnp.int32(0),
  )
  empty_fine = selector.finely_align_and_recreate_patches(
      empty_sr, ref, deg, sr, use_speech_mode=True
  )
  assert empty_fine.similarity.shape == (0,)
  assert int(empty_fine.num_patches) == 0

  ones_rp = jnp.ones((1, 3, 2), dtype=jnp.float32)
  ones_ds = jnp.ones((3, 4), dtype=jnp.float32)
  zero_i = jnp.array([0], jnp.int32)
  with pytest.raises(ValueError, match="search_window_radius"):
    selector.find_most_optimal_deg_patches(ones_rp, zero_i, ones_ds, 0.02, -1)
  for bad_fd in (
      0.0,
      -0.02,
      True,
      _NP_BOOL_TRUE,
      float("nan"),
      float("inf"),
      1e300,
      1e-300,
      jnp.array(0.0),
      jnp.array(jnp.nan),
      jnp.array([0.02]),
  ):
    with pytest.raises(ValueError, match="frame_duration"):
      selector.find_most_optimal_deg_patches(
          ones_rp, zero_i, ones_ds, bad_fd, 1
      )
  with pytest.raises(ValueError, match="integer dtype"):
    selector.find_most_optimal_deg_patches(
        ones_rp, jnp.array([0.5]), ones_ds, 0.02, 1
    )
  with pytest.raises(ValueError, match="non-negative"):
    selector.find_most_optimal_deg_patches(
        ones_rp, jnp.array([-1], jnp.int32), ones_ds, 0.02, 1
    )
  with pytest.raises(ValueError, match="band count mismatch"):
    selector.find_most_optimal_deg_patches(
        jnp.ones((1, 4, 2)), zero_i, ones_ds, 0.02, 1
    )
  ones10 = jnp.ones((10,), dtype=jnp.float32)
  for bad_sr in (
      0.0,
      -100.0,
      True,
      _NP_BOOL_TRUE,
      float("nan"),
      1e300,
      1e-300,
      jnp.array(0),
  ):
    with pytest.raises(ValueError, match="sample_rate"):
      selector.slice_audio(ones10, 0.0, 0.5, bad_sr)
  for bad_t in (
      True,
      _NP_BOOL_TRUE,
      float("nan"),
      float("inf"),
      1e300,
      jnp.array(jnp.nan),
  ):
    with pytest.raises(ValueError, match="start_time"):
      selector.slice_audio(ones10, bad_t, 0.5, 100)
    with pytest.raises(ValueError, match="end_time"):
      selector.slice_audio(ones10, 0.0, bad_t, 100)
  with pytest.raises(ValueError, match="use_speech_mode"):
    selector.finely_align_and_recreate_patches(
        empty_sr, ref, deg, sr, use_speech_mode="no"  # type: ignore[arg-type]
    )
  with pytest.raises(ValueError, match="PatchSimilarityResult"):
    selector.finely_align_and_recreate_patches(
        tuple(empty_sr),  # type: ignore[arg-type]
        ref,
        deg,
        sr,
        use_speech_mode=True,
    )
  with pytest.raises(ValueError, match="real-valued"):
    selector.finely_align_and_recreate_patches(
        empty_sr._replace(
            freq_band_means=jnp.zeros((0, 21), dtype=jnp.complex64)
        ),
        ref,
        deg,
        sr,
        use_speech_mode=True,
    )
  with pytest.raises(ValueError, match="smaller than window_size"):
    selector.finely_align_and_recreate_patches(
        empty_sr, ref, deg, sr, use_speech_mode=True, patch_size=1
    )
  with pytest.raises(ValueError, match="sample_rate"):
    selector.finely_align_and_recreate_patches(
        empty_sr, ref, deg, 10**400, use_speech_mode=True
    )
  # Speech-band (21) sim_results passed with audio mode (32 bands) must fail.
  with pytest.raises(ValueError, match="freq_band_means must have shape"):
    selector.finely_align_and_recreate_patches(
        empty_sr, ref, deg, sr, use_speech_mode=False
    )
