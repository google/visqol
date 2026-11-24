"""Tests for _check.py and alignment.py against C++ ViSQOL."""

import array
import math
import re

import jax
import jax.numpy as jnp
import pytest

from visqol_jax import _check
from visqol_jax import alignment

# Reference pairs from tests/misc_math_test.cc.
_CPP_NEXT_POW_TWO = (
    (2, 2), (10, 16), (3, 4), (5, 8), (48000, 65536), (7, 8), (23, 32), (32, 32)
)

# Exact 65-sample input and 128-pt FFT bins (fast_fourier_transform_test.cc).
_CPP_FFT_TOLERANCE = 1e-8
_CPP_65_SAMPLES = (
    0.000150529, 5.89739e-05, -9.36187e-05, -9.36187e-05, 0.000394677,
    0.000303122, -9.36187e-05, 0.000303122, 8.94924e-05, 0.000150529,
    -2.06314e-06, -0.000154656, 2.84554e-05, 0.000272603, -0.000185174,
    5.89739e-05, 0.000364159, 0.000425196, 0.000120011, 5.89739e-05,
    0.000120011, 0.000211566, 0.000150529, 2.84554e-05, 0.000120011,
    0.000150529, 8.94924e-05, -9.36187e-05, 0.000242085, 0.000486233,
    8.94924e-05, 0.000120011, 8.94924e-05, 8.94924e-05, 0.000150529,
    -0.000154656, -2.06314e-06, 2.84554e-05, 0.000150529, -3.25817e-05,
    5.89739e-05, 5.89739e-05, -0.000124137, 5.89739e-05, 2.84554e-05,
    0.000181048, -0.000124137, -0.000368285, -9.36187e-05, -0.000246211,
    -3.25817e-05, 2.84554e-05, -6.31002e-05, -9.36187e-05, -2.06314e-06,
    -2.06314e-06, -0.000185174, -0.000124137, 2.84554e-05, 8.94924e-05,
    -0.00027673, -6.31002e-05, -0.000215693, 0.000425196, 8.94924e-05,
)

_CPP_65_FORWARD_RFFT = (
    0.00322292904000000 + 0.0j, 0.00301682385173052 - 0.00234848985608509j,
    -0.0010327859216389 - 0.00340299713362453j,
    -0.00144609203801138 + 0.000879145244592523j,
    0.00111610613898786 - 0.000302181002376930j,
    -0.000318859126372253 - 0.000367181194435114j,
    0.000673949619008653 - 0.000310723599151972j,
    -3.49644446093239e-05 - 0.000563954656030864j,
    0.000339817207812471 - 0.000595601799964675j,
    0.000101609221059575 - 0.000744744002384681j,
    -0.000586264026928805 - 0.00191608882520754j,
    -0.00259248055250948 + 0.000471875974326385j,
    0.000227097152831817 + 0.00110287047767037j,
    -0.000722511594188183 + 0.000313570758095348j,
    0.000407311564417282 + 0.000853530053469442j,
    -0.000195617316848781 + 0.000154945338489743j,
    0.000233147810677542 + 0.000734349802916912j,
    0.000691246491351084 + 0.000556673374244994j,
    0.000755913153376688 - 0.00108748635434686j,
    -0.00157770746633928 + 0.000303794653066800j,
    0.000896182317264214 + 0.000835164328459981j,
    -0.000822997471347754 + 0.000140031434634188j,
    0.00109713925301471 + 0.00153219988568366j,
    0.000868409640532475 - 0.000373929386206932j,
    0.000535362381861835 - 0.000461656449099621j,
    -0.000994143865712840 + 0.000213392912383107j,
    0.00128029686672661 + 0.000927358696182279j,
    -0.000558399599341835 - 0.000596241613983044j,
    0.000590345970887055 + 0.00153746054605763j,
    0.000123623000739260 + 0.000159309107927759j,
    0.00177647225128173 + 0.00233356699195329j,
    0.00301793611579363 - 0.00119668950186613j,
    0.00124919604000000 - 0.00161748004000000j,
    -0.000322728346193144 - 0.00214236853854895j,
    -0.000527239999720400 + 0.000360874054041502j,
    0.000703316552686529 - 0.000897998866185932j,
    -0.000525863898635626 - 0.000277837377775431j,
    0.000432650139994782 - 9.95533541164137e-05j,
    0.000151305582086382 - 0.000519865718932982j,
    -6.14596442789589e-05 - 7.84463404614790e-06j,
    0.00118869378923226 - 0.000194024415891887j,
    0.000172206834052672 - 0.00166972697788257j,
    -0.000273651726999761 - 0.000867166897935699j,
    -0.00113046085098264 - 0.00125829455984567j,
    -0.000958346131576863 + 0.000520388328872528j,
    -0.000253487462883782 - 0.000415594953354160j,
    -0.000970643681183129 - 4.25544922587714e-05j,
    -0.00121987061492059 + 0.000551139974883648j,
    0.000189986069322458 + 0.00146679520291691j,
    3.91316837376560e-05 - 3.82684506226442e-05j,
    0.000364085132964744 + 0.00119600012367842j,
    0.000438119142196402 - 0.000741712630986012j,
    -0.000619592071135628 + 0.00114204782699505j,
    0.00128409408020558 + 0.000424538502147734j,
    0.000651940026079785 - 0.000330701565255067j,
    -0.000378907596869206 - 0.000711769960239768j,
    3.12982109343697e-06 + 0.00138106567324306j,
    0.00118819196612810 - 0.000672809033262345j,
    -0.000236735235269374 + 0.000463610199849060j,
    0.00121540962643760 - 0.000589114879992856j,
    -0.000254141478622833 - 5.44911772978580e-05j,
    0.00102320576458072 - 0.000954731288479258j,
    -0.00112262885721622 - 0.000516904492870956j,
    0.000267885080182829 - 0.000205545255713845j,
    -0.00110072968000000 + 0.0j,
)

# Exact signals from tests/{xcorr_test,alignment_test}.cc.
_CPP_REF_SIGNAL = (
    2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0
)
_CPP_DEG_LAG2 = (
    1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2, 2.1
)
_CPP_LONG_DEG_LAG2 = (*_CPP_DEG_LAG2, 2.0)
_CPP_SHORT_DEG_LAG2 = _CPP_DEG_LAG2[:13]
_CPP_DEG_NEG_LAG2 = (
    2.0, 2.0, 2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0
)

# Exact C++ Envelope::CalcUpperEnv(_CPP_REF_SIGNAL) output.
_CPP_REF_UPPER_ENV = (
    2.36071430, 3.01071425, 4.36071421, 6.81071426, 12.36071445,
    6.81071426, 4.36071421, 3.01071425, 2.36071430, 9.44642879,
    14.09642841, 9.44642879, 2.36071430, 3.01071425,
)

_MAX_FFT_WIDTH = 32
_FFT_LINE = re.compile(r"^\s*%\S+ = stablehlo\.fft .*?:(?P<types>.*)$")
_FLOAT_WIDTH = re.compile(r"(?:<|x|\b)f(?P<width>16|32|64)\b")
_FakeBoolScalar = type("bool_", (), {"__index__": lambda self: 1})
_NP_BOOL_TRUE = jnp.dtype(jnp.bool_).type(True)


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
  assert bool(jnp.allclose(a, b, rtol=rtol, atol=atol)), (
      f"max diff {float(jnp.max(jnp.abs(a - b)))}"
  )


def _fft_float_widths(hlo_text: str) -> list[int]:
  widths: list[int] = []
  for line in hlo_text.splitlines():
    match = _FFT_LINE.match(line)
    if match is None:
      continue
    types = match.group("types")
    found = _FLOAT_WIDTH.findall(types)
    assert found, f"FFT operand of unexpected type {types.strip()}"
    widths.extend(int(width) for width in found)
  return widths


def _assert_single_precision_transforms(hlo_text: str) -> None:
  widths = _fft_float_widths(hlo_text)
  assert widths, "expected at least one stablehlo.fft op in lowered IR"
  offending = sorted(w for w in set(widths) if w > _MAX_FFT_WIDTH)
  assert not offending, f"FFT operands wider than float32: {offending}"


def _synth_burst(sample_rate: int, n_samples: int, delay: int) -> jax.Array:
  """Deterministic amplitude-modulated burst shifted by `delay` samples."""
  t = jnp.arange(n_samples, dtype=jnp.float32) / jnp.float32(sample_rate)
  two_pi = jnp.float32(2.0 * math.pi)
  carrier = (
      jnp.sin(two_pi * 220.0 * t)
      + 0.6 * jnp.sin(two_pi * 510.0 * t + 0.4)
      + 0.3 * jnp.cos(two_pi * 1130.0 * t - 0.7)
  )
  env = jnp.exp(-0.5 * ((t - 0.35) / 0.055) ** 2) + 0.7 * jnp.exp(
      -0.5 * ((t - 0.68) / 0.04) ** 2
  )
  sig = env * carrier
  if delay > 0:
    return jnp.pad(sig[: n_samples - delay], (delay, 0))
  if delay < 0:
    return jnp.pad(sig[-delay:], (0, -delay))
  return sig


def test_check_helpers_and_asarray_f32():
  assert _check.is_bool(True) and _check.is_bool(False)
  assert _check.is_bool(_FakeBoolScalar()) and _check.is_bool(_NP_BOOL_TRUE)
  assert not _check.is_bool(1) and not _check.is_bool(bool)
  assert not _check.is_bool(jnp.array(True))

  _check.check_1d_nonempty(jnp.ones((3,)), "x")
  for bad_1d in (jnp.array(1.0), jnp.zeros((0,)), jnp.ones((2, 3))):
    with pytest.raises(ValueError, match="non-empty 1D"):
      _check.check_1d_nonempty(bad_1d, "x")

  _check.check_2d_nonempty(jnp.ones((2, 3)), "m")
  for bad_2d in (
      jnp.ones((3,)), jnp.zeros((0, 3)), jnp.zeros((2, 0)), jnp.ones((1, 2, 3))
  ):
    with pytest.raises(ValueError, match="non-empty 2D"):
      _check.check_2d_nonempty(bad_2d, "m")

  assert _check.check_positive_int(1, "k") == 1
  assert _check.check_positive_int(jnp.int32(7), "k") == 7
  for bad_pos in (
      0,
      -2,
      True,
      False,
      _FakeBoolScalar(),
      _NP_BOOL_TRUE,
      jnp.array(True),
      1.5,
      "3",
  ):
    with pytest.raises(ValueError, match="positive int"):
      _check.check_positive_int(bad_pos, "k")

  assert _check.check_nonneg_int(0, "k") == 0
  assert _check.check_nonneg_int(jnp.int32(4), "k") == 4
  for bad_nn in (
      -1,
      True,
      False,
      _FakeBoolScalar(),
      _NP_BOOL_TRUE,
      jnp.array(False),
      0.0,
      None,
  ):
    with pytest.raises(ValueError, match="non-negative int"):
      _check.check_nonneg_int(bad_nn, "k")

  for valid_in in (
      3,
      2.5,
      2**40,
      2**70,
      [1.0, 2.0],
      (1, 2),
      array.array("d", [1.0, -2.0]),
      jnp.ones((2,), dtype=jnp.int16),
      jnp.ones((2,), dtype=jnp.float16),
      jnp.ones((2,), dtype=jnp.bfloat16),
      jnp.ones((2,), dtype=jnp.float32),
  ):
    out = _check.asarray_f32(valid_in, "x")
    assert isinstance(out, jax.Array) and out.dtype == jnp.float32
  assert float(_check.asarray_f32(2**40, "x")) == float(2**40)
  assert float(_check.asarray_f32(2**70, "x")) == float(2**70)

  for bad_bool in (
      True,
      False,
      _FakeBoolScalar(),
      _NP_BOOL_TRUE,
      jnp.array(True),
      jnp.ones((3,), dtype=jnp.bool_),
      [True, False],
  ):
    with pytest.raises(ValueError, match=r"numeric.*bool"):
      _check.asarray_f32(bad_bool, "x")

  for bad_complex in (
      1 + 2j,
      jnp.zeros((4,), dtype=jnp.complex64),
      [1.0, 2.0 + 3.0j],
  ):
    with pytest.raises(ValueError, match="real-valued"):
      _check.asarray_f32(bad_complex, "x")

  for bad_other in ("invalid", None, object(), ["a"], jnp.float32, 10**400):
    with pytest.raises(ValueError, match="numeric"):
      _check.asarray_f32(bad_other, "x")


def test_next_pow_two_and_float32_fft_match_cpp_tests():
  for n, expected in _CPP_NEXT_POW_TWO:
    assert alignment.next_pow_two(n) == expected
  assert alignment.next_pow_two(0) == 0
  assert alignment.next_pow_two(1) == 1
  assert alignment.next_pow_two(jnp.int32(23)) == 32
  assert alignment.fft_size(0) == alignment.fft_size(1) == 32
  assert alignment.fft_size(32) == 32
  with pytest.raises(ValueError, match="non-negative int"):
    alignment.next_pow_two(-1)

  assert alignment.fft_size(len(_CPP_65_SAMPLES)) == 128
  spec = jnp.fft.rfft(jnp.asarray(_CPP_65_SAMPLES, dtype=jnp.float32), n=128)
  assert spec.dtype == jnp.complex64
  _assert_allclose(spec, _CPP_65_FORWARD_RFFT, atol=_CPP_FFT_TOLERANCE)

  # Forward1d_Resize_MatrixContentsRandom: 65 -> 66 samples keeps fft_size=128.
  resized_66 = (*_CPP_65_SAMPLES, 0.0)
  assert alignment.fft_size(len(resized_66)) == 128
  spec_66 = jnp.fft.rfft(jnp.asarray(resized_66, dtype=jnp.float32), n=128)
  _assert_allclose(spec_66, _CPP_65_FORWARD_RFFT, atol=_CPP_FFT_TOLERANCE)

  # FftReconstructConjsym and FftReconstructConjsymZero.
  recon = jnp.fft.irfft(spec, n=128)[: len(_CPP_65_SAMPLES)]
  _assert_allclose(recon, _CPP_65_SAMPLES, atol=_CPP_FFT_TOLERANCE)
  zeros_65 = jnp.zeros((65,), dtype=jnp.float32)
  zero_recon = jnp.fft.irfft(jnp.fft.rfft(zeros_65, n=128), n=128)[:65]
  assert bool(jnp.array_equal(zero_recon, zeros_65))


def test_calc_upper_env_hilbert_quirks_masking_and_validation():
  env_ref = alignment.calc_upper_env(_CPP_REF_SIGNAL)
  assert env_ref.dtype == jnp.float32 and env_ref.shape == (14,)
  _assert_allclose(env_ref, _CPP_REF_UPPER_ENV, atol=2e-6)

  # Single-element signal (valid_len == 1 -> DC weight 2.0) equals the sample.
  single = jnp.array([4.25], dtype=jnp.float32)
  _assert_allclose(alignment.calc_upper_env(single), single, atol=1e-6)
  padded_single = jnp.array([4.25, 99.0, -50.0], dtype=jnp.float32)
  _assert_allclose(
      alignment.calc_upper_env(padded_single, valid_len=1),
      [4.25, 0.0, 0.0],
      atol=1e-6,
  )

  const = jnp.full((40,), 3.5, dtype=jnp.float32)
  _assert_allclose(alignment.calc_upper_env(const), 3.5, atol=1e-6)

  # Same fft_size bucket: padded valid_len=31 matches unpadded 31-sample slice.
  sig31 = jax.random.normal(jax.random.key(7), (31,), dtype=jnp.float32)
  padded32 = jnp.concatenate([sig31, jnp.array([999.0], dtype=jnp.float32)])
  env_unpadded = alignment.calc_upper_env(sig31)
  env_masked = alignment.calc_upper_env(padded32, valid_len=31)
  _assert_allclose(env_masked[:31], env_unpadded, atol=1e-6)
  assert float(env_masked[31]) == 0.0

  # valid_len == fft_size (32) switches Nyquist weight from 0.0 to 1.0.
  mean32 = jnp.mean(padded32)
  c32 = padded32 - mean32
  signs = jnp.where(jnp.arange(32) % 2 == 0, 1.0, -1.0)
  alt = jnp.sum(signs * c32) / jnp.float32(32.0)
  want_full = jnp.abs(2.0 * c32 - signs * alt) + mean32
  env_full = alignment.calc_upper_env(padded32)
  env_full_masked = alignment.calc_upper_env(padded32, valid_len=32)
  _assert_allclose(env_full, want_full, atol=2e-4)
  _assert_allclose(env_full_masked, env_full, atol=1e-6)

  assert bool(
      jnp.array_equal(
          alignment.calc_upper_env(padded32, valid_len=0),
          jnp.zeros((32,), dtype=jnp.float32),
      )
  )
  _assert_allclose(
      alignment.calc_upper_env(padded32, valid_len=jnp.uint8(31))[:31],
      env_unpadded,
      atol=1e-6,
  )
  _assert_allclose(
      alignment.calc_upper_env(padded32, valid_len=jnp.uint32(0xFFFFFFFF)),
      env_full,
      atol=1e-6,
  )
  _assert_allclose(
      alignment.calc_upper_env(padded32, valid_len=10**30),
      env_full,
      atol=1e-6,
  )
  with jax.enable_x64(True):
    _assert_allclose(
        alignment.calc_upper_env(padded32, valid_len=jnp.int64(1 << 31)),
        env_full,
        atol=1e-6,
    )
    assert bool(
        jnp.array_equal(
            alignment.calc_upper_env(
                padded32, valid_len=jnp.int64(-(1 << 32) + 5)
            ),
            jnp.zeros((32,), dtype=jnp.float32),
        )
    )

  for bad_sig in (jnp.zeros((2, 4)), jnp.zeros((0,))):
    with pytest.raises(ValueError, match="non-empty 1D"):
      alignment.calc_upper_env(bad_sig)
  for bad_len in (
      True, _FakeBoolScalar(), _NP_BOOL_TRUE, 1.5, "3", jnp.array([1, 2])
  ):
    with pytest.raises(ValueError, match="integer scalar"):
      alignment.calc_upper_env(padded32, valid_len=bad_len)


def test_xcorr_best_lag_matches_cpp_unit_tests_and_masking():
  for deg, want in (
      (_CPP_DEG_LAG2, 2),
      (_CPP_LONG_DEG_LAG2, 2),
      (_CPP_SHORT_DEG_LAG2, 2),
      (_CPP_DEG_NEG_LAG2, -2),
  ):
    lag = alignment.find_lowest_lag_index(_CPP_REF_SIGNAL, deg)
    assert lag.dtype == jnp.int32 and int(lag) == want

  # max_lag == 0 must not slice the full array via res[-0:].
  assert int(alignment.find_lowest_lag_index([1.5], [2.5])) == 0
  # Small lengths where fft_points < 32 (M = 5, 8 -> fft_points = 16).
  s5 = [0.0, 0.0, 1.0, 3.0, 1.0]
  d5 = [1.0, 3.0, 1.0, 0.0, 0.0]
  assert int(alignment.find_lowest_lag_index(s5, d5)) == 2
  assert int(alignment.find_lowest_lag_index(d5, s5)) == 0
  s8 = [*s5, 0.0, 0.0, 0.0]
  d8 = [*d5, 0.0, 0.0, 0.0]
  assert int(alignment.find_lowest_lag_index(s8, d8)) == 2
  assert int(alignment.find_lowest_lag_index(d8, s8)) == 0

  # Static-shape padding with junk past len_1 / len_2 matches unpadded calls.
  ref_pad = (*_CPP_REF_SIGNAL, 100.0, 100.0)
  deg_short_pad = (*_CPP_SHORT_DEG_LAG2, -100.0, -100.0, -100.0)
  deg_neg_pad = (*_CPP_DEG_NEG_LAG2, 50.0, 50.0)
  lag_short = alignment.find_lowest_lag_index(
      ref_pad,
      deg_short_pad,
      len_1=len(_CPP_REF_SIGNAL),
      len_2=len(_CPP_SHORT_DEG_LAG2),
  )
  lag_neg = alignment.find_lowest_lag_index(
      ref_pad,
      deg_neg_pad,
      len_1=len(_CPP_REF_SIGNAL),
      len_2=len(_CPP_DEG_NEG_LAG2),
  )
  assert int(lag_short) == 2 and int(lag_neg) == -2
  assert int(alignment.find_lowest_lag_index(ref_pad, ref_pad, len_1=0)) == 0
  assert int(alignment.find_lowest_lag_index(ref_pad, ref_pad, len_2=0)) == 0

  for bad_sig in (jnp.array(1.0), jnp.zeros((0,)), jnp.ones((2, 4))):
    with pytest.raises(ValueError, match="signal_1.*non-empty 1D"):
      alignment.find_lowest_lag_index(bad_sig, _CPP_REF_SIGNAL)
    with pytest.raises(ValueError, match="signal_2.*non-empty 1D"):
      alignment.find_lowest_lag_index(_CPP_REF_SIGNAL, bad_sig)


def test_globally_align_and_align_and_truncate_match_cpp_tests():
  aligned_p, vlen_p, lag_p = alignment.globally_align(
      _CPP_REF_SIGNAL, _CPP_DEG_LAG2, 1
  )
  assert aligned_p.dtype == jnp.float32 and lag_p.dtype == jnp.float32
  assert aligned_p.shape == (len(_CPP_DEG_LAG2) + len(_CPP_REF_SIGNAL) // 2,)
  assert int(vlen_p) == 16 and float(lag_p) == pytest.approx(2.0)
  _assert_allclose(aligned_p[:2], 0.0)
  _assert_allclose(aligned_p[2:16], _CPP_DEG_LAG2)
  assert bool(jnp.all(aligned_p[16:] == 0.0))
  final_lag_p = alignment.find_lowest_lag_index(
      _CPP_REF_SIGNAL, aligned_p, len_2=vlen_p
  )
  assert int(final_lag_p) == 0

  aligned_n, vlen_n, lag_n = alignment.globally_align(
      _CPP_REF_SIGNAL, _CPP_DEG_NEG_LAG2, 1
  )
  assert int(vlen_n) == 12 and float(lag_n) == pytest.approx(-2.0)
  _assert_allclose(aligned_n[:12], _CPP_DEG_NEG_LAG2[2:])
  assert bool(jnp.all(aligned_n[12:] == 0.0))
  final_lag_n = alignment.find_lowest_lag_index(
      _CPP_REF_SIGNAL, aligned_n, len_2=vlen_n
  )
  assert int(final_lag_n) == 0

  aligned_z, vlen_z, lag_z = alignment.globally_align(
      _CPP_REF_SIGNAL, _CPP_REF_SIGNAL, jnp.int32(1)
  )
  assert int(vlen_z) == 14 and float(lag_z) == 0.0
  _assert_allclose(aligned_z[:14], _CPP_REF_SIGNAL)

  ref_tn, deg_tn, vlen_tn, lag_tn = alignment.align_and_truncate(
      _CPP_REF_SIGNAL, _CPP_DEG_NEG_LAG2, 1
  )
  assert int(vlen_tn) == 12 and float(lag_tn) == pytest.approx(-2.0)
  _assert_allclose(ref_tn[:12], _CPP_REF_SIGNAL[:12])
  _assert_allclose(deg_tn[:12], _CPP_DEG_NEG_LAG2[2:])
  assert bool(jnp.all(ref_tn[12:] == 0.0)) and bool(jnp.all(deg_tn[12:] == 0.0))

  ref_tp, deg_tp, vlen_tp, lag_tp = alignment.align_and_truncate(
      _CPP_REF_SIGNAL, _CPP_DEG_LAG2, 1
  )
  assert int(vlen_tp) == 12 and float(lag_tp) == pytest.approx(2.0)
  _assert_allclose(ref_tp[:12], _CPP_REF_SIGNAL[2:])
  _assert_allclose(deg_tp[:12], _CPP_DEG_LAG2[:12])
  assert bool(jnp.all(ref_tp[12:] == 0.0)) and bool(jnp.all(deg_tp[12:] == 0.0))

  # Equal aligned lengths after positive lag (12 + 2 == 14) keep leading zeros.
  deg_12 = _CPP_DEG_LAG2[:12]
  ref_eq, deg_eq, vlen_eq, lag_eq = alignment.align_and_truncate(
      _CPP_REF_SIGNAL, deg_12, 1
  )
  assert int(vlen_eq) == 14 and float(lag_eq) == pytest.approx(2.0)
  _assert_allclose(ref_eq, _CPP_REF_SIGNAL)
  assert bool(jnp.all(deg_eq[:2] == 0.0))
  _assert_allclose(deg_eq[2:], deg_12)

  # Longer degraded clips with zero or negative lag clamp start to 0 safely.
  deg_long_z = (*_CPP_REF_SIGNAL, 1.0, 2.0, 3.0)
  ref_lz, deg_lz, vlen_lz, lag_lz = alignment.align_and_truncate(
      _CPP_REF_SIGNAL, deg_long_z, 1
  )
  assert int(vlen_lz) == 14 and float(lag_lz) == 0.0
  _assert_allclose(ref_lz, _CPP_REF_SIGNAL)
  _assert_allclose(deg_lz, _CPP_REF_SIGNAL)

  deg_long_n = (*_CPP_DEG_NEG_LAG2, 2.0, 2.0, 2.0, 2.0)
  ref_ln, deg_ln, vlen_ln, lag_ln = alignment.align_and_truncate(
      _CPP_REF_SIGNAL, deg_long_n, 1
  )
  assert int(vlen_ln) == 14 and float(lag_ln) == pytest.approx(-2.0)
  _assert_allclose(ref_ln, _CPP_REF_SIGNAL)
  _assert_allclose(deg_ln, deg_long_n[2:16])

  # |lag| > len(ref) / 2.0 clamps to 0; boundary |lag| == len(ref) // 2 is kept.
  r20 = jnp.zeros((20,), dtype=jnp.float32).at[15].set(5.0)
  d20 = jnp.zeros((20,), dtype=jnp.float32).at[3].set(5.0)
  d_exact = jnp.zeros((20,), dtype=jnp.float32).at[5].set(5.0)
  _, vlen_c, lag_c = alignment.globally_align(r20, d20, 16000)
  assert float(lag_c) == 0.0 and int(vlen_c) == 20
  _, vlen_e, lag_e = alignment.globally_align(r20, d_exact, 10)
  assert float(lag_e) == pytest.approx(1.0) and int(vlen_e) == 30

  r21 = jnp.zeros((21,), dtype=jnp.float32).at[15].set(5.0)
  d21_keep = jnp.zeros((21,), dtype=jnp.float32).at[5].set(5.0)
  d21_clamp = jnp.zeros((21,), dtype=jnp.float32).at[4].set(5.0)
  _, v21_k, lag21_k = alignment.globally_align(r21, d21_keep, 1)
  _, v21_c, lag21_c = alignment.globally_align(r21, d21_clamp, 1)
  assert float(lag21_k) == pytest.approx(10.0) and int(v21_k) == 31
  assert float(lag21_c) == 0.0 and int(v21_c) == 21

  # Positive-lag truncation under jax.jit at 48 kHz.
  ref_27 = jnp.zeros((128,), dtype=jnp.float32).at[60].set(5.0)
  deg_27 = jnp.zeros((128,), dtype=jnp.float32).at[33].set(5.0)
  r27, d27, v27, lag27 = jax.jit(
      lambda r, d: alignment.align_and_truncate(r, d, 48000)
  )(ref_27, deg_27)
  assert float(lag27) == pytest.approx(27.0 / 48000.0, rel=1e-6)
  n27 = int(v27)
  assert n27 in (101, 102)
  start27 = 128 - n27
  _assert_allclose(r27[:n27], ref_27[start27:])
  assert bool(jnp.all(r27[n27:] == 0.0)) and bool(jnp.all(d27[n27:] == 0.0))

  # vmap over masked batch including non-zero junk in a zero valid_len row.
  def pad16(seq: tuple[float, ...]) -> jax.Array:
    arr = jnp.asarray(seq, dtype=jnp.float32)
    return jnp.pad(arr, (0, 16 - arr.shape[0]), constant_values=123.0)

  junk16 = jnp.full((16,), 123.0, dtype=jnp.float32)
  refs = jnp.stack([pad16(_CPP_REF_SIGNAL)] * 4 + [junk16])
  degs = jnp.stack([
      pad16(_CPP_DEG_LAG2),
      pad16(_CPP_DEG_NEG_LAG2),
      pad16(_CPP_REF_SIGNAL),
      pad16(_CPP_SHORT_DEG_LAG2),
      junk16,
  ])
  vmap_fn = jax.jit(
      jax.vmap(
          lambda r, d, rl, dl: alignment.align_and_truncate(
              r, d, 1, ref_len=rl, deg_len=dl
          )
      )
  )
  out_r, out_d, out_v, out_lag = vmap_fn(
      refs,
      degs,
      jnp.array([14, 14, 14, 14, 0], dtype=jnp.int32),
      jnp.array([14, 14, 14, 13, 0], dtype=jnp.int32),
  )
  for i, d_i in enumerate(
      (_CPP_DEG_LAG2, _CPP_DEG_NEG_LAG2, _CPP_REF_SIGNAL, _CPP_SHORT_DEG_LAG2)
  ):
    er, ed, ev, elag = alignment.align_and_truncate(_CPP_REF_SIGNAL, d_i, 1)
    n = int(ev)
    assert int(out_v[i]) == n and float(out_lag[i]) == pytest.approx(
        float(elag)
    )
    _assert_allclose(out_r[i, :n], er[:n])
    _assert_allclose(out_d[i, :n], ed[:n])
    assert bool(jnp.all(out_r[i, n:] == 0.0))
    assert bool(jnp.all(out_d[i, n:] == 0.0))
  assert int(out_v[4]) == 0 and float(out_lag[4]) == 0.0
  assert bool(jnp.all(out_r[4] == 0.0)) and bool(jnp.all(out_d[4] == 0.0))

  # Unequal buffer lengths with deg_len / ref_len mask out trailing junk.
  deg_pad20 = jnp.pad(
      jnp.asarray(_CPP_SHORT_DEG_LAG2, dtype=jnp.float32),
      (0, 7),
      constant_values=-456.0,
  )
  ga_m, ga_vm, ga_lm = alignment.globally_align(
      pad16(_CPP_REF_SIGNAL), deg_pad20, 1, ref_len=14, deg_len=13
  )
  assert int(ga_vm) == 15 and float(ga_lm) == pytest.approx(2.0)
  assert bool(jnp.all(ga_m[:2] == 0.0))
  _assert_allclose(ga_m[2:15], _CPP_SHORT_DEG_LAG2)
  assert bool(jnp.all(ga_m[15:] == 0.0))
  at_rm, at_dm, at_vm, at_lm = alignment.align_and_truncate(
      _CPP_REF_SIGNAL, deg_pad20, 1, deg_len=13
  )
  assert int(at_vm) == 12 and float(at_lm) == pytest.approx(2.0)
  _assert_allclose(at_rm[:12], _CPP_REF_SIGNAL[2:])
  _assert_allclose(at_dm[:12], _CPP_SHORT_DEG_LAG2[:12])
  assert bool(jnp.all(at_rm[12:] == 0.0)) and bool(jnp.all(at_dm[12:] == 0.0))

  # Traced scalar sample_rate works under jax.jit; invalid scalars fail eagerly.
  traced_sr = jax.jit(alignment.globally_align)
  _, v_tr, lag_tr = traced_sr(_CPP_REF_SIGNAL, _CPP_DEG_LAG2, jnp.float32(2.0))
  assert int(v_tr) == 16 and float(lag_tr) == pytest.approx(1.0)

  for bad_sr in (
      0,
      -16000,
      True,
      False,
      _FakeBoolScalar(),
      _NP_BOOL_TRUE,
      float("nan"),
      float("inf"),
      1e300,
      1e-300,
      16000 + 0j,
      "16000",
      None,
      jnp.array(0.0),
      jnp.array(-16000.0),
      jnp.array(jnp.nan),
      jnp.array(jnp.inf),
      jnp.array(True),
      jnp.array(16000 + 0j),
      jnp.array([16000, 48000]),
  ):
    with pytest.raises(ValueError, match="sample_rate"):
      alignment.globally_align(r20, d20, bad_sr)

  for bad_sig in (jnp.array(1.0), jnp.zeros((0,)), jnp.ones((2, 4))):
    with pytest.raises(ValueError, match="reference.*non-empty 1D"):
      alignment.globally_align(bad_sig, d20, 16000)
    with pytest.raises(ValueError, match="degraded.*non-empty 1D"):
      alignment.globally_align(r20, bad_sig, 16000)


def test_synthetic_waveforms_silence_and_single_jit_trace():
  jit_align = jax.jit(alignment.globally_align, static_argnums=(2,))
  ref_16k = _synth_burst(16000, 16000, 0)
  deg_16k_delayed = _synth_burst(16000, 16000, 800)
  deg_16k_early = _synth_burst(16000, 16000, -800)

  aln_neg, vlen_neg, lag_neg = jit_align(ref_16k, deg_16k_delayed, 16000)
  assert aln_neg.dtype == jnp.float32 and lag_neg.dtype == jnp.float32
  assert float(lag_neg) == pytest.approx(-0.05, abs=1e-6)
  assert int(vlen_neg) == 15200

  _, vlen_pos, lag_pos = jit_align(ref_16k, deg_16k_early, 16000)
  assert float(lag_pos) == pytest.approx(0.05, abs=1e-6)
  assert int(vlen_pos) == 16800

  _, vlen_mm, lag_mm = jit_align(ref_16k, deg_16k_early[:12000], 16000)
  assert float(lag_mm) == pytest.approx(0.05, abs=1e-6)
  assert int(vlen_mm) == 12800

  ref_48k = _synth_burst(48000, 24000, 0)
  deg_48k_delayed = _synth_burst(48000, 24000, 1200)
  _, vlen_48k, lag_48k = jit_align(ref_48k, deg_48k_delayed, 48000)
  assert float(lag_48k) == pytest.approx(-0.025, abs=1e-6)
  assert int(vlen_48k) == 22800

  # Silence: XCorr picks -max_lag first; GloballyAlign clamps > N/2 to 0.
  sil14 = jnp.zeros((14,), dtype=jnp.float32)
  sil16 = jnp.zeros((16,), dtype=jnp.float32)
  assert int(alignment.find_lowest_lag_index(sil14, sil14)) == -13
  sil_masked = alignment.find_lowest_lag_index(sil16, sil16, len_1=14, len_2=14)
  assert int(sil_masked) == -13
  for n_sil, sr in ((16000, 16000), (24000, 48000)):
    zeros = jnp.zeros((n_sil,), dtype=jnp.float32)
    aln_s, vlen_s, lag_s = jit_align(zeros, zeros, sr)
    assert int(vlen_s) == n_sil and float(lag_s) == 0.0
    assert bool(jnp.all(aln_s == 0.0))
    tr_r, tr_d, tr_v, tr_lag = alignment.align_and_truncate(zeros, zeros, sr)
    assert int(tr_v) == n_sil and float(tr_lag) == 0.0
    assert bool(jnp.all(tr_r == 0.0)) and bool(jnp.all(tr_d == 0.0))

  aln_one_sil, vlen_one_sil, lag_one_sil = jit_align(
      ref_16k, jnp.zeros_like(ref_16k), 16000
  )
  assert int(vlen_one_sil) == 16000 and float(lag_one_sil) == 0.0
  assert bool(jnp.all(aln_one_sil == 0.0))

  # Static-shape JIT compiles once across zero, pos, neg, and clamped lags.
  traces = 0

  @jax.jit
  def run_both(ref: jax.Array, deg: jax.Array):
    nonlocal traces
    traces += 1
    return (
        alignment.globally_align(ref, deg, 16000),
        alignment.align_and_truncate(ref, deg, 16000),
    )

  n = 128
  base = jnp.zeros((n,), dtype=jnp.float32).at[50].set(4.0)
  ref_late = jnp.zeros((n,), dtype=jnp.float32).at[100].set(4.0)
  deg_clamped = jnp.zeros((n,), dtype=jnp.float32).at[0].set(4.0)
  for r, d, want_lag in (
      (base, base, 0),
      (base, jnp.roll(base, -12), 12),
      (base, jnp.roll(base, 15), -15),
      (ref_late, deg_clamped, 0),
  ):
    (ga_d, ga_v, ga_lag), (at_r, at_d, at_v, at_lag) = run_both(r, d)
    assert ga_d.shape == (n + n // 2,) and at_r.shape == at_d.shape == (n,)
    assert float(ga_lag) == pytest.approx(want_lag / 16000.0, abs=1e-7)
    assert float(at_lag) == pytest.approx(want_lag / 16000.0, abs=1e-7)
    assert int(ga_v) == n + want_lag and int(at_v) == n - abs(want_lag)
  assert traces == 1


def test_no_float64_stablehlo_fft_ops_when_caller_enables_x64():
  assert not jax.config.read("jax_enable_x64")
  with jax.enable_x64(True):
    s64 = jax.ShapeDtypeStruct((128,), jnp.float64)
    s96 = jax.ShapeDtypeStruct((96,), jnp.float64)
    i64 = jax.ShapeDtypeStruct((), jnp.int64)

    bad = jax.jit(lambda a: jnp.fft.irfft(jnp.fft.rfft(a, 128), 128)).lower(s64)
    assert 64 in _fft_float_widths(bad.as_text())
    with pytest.raises(AssertionError, match="wider than float32"):
      _assert_single_precision_transforms(bad.as_text())

    for lowered in (
        jax.jit(alignment.calc_upper_env).lower(s64),
        jax.jit(alignment.calc_upper_env).lower(s64, i64),
        jax.jit(alignment.find_lowest_lag_index).lower(s64, s64),
        jax.jit(alignment.find_lowest_lag_index).lower(s64, s96),
        jax.jit(lambda r, d: alignment.globally_align(r, d, 16000)).lower(
            s64, s64
        ),
        jax.jit(lambda r, d: alignment.globally_align(r, d, 16000)).lower(
            s64, s96
        ),
        jax.jit(
            lambda r, d, rl, dl: alignment.align_and_truncate(
                r, d, 48000, ref_len=rl, deg_len=dl
            )
        ).lower(s64, s64, i64, i64),
    ):
      _assert_single_precision_transforms(lowered.as_text())
      for out_aval in jax.tree.leaves(lowered.out_info):
        assert out_aval.dtype in (jnp.float32, jnp.int32)

    coerced = _check.asarray_f32(jnp.ones((4,), dtype=jnp.float64), "x")
    assert coerced.dtype == jnp.float32

    s32 = jax.ShapeDtypeStruct((128,), jnp.float32)
    i32 = jax.ShapeDtypeStruct((), jnp.int32)
    hlo_f32 = (
        jax.jit(
            lambda r, d, rl, dl: alignment.align_and_truncate(
                r, d, 48000.0, ref_len=rl, deg_len=dl
            )
        )
        .lower(s32, s32, i32, i32)
        .as_text()
    )
    assert {int(w) for w in _FLOAT_WIDTH.findall(hlo_f32)} == {32}
  assert not jax.config.read("jax_enable_x64")
