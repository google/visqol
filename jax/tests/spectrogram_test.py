"""Tests for spectrogram.py against C++ ViSQOL v3.3.3."""

import math
import re

import jax
import jax.numpy as jnp
import pytest

from visqol_jax import spectrogram

# Exact Slaney ERB speech center frequencies (21 bands, 50..8000 Hz).
_CPP_SPEECH_CF_21 = (
    49.999999999999886,
    98.766985466538301,
    156.06316116100868,
    223.38025525254665,
    302.47089440957382,
    395.39423411525149,
    504.56956957958243,
    632.83932302905896,
    783.54304727238753,
    960.60437225384567,
    1168.6331582815233,
    1413.045515529393,
    1700.2048145675187,
    2037.5873591842828,
    2433.977034856845,
    2899.6940006183008,
    3446.8633784030585,
    4089.7309353017931,
    4845.0339776360088,
    5732.43711322337,
    6775.0442270734711,
)

# Exact Slaney ERB audio center frequencies (32 bands, 50..24000 Hz at 48 kHz).
_CPP_AUDIO_CF_32 = (
    49.999999999999829,
    91.747824516819435,
    139.74627739612919,
    194.93122428789127,
    258.3786518769453,
    331.32564728791931,
    415.19451859905098,
    511.62052676257497,
    622.48376964394095,
    749.94583984941528,
    896.49197109014221,
    1064.9794948452839,
    1258.6935521239923,
    1481.4111465846556,
    1737.4747879082029,
    2031.8771613114195,
    2370.3584740719925,
    2759.5183771112434,
    3206.9446438624741,
    3721.36111538357,
    4312.7977963224348,
    4992.7864182373851,
    5774.5852833327772,
    6673.4377725747145,
    7706.8695585339374,
    8895.0303179612893,
    10261.086606749268,
    11831.673557488639,
    13637.414206742113,
    15713.516577789669,
    18100.460160663104,
    20844.785174340454,
)

# Exact input and C++ MakeFilters/ApplyFilter rows (gammatone_filterbank_test).
_CPP_K10_SAMPLES = (0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9)
_CPP_COEFF_ROWS_48K = {
    0: (
        2.0833333333333333e-05,
        -2.1077286485028735e-05,
        -2.0421554319409269e-05,
        -2.0805673328744874e-05,
        -2.0693167475693131e-05,
        0.0,
        1.0,
        -1.9919443586130243,
        0.99200307560760292,
        3.6649852649301331e-10,
    ),
    15: (
        2.0833333333333333e-05,
        -3.225387863050954e-05,
        -6.6602315936860576e-06,
        -2.1652642917267274e-05,
        -1.7261467306928321e-05,
        0.0,
        1.0,
        -1.8678772907613888,
        0.9369758437566148,
        8.9548227258856191e-14,
    ),
    31: (
        2.0833333333333333e-05,
        -8.1656659625040867e-07,
        2.8992226667300499e-05,
        1.1530639851488132e-05,
        1.6645020219561957e-05,
        0.0,
        1.0,
        1.3524316834104042,
        0.54508108464140681,
        1.9766464077954128e-17,
    ),
}
_CPP_FILTERED_ROWS_48K = {
    0: (
        1.0279992865745242e-10,
        6.1514333322497233e-10,
        2.1471778249027126e-09,
        5.7102061950857463e-09,
        1.2761303067565419e-08,
        2.478440122721556e-08,
        4.339189015924688e-08,
        7.0424558056729325e-08,
        1.0804985064224385e-07,
        1.5885842106637969e-07,
    ),
    15: (
        4.2073443025994812e-07,
        2.4132294359678726e-06,
        7.8031841772563822e-06,
        1.8521706234460474e-05,
        3.5306121411184129e-05,
        5.41889372048484e-05,
        6.5675998463176122e-05,
        5.5534557641715811e-05,
        6.9788972724685241e-06,
        -9.6195603095977587e-05,
    ),
    31: (
        0.0019060577667284412,
        -0.0013435102948111592,
        0.0024489405897708452,
        0.0012448629909184226,
        -0.0025213321304550793,
        -0.0011792454942162571,
        0.0089437712764189915,
        -0.0094072262233261435,
        0.0033438509367506616,
        0.015725848982196269,
    ),
}

# Exact vectors from tests/spectrogram_test.cc.
_CPP_DB_TOLERANCE = 0.0001
_CPP_MIN_ELEM = -53.2
_CPP_FLOOR = 0.1
_CPP_10_ELEM_MAT = (
    10.21, -4.63, 0.54, 87.98, 0.065, 0.0, _CPP_MIN_ELEM, 8.7, 0.0, -2.76
)
_CPP_10_ELEM_TO_DB = (
    10.0903, 6.6558, -2.6761, 19.4438, -11.8709,
    -156.5356, 17.2591, 9.3952, -156.5356, 4.4091,
)

_JIT_BUILD = jax.jit(
    spectrogram.build_gammatone_spectrogram,
    static_argnums=(1,),
    static_argnames=("use_speech_mode",),
)
_FFT_LINE = re.compile(r"^\s*%\S+ = stablehlo\.fft .*?:(?P<types>.*)$")
_FLOAT_WIDTH = re.compile(r"(?:<|x|\b)f(?P<width>16|32|64)\b")
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


def _assert_array_equal(
    actual: jax.typing.ArrayLike, expected: jax.typing.ArrayLike
) -> None:
  a = jnp.asarray(actual)
  b = jnp.asarray(expected, dtype=a.dtype)
  if b.ndim == 0:
    assert bool(jnp.all(a == b))
  else:
    assert a.shape == b.shape and bool(jnp.array_equal(a, b))


def _fft_float_widths(hlo_text: str) -> list[int]:
  widths: list[int] = []
  for line in hlo_text.splitlines():
    match = _FFT_LINE.match(line)
    if match is not None:
      types = match.group("types")
      found = _FLOAT_WIDTH.findall(types)
      assert found, f"FFT operand of unexpected type {types.strip()}"
      widths.extend(int(w) for w in found)
  return widths


def _ref_spectrogram_f64(
    signal: list[float],
    sample_rate: int,
    *,
    use_speech_mode: bool = False,
    num_bands: int | None = None,
) -> list[list[float]]:
  """Exact Python float64 4-biquad DF2T per-frame RMS reference."""
  ws, hop, _ = spectrogram.analysis_window(sample_rate)
  hann = [
      0.5 - 0.5 * math.cos(2.0 * math.pi * i / (ws - 1)) for i in range(ws)
  ]
  nb = (
      (21 if use_speech_mode else 32)
      if num_bands is None
      else num_bands
  )
  hi = 8000.0 if use_speech_mode else sample_rate / 2.0
  # pylint: disable-next=protected-access
  _, rows = spectrogram._design_erb_filters(sample_rate, nb, 50.0, hi)
  num_cols = 1 + (len(signal) - ws) // hop
  frames = [
      [signal[c * hop + m] * hann[m] for m in range(ws)]
      for c in range(num_cols)
  ]
  out: list[list[float]] = []
  for a0, a11, a12, a13, a14, a2, _, b1, b2, gain in rows:
    a10, a11_g, a12_g = a0 / gain, a11 / gain, a2 / gain
    band_rms: list[float] = []
    for frame in frames:
      s10 = s11 = s20 = s21 = s30 = s31 = s40 = s41 = 0.0
      sum_sq = 0.0
      for x in frame:
        y1 = a10 * x + s10
        s10, s11 = a11_g * x + s11 - b1 * y1, a12_g * x - b2 * y1
        y2 = a0 * y1 + s20
        s20, s21 = a12 * y1 + s21 - b1 * y2, a2 * y1 - b2 * y2
        y3 = a0 * y2 + s30
        s30, s31 = a13 * y2 + s31 - b1 * y3, a2 * y2 - b2 * y3
        y4 = a0 * y3 + s40
        s40, s41 = a14 * y3 + s41 - b1 * y4, a2 * y3 - b2 * y4
        sum_sq += y4 * y4
      band_rms.append(math.sqrt(sum_sq / ws))
    out.append(band_rms)
  return out


def test_calc_spl_scale_to_match_spl_and_silence():
  ref = 0.2 * jax.random.normal(jax.random.key(11), (500,), dtype=jnp.float32)
  deg = 0.03 * jax.random.normal(jax.random.key(12), (400,), dtype=jnp.float32)
  mean_sq = sum(float(v) ** 2 for v in ref.tolist()) / ref.shape[0]
  expected_spl = 20.0 * math.log10(math.sqrt(mean_sq) / 0.00002)

  spl_ref = spectrogram.calc_spl(ref)
  assert spl_ref.dtype == jnp.float32
  assert float(spl_ref) == pytest.approx(expected_spl, abs=2e-5)
  scaled = spectrogram.scale_to_match_spl(ref, deg)
  assert scaled.dtype == jnp.float32 and scaled.shape == deg.shape
  assert float(spectrogram.calc_spl(scaled)) == pytest.approx(
      expected_spl, abs=2e-5
  )

  # Padded valid lengths ignore non-zero junk beyond ref_len / deg_len.
  ref_pad = jnp.concatenate([ref, jnp.full((50,), 99.0, dtype=jnp.float32)])
  deg_pad = jnp.concatenate([deg, jnp.full((80,), -77.0, dtype=jnp.float32)])
  scaled_pad = spectrogram.scale_to_match_spl(
      ref_pad, deg_pad, ref_len=ref.shape[0], deg_len=deg.shape[0]
  )
  _assert_allclose(scaled_pad[: deg.shape[0]], scaled, atol=1e-6)
  _assert_array_equal(scaled_pad[deg.shape[0] :], 0.0)

  # Over-large and wide-integer valid_len clamp without int32 wraparound.
  for big_len in (9999, jnp.uint32(0xFFFFFFFF), 10**30):
    assert float(spectrogram.calc_spl(ref, valid_len=big_len)) == pytest.approx(
        expected_spl, abs=2e-5
    )
  with jax.enable_x64(True):
    assert float(
        spectrogram.calc_spl(ref, valid_len=jnp.int64(1 << 31))
    ) == pytest.approx(expected_spl, abs=2e-5)
    assert math.isinf(
        float(spectrogram.calc_spl(ref, valid_len=jnp.int64(-(1 << 32) + 5)))
    )

  # Bit-identical padded signals keep exact 1.0 scale under jax.jit.
  ref_zpad = jnp.concatenate(
      [ref, jnp.zeros((ref.shape[0] // 2,), dtype=jnp.float32)]
  )
  id_jit = jax.jit(spectrogram.scale_to_match_spl)(
      ref, ref_zpad, deg_len=jnp.int32(ref.shape[0])
  )
  _assert_array_equal(id_jit[: ref.shape[0]], ref)
  ref_zpad_b = jnp.concatenate(
      [ref, jnp.full((512,), 5.0, dtype=jnp.float32)]
  )
  id_both_jit = jax.jit(spectrogram.scale_to_match_spl)(
      ref_zpad,
      ref_zpad_b,
      ref_len=jnp.int32(ref.shape[0]),
      deg_len=jnp.int32(ref.shape[0]),
  )
  _assert_array_equal(id_both_jit[: ref.shape[0]], ref)
  _assert_array_equal(id_both_jit[ref.shape[0] :], 0.0)

  # Silence combinations remain finite zeros (matches C++ post-RaiseFloor(-45)).
  zeros = jnp.zeros((64,), dtype=jnp.float32)
  assert math.isinf(float(spectrogram.calc_spl(zeros)))
  assert math.isinf(float(spectrogram.calc_spl(ref[:64], valid_len=0)))
  for r_s, d_s in ((ref[:64], zeros), (zeros, zeros), (zeros, ref[:64])):
    out = spectrogram.scale_to_match_spl(r_s, d_s)
    assert out.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(out)))
    _assert_array_equal(out, 0.0)
  out0 = spectrogram.scale_to_match_spl(ref[:64], ref[:64], deg_len=0)
  _assert_array_equal(out0, 0.0)

  with pytest.raises(ValueError, match="non-empty 1D"):
    spectrogram.calc_spl(jnp.zeros((2, 3), dtype=jnp.float32))
  for bad_sig in (
      jnp.float32(0.0),
      jnp.zeros((0,), dtype=jnp.float32),
      jnp.zeros((2, 3), dtype=jnp.float32),
  ):
    with pytest.raises(ValueError, match="reference.*non-empty 1D"):
      spectrogram.scale_to_match_spl(bad_sig, ref)
    with pytest.raises(ValueError, match="degraded.*non-empty 1D"):
      spectrogram.scale_to_match_spl(ref, bad_sig)
  for bad_len in (True, _NP_BOOL_TRUE, 1.5, jnp.array([1, 2])):
    with pytest.raises(ValueError, match="integer scalar"):
      spectrogram.calc_spl(ref, valid_len=bad_len)
    with pytest.raises(ValueError, match="integer scalar"):
      spectrogram.scale_to_match_spl(ref, deg, deg_len=bad_len)


@pytest.mark.parametrize(
    "sample_rate", [8000, 11025, 16000, 22050, 44100, 48000, 96000]
)
def test_analysis_window_matches_cpp_unit_test(sample_rate: int):
  size, hop, hann = spectrogram.analysis_window(sample_rate)
  assert math.floor((1000.0 / sample_rate) * size + 0.5) == 80
  assert hop == int(size * 0.25)
  assert hann.dtype == jnp.float32 and hann.shape == (size,)
  assert float(hann[0]) == pytest.approx(0.0, abs=1e-7)
  assert float(hann[-1]) == pytest.approx(0.0, abs=1e-7)
  expected = [
      0.5 - 0.5 * math.cos(2.0 * math.pi * i / (size - 1)) for i in range(size)
  ]
  _assert_allclose(hann, expected, atol=1e-6)


def test_analysis_window_half_away_from_zero_and_frame_count():
  assert spectrogram.analysis_window(16000)[:2] == (1280, 320)
  assert spectrogram.analysis_window(48000)[:2] == (3840, 960)
  assert spectrogram.analysis_window(11025)[:2] == (882, 220)

  # C++ std::round rounds half away from zero, unlike Python round() to even.
  assert round(16025 * 0.02) == 320
  assert spectrogram.analysis_window(16025, window_duration=0.02)[:2] == (
      321,
      80,
  )
  assert round(625 * 0.004) == 2
  assert spectrogram.analysis_window(
      625, overlap=0.5, window_duration=0.004
  )[:2] == (3, 1)

  assert int(spectrogram.num_spectrogram_frames(3839, 3840, 960)) == 0
  assert int(spectrogram.num_spectrogram_frames(3840, 3840, 960)) == 1
  assert (
      int(spectrogram.num_spectrogram_frames(3840 + 960 * 5 + 959, 3840, 960))
      == 6
  )
  batched_frames = spectrogram.num_spectrogram_frames(
      jnp.array([0, 1279, 1280, 1600], dtype=jnp.int32), 1280, 320
  )
  assert batched_frames.dtype == jnp.int32
  _assert_array_equal(batched_frames, [0, 0, 1, 2])
  with jax.enable_x64(True):
    neg64 = spectrogram.num_spectrogram_frames(
        jnp.int64(-(1 << 32) + 5000), 1280, 320
    )
    assert neg64.dtype == jnp.int32 and int(neg64) == 0
  u32_big = spectrogram.num_spectrogram_frames(
      jnp.uint32(3_000_000_000), 3840, 960
  )
  assert u32_big.dtype == jnp.int32
  assert int(u32_big) == int(
      spectrogram.num_spectrogram_frames(3_000_000_000, 3840, 960)
  )
  u32_sat = spectrogram.num_spectrogram_frames(jnp.uint32(3_000_000_000), 1, 1)
  assert u32_sat.dtype == jnp.int32 and int(u32_sat) == 0x7FFFFFFF

  with pytest.raises(ValueError, match="sample_rate"):
    spectrogram.analysis_window(0)
  for bad in (
      0.0, -0.1, True, _NP_BOOL_TRUE, float("nan"), float("inf"), 0.25 + 0j
  ):
    with pytest.raises(ValueError, match="overlap"):
      spectrogram.analysis_window(16000, overlap=bad)
    with pytest.raises(ValueError, match="window_duration"):
      spectrogram.analysis_window(16000, window_duration=bad)
  with pytest.raises(ValueError, match="window_size overflow"):
    spectrogram.analysis_window(48000, window_duration=1e20)
  with pytest.raises(ValueError, match="invalid window_size=0"):
    spectrogram.analysis_window(
        1, overlap=1.0, window_duration=math.nextafter(0.5, 0.0)
    )
  with pytest.raises(ValueError, match="invalid window_size"):
    spectrogram.analysis_window(100, overlap=0.25, window_duration=0.01)
  for bad_n in (
      True,
      _NP_BOOL_TRUE,
      [True, 1280],
      [jnp.array(True), 1280],
      1280.0,
      "1280",
  ):
    with pytest.raises(ValueError, match="num_samples"):
      spectrogram.num_spectrogram_frames(bad_n, 1280, 320)


def test_make_erb_filters_matches_cpp_center_freqs_coeffs_and_caches():
  for sr in (16000, 48000):
    # pylint: disable-next=protected-access
    cfs_f64, _ = spectrogram._design_erb_filters(
        sr,
        spectrogram.SPEECH_MODE_NUM_BANDS,
        spectrogram.MIN_FREQ,
        spectrogram.SPEECH_MODE_MAX_FREQ,
    )
    assert cfs_f64 == pytest.approx(_CPP_SPEECH_CF_21, rel=1e-12, abs=1e-12)
    cf_sp, coeffs_sp = spectrogram.make_erb_filters(
        sr,
        spectrogram.SPEECH_MODE_NUM_BANDS,
        spectrogram.MIN_FREQ,
        spectrogram.SPEECH_MODE_MAX_FREQ,
    )
    assert cf_sp.dtype == coeffs_sp.dtype == jnp.float32
    assert bool(jnp.all(jnp.diff(cf_sp) > 0.0))
    assert bool(jnp.all(jnp.isfinite(coeffs_sp)))
    _assert_allclose(cf_sp, _CPP_SPEECH_CF_21, rtol=1e-6, atol=1e-5)

  # pylint: disable-next=protected-access
  cfs_au64, rows_au64 = spectrogram._design_erb_filters(
      48000, spectrogram.AUDIO_MODE_NUM_BANDS, spectrogram.MIN_FREQ, 24000.0
  )
  assert cfs_au64 == pytest.approx(_CPP_AUDIO_CF_32, rel=1e-12, abs=1e-12)
  for row_idx, want_row in _CPP_COEFF_ROWS_48K.items():
    assert rows_au64[row_idx] == pytest.approx(want_row, rel=1e-12, abs=1e-22)

  cf_au, coeffs_au = spectrogram.make_erb_filters(
      48000, spectrogram.AUDIO_MODE_NUM_BANDS, spectrogram.MIN_FREQ, 24000.0
  )
  assert cf_au.dtype == coeffs_au.dtype == jnp.float32
  _assert_allclose(cf_au, _CPP_AUDIO_CF_32, rtol=1e-6, atol=1e-5)
  for row_idx, want_row in _CPP_COEFF_ROWS_48K.items():
    assert coeffs_au[row_idx].tolist() == pytest.approx(
        want_row, rel=1e-6, abs=1e-22
    )

  # reverse=False matches C++ CalcUniformCenterFreqs before FlipUpDown.
  cf_desc, coeffs_desc = spectrogram.make_erb_filters(
      48000, 32, 50.0, 24000.0, reverse=False
  )
  _assert_array_equal(cf_desc, cf_au[::-1])
  _assert_array_equal(coeffs_desc, coeffs_au[::-1])

  # Clamping high_freq > sample_rate / 2 matches Nyquist explicitly.
  cf_clamp, _ = spectrogram.make_erb_filters(8000, 21, 50.0, 8000.0)
  cf_nyq, _ = spectrogram.make_erb_filters(8000, 21, 50.0, 4000.0)
  _assert_array_equal(cf_clamp, cf_nyq)

  # Caches store immutable Python tuples/bytes and never capture JAX tracers.
  sig32 = jnp.ones((2000,), dtype=jnp.float32) * jnp.float32(0.1)
  # pylint: disable=protected-access
  spectrogram._design_erb_filters.cache_clear()
  spectrogram._gammatone_impulse_f32_bytes.cache_clear()
  spec1, _ = jax.jit(
      lambda s: spectrogram.build_gammatone_spectrogram(
          s, 16000, use_speech_mode=True
      )
  )(sig32)
  cf_tup, rows_tup = spectrogram._design_erb_filters(16000, 21, 50.0, 8000.0)
  imp_bytes = spectrogram._gammatone_impulse_f32_bytes(
      16000, 21, 50.0, 8000.0, 1280
  )
  # pylint: enable=protected-access
  assert isinstance(cf_tup, tuple) and all(isinstance(v, float) for v in cf_tup)
  assert isinstance(rows_tup, tuple) and all(
      isinstance(r, tuple) and all(isinstance(v, float) for v in r)
      for r in rows_tup
  )
  assert isinstance(imp_bytes, bytes) and len(imp_bytes) == 21 * 1280 * 4
  spec2, _ = jax.jit(
      lambda s: spectrogram.build_gammatone_spectrogram(
          s, 16000, use_speech_mode=True
      )
  )(sig32)
  _assert_allclose(spec2, spec1, atol=1e-7)

  for bad_f in (-1.0, True, _NP_BOOL_TRUE, float("nan"), float("inf")):
    with pytest.raises(ValueError):
      spectrogram.make_erb_filters(16000, 21, low_freq=bad_f, high_freq=4000.0)
  for bad_hi in (True, _NP_BOOL_TRUE, float("nan"), float("inf"), 25.0):
    with pytest.raises(ValueError):
      spectrogram.make_erb_filters(16000, 21, low_freq=50.0, high_freq=bad_hi)
  with pytest.raises(ValueError, match="reverse"):
    spectrogram.make_erb_filters(16000, 21, reverse=2)  # type: ignore[arg-type]


def test_filterbank_and_spectrogram_builder_match_cpp_tests_and_short_errors():
  _, coeffs_48k = spectrogram.make_erb_filters(48000, 32, 50.0, 24000.0)
  filtered = spectrogram.apply_gammatone_filterbank(
      _CPP_K10_SAMPLES, coeffs_48k
  )
  assert filtered.shape == (32, 10) and filtered.dtype == jnp.float32
  assert bool(jnp.all(jnp.isfinite(filtered)))
  for row_idx, want_row in _CPP_FILTERED_ROWS_48K.items():
    assert filtered[row_idx].tolist() == pytest.approx(
        want_row, rel=4e-3, abs=1e-12
    )

  for bad_coeffs in (
      jnp.zeros((32,), dtype=jnp.float32),
      jnp.zeros((0, 10), dtype=jnp.float32),
      jnp.zeros((32, 9), dtype=jnp.float32),
  ):
    with pytest.raises(ValueError, match="filter_coeffs"):
      spectrogram.apply_gammatone_filterbank(_CPP_K10_SAMPLES, bad_coeffs)

  # Exact col counts from gammatone_spectrogram_builder_test.cc (802, 807).
  ws48, hop48, hann48 = spectrogram.analysis_window(48000)
  assert int(spectrogram.num_spectrogram_frames(773193, ws48, hop48)) == 802
  assert int(spectrogram.num_spectrogram_frames(778240, ws48, hop48)) == 807

  ref = 0.1 * jax.random.normal(
      jax.random.key(23), (ws48 + 5 * hop48 + 123,), dtype=jnp.float32
  )
  deg = 0.08 * jax.random.normal(
      jax.random.key(24), (ws48 + 8 * hop48,), dtype=jnp.float32
  )
  ref_spec, ref_cf = _JIT_BUILD(ref, 48000)
  deg_spec, deg_cf = _JIT_BUILD(deg, 48000)
  assert ref_spec.dtype == deg_spec.dtype == jnp.float32
  assert ref_spec.shape == (32, 6) and deg_spec.shape == (32, 9)
  _assert_array_equal(ref_cf, deg_cf)
  _assert_allclose(ref_cf, _CPP_AUDIO_CF_32, rtol=1e-6, atol=1e-5)

  # FFT spectrogram matches exact float64 4-biquad DF2T reference (~2e-5 rel).
  want_ref64 = _ref_spectrogram_f64(ref.tolist(), 48000)
  _assert_allclose(ref_spec, want_ref64, rtol=2e-5, atol=1e-7)

  # 1D float32 4-biquad filterbank agrees within ~4e-3 rel across frames.
  for col in range(3):
    frame = ref[col * hop48 : col * hop48 + ws48] * hann48
    filt = spectrogram.apply_gammatone_filterbank(frame, coeffs_48k)
    got_col = jnp.sqrt(jnp.mean(filt * filt, axis=1))
    _assert_allclose(got_col, ref_spec[:, col], rtol=4e-3, atol=1e-7)

  # Short-signal ValueErrors and exact window_size boundary acceptance.
  for n_bad, sr, sp in (
      (1, 48000, False),
      (3839, 48000, False),
      (1279, 16000, True),
  ):
    with pytest.raises(ValueError, match="Too few samples"):
      spectrogram.build_gammatone_spectrogram(
          jnp.zeros((n_bad,), dtype=jnp.float32), sr, use_speech_mode=sp
      )
  s48_min, _ = spectrogram.build_gammatone_spectrogram(
      jnp.ones((3840,), dtype=jnp.float32), 48000, use_speech_mode=False
  )
  s16_min, _ = spectrogram.build_gammatone_spectrogram(
      jnp.ones((1280,), dtype=jnp.float32), 16000, use_speech_mode=True
  )
  assert s48_min.shape == (32, 1) and s16_min.shape == (21, 1)
  want_s16 = _ref_spectrogram_f64([1.0] * 1280, 16000, use_speech_mode=True)
  _assert_allclose(s16_min, want_s16, rtol=2e-5, atol=1e-7)
  huge_hop_spec, _ = spectrogram.build_gammatone_spectrogram(
      jnp.ones((2000,), dtype=jnp.float32),
      16000,
      use_speech_mode=True,
      overlap=2e6,
  )
  _assert_allclose(huge_hop_spec, s16_min, rtol=1e-6, atol=1e-7)

  # Prime num_bands=7 matches float64 DF2T reference.
  short_ref = ref[: ws48 + hop48]
  prime32, _ = spectrogram.build_gammatone_spectrogram(
      short_ref, 48000, num_bands=7
  )
  assert prime32.dtype == jnp.float32 and prime32.shape == (7, 2)
  want_prime = _ref_spectrogram_f64(short_ref.tolist(), 48000, num_bands=7)
  _assert_allclose(prime32, want_prime, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize(("sr", "use_speech"), [(16000, True), (48000, False)])
def test_build_gammatone_spectrogram_vmap_and_fft_widths(
    sr: int, use_speech: bool
):
  ws, hop, _ = spectrogram.analysis_window(sr)
  batch = 0.1 * jax.random.normal(
      jax.random.key(sr), (3, ws + 4 * hop), dtype=jnp.float32
  )
  batched = jax.vmap(
      lambda s: spectrogram.build_gammatone_spectrogram(
          s, sr, use_speech_mode=use_speech
      )
  )
  spec32, cf32 = jax.jit(batched)(batch)
  assert spec32.dtype == cf32.dtype == jnp.float32
  assert bool(jnp.all(jnp.isfinite(spec32)))
  want_batch = [
      _ref_spectrogram_f64(row, sr, use_speech_mode=use_speech)
      for row in batch.tolist()
  ]
  _assert_allclose(spec32, want_batch, rtol=2e-5, atol=1e-7)

  # Paired odd-frame float32 build isolates ref and deg across complex pairs.
  sig32 = batch[0]
  assert (1 + (sig32.shape[0] - ws) // hop) % 2 == 1
  # pylint: disable=protected-access
  (s_ref, s_deg), _ = spectrogram._build_gammatone_spectrograms(
      (sig32, sig32), sr, use_speech_mode=use_speech
  )
  (s_ref_scaled, _), _ = spectrogram._build_gammatone_spectrograms(
      (sig32, sig32 * jnp.float32(1e6)), sr, use_speech_mode=use_speech
  )
  # pylint: enable=protected-access
  _assert_allclose(s_ref, s_deg, rtol=1e-6, atol=1e-8)
  _assert_allclose(s_ref, s_ref_scaled, rtol=1e-6, atol=1e-8)

  # Lowered StableHLO uses only 32-bit FFTs and floats even with x64 enabled.
  fn = jax.jit(
      lambda s: spectrogram.build_gammatone_spectrogram(
          s, sr, use_speech_mode=use_speech
      )
  )
  hlo_default = fn.lower(
      jax.ShapeDtypeStruct((ws + 2 * hop,), jnp.float32)
  ).as_text()
  widths_default = _fft_float_widths(hlo_default)
  assert widths_default and max(widths_default) <= 32

  with jax.enable_x64(True):
    low64 = fn.lower(jax.ShapeDtypeStruct((ws + 2 * hop,), jnp.float64))
    widths64 = _fft_float_widths(low64.as_text())
    assert widths64 and max(widths64) <= 32
    for out_aval in jax.tree.leaves(low64.out_info):
      assert out_aval.dtype == jnp.float32

    hlo32_x64 = fn.lower(
        jax.ShapeDtypeStruct((ws + 2 * hop,), jnp.float32)
    ).as_text()
    assert {int(w) for w in _FLOAT_WIDTH.findall(hlo32_x64)} == {32}


def test_convert_to_db_and_prepare_spectrograms_cpp_parity_and_masking():
  got_db = spectrogram.convert_to_db(_CPP_10_ELEM_MAT)
  assert got_db.dtype == jnp.float32
  _assert_allclose(got_db, _CPP_10_ELEM_TO_DB, atol=_CPP_DB_TOLERANCE)
  mat = jnp.asarray(_CPP_10_ELEM_MAT, dtype=jnp.float32)
  assert float(jnp.min(mat)) == pytest.approx(
      _CPP_MIN_ELEM, abs=_CPP_DB_TOLERANCE
  )
  _assert_allclose(
      [v - _CPP_FLOOR for v in _CPP_10_ELEM_MAT],
      mat - jnp.float32(_CPP_FLOOR),
      atol=_CPP_DB_TOLERANCE,
  )
  # Power-dB quirk: 10*log10(abs(10.0)) == 10.0 (not 20.0), 0 -> 10*log10(eps).
  assert float(spectrogram.convert_to_db([10.0])[0]) == pytest.approx(
      10.0, abs=1e-6
  )
  eps_db = 10.0 * math.log10(math.ulp(1.0))
  assert float(spectrogram.convert_to_db([0.0, -0.0])[1]) == pytest.approx(
      eps_db, abs=2e-5
  )
  assert float(spectrogram.convert_to_db([1e-20])[0]) == pytest.approx(
      -200.0, abs=2e-5
  )

  # RaiseFloorPerFrame stops at min_cols=2, leaving ref[1, 2] at -45 dB floor.
  ref_in = [[100.0, 0.0, 100.0], [0.0, 0.1, 0.0]]
  deg_in = [[0.0, 10.0], [0.0, 0.0]]
  want_ref = [[65.0, 10.0, 65.0], [20.0, 35.0, 0.0]]
  want_deg = [[20.0, 55.0], [20.0, 10.0]]
  pr, pd = spectrogram.prepare_spectrograms(ref_in, deg_in)
  assert pr.dtype == pd.dtype == jnp.float32
  _assert_allclose(pr, want_ref, atol=1e-5)
  _assert_allclose(pd, want_deg, atol=1e-5)

  ref = jax.random.uniform(
      jax.random.key(42), (8, 12), minval=0.0, maxval=2.5, dtype=jnp.float32
  ).at[0, 3].set(0.0)
  deg = jax.random.uniform(
      jax.random.key(43), (8, 9), minval=0.0, maxval=2.5, dtype=jnp.float32
  ).at[2, 5].set(0.0)
  got_r, got_d = spectrogram.prepare_spectrograms(ref, deg)
  assert float(
      jnp.min(jnp.minimum(jnp.min(got_r), jnp.min(got_d)))
  ) == pytest.approx(0.0, abs=1e-6)

  # Padded trailing columns (zero and loud 500.0) must not alter valid slices.
  ref_pad = jnp.concatenate(
      [
          ref,
          jnp.zeros((8, 2), dtype=jnp.float32),
          jnp.full((8, 2), 500.0, dtype=jnp.float32),
      ],
      axis=1,
  )
  deg_pad = jnp.concatenate(
      [
          deg,
          jnp.zeros((8, 3), dtype=jnp.float32),
          jnp.full((8, 3), 500.0, dtype=jnp.float32),
      ],
      axis=1,
  )
  got_rp, got_dp = spectrogram.prepare_spectrograms(
      ref_pad, deg_pad, ref_valid_cols=12, deg_valid_cols=9
  )
  _assert_allclose(got_rp[:, :12], got_r, atol=1e-6)
  _assert_allclose(got_dp[:, :9], got_d, atol=1e-6)
  _assert_array_equal(got_rp[:, 12:], 0.0)
  _assert_array_equal(got_dp[:, 9:], 0.0)

  # deg_valid_cols == 0 with non-empty ref normalizes ref and zeroes deg.
  r_only, d_zero = spectrogram.prepare_spectrograms(
      ref_pad, deg_pad, ref_valid_cols=12, deg_valid_cols=0
  )
  assert bool(jnp.all(jnp.isfinite(r_only)))
  assert float(jnp.min(r_only[:, :12])) == pytest.approx(0.0, abs=1e-6)
  _assert_array_equal(d_zero, 0.0)

  # T_ref < T_deg (post-GloballyAlign positive lag) and ref_valid_cols == 0.
  sw_r, sw_d = spectrogram.prepare_spectrograms(deg, ref)
  sw_rp, sw_dp = spectrogram.prepare_spectrograms(
      deg_pad, ref_pad, ref_valid_cols=9, deg_valid_cols=12
  )
  _assert_allclose(sw_rp[:, :9], sw_r, atol=1e-6)
  _assert_allclose(sw_dp[:, :12], sw_d, atol=1e-6)
  _assert_array_equal(sw_rp[:, 9:], 0.0)
  _assert_array_equal(sw_dp[:, 12:], 0.0)
  r_zero, d_only = spectrogram.prepare_spectrograms(
      ref_pad, deg_pad, ref_valid_cols=0, deg_valid_cols=9
  )
  _assert_array_equal(r_zero, 0.0)
  assert float(jnp.min(d_only[:, :9])) == pytest.approx(0.0, abs=1e-6)

  zr, zd = spectrogram.prepare_spectrograms(
      ref_pad, deg_pad, ref_valid_cols=0, deg_valid_cols=0
  )
  _assert_array_equal(zr, 0.0)
  _assert_array_equal(zd, 0.0)

  with pytest.raises(ValueError, match="band count mismatch"):
    spectrogram.prepare_spectrograms(jnp.ones((4, 3)), jnp.ones((5, 3)))
  for bad_cols in (True, _NP_BOOL_TRUE, 1.5, jnp.array([1, 2])):
    with pytest.raises(ValueError, match="integer scalar"):
      spectrogram.prepare_spectrograms(ref, deg, deg_valid_cols=bad_cols)


def test_static_shape_jit_compiles_once_and_coerces_float64_under_x64():
  traces = 0
  ws, hop, _ = spectrogram.analysis_window(16000)

  @jax.jit
  def run_prep(ref: jax.Array, deg: jax.Array, deg_vlen: jax.Array):
    nonlocal traces
    traces += 1
    deg_s = spectrogram.scale_to_match_spl(ref, deg, deg_len=deg_vlen)
    # pylint: disable-next=protected-access
    (ref_lin, deg_lin), _ = spectrogram._build_gammatone_spectrograms(
        (ref, deg_s), 16000, use_speech_mode=True
    )
    deg_vcols = spectrogram.num_spectrogram_frames(deg_vlen, ws, hop)
    return spectrogram.prepare_spectrograms(
        ref_lin, deg_lin, deg_valid_cols=deg_vcols
    )

  n = ws + 5 * hop
  ref = 0.1 * jax.random.normal(jax.random.key(9), (n,), dtype=jnp.float32)
  deg = 0.05 * jax.random.normal(jax.random.key(10), (n,), dtype=jnp.float32)
  for vlen in (0, ws - 1, ws, ws + 2 * hop, n):
    r_out, d_out = run_prep(ref, deg, jnp.int32(vlen))
    assert r_out.dtype == d_out.dtype == jnp.float32
    assert r_out.shape == d_out.shape == (21, 6)
    assert bool(jnp.all(jnp.isfinite(r_out)))
    assert bool(jnp.all(jnp.isfinite(d_out)))
  assert traces == 1

  # Differentiability across silent leading frames and odd frame counts.
  deg_with_silence = jnp.concatenate([
      jnp.zeros((ws + hop,), dtype=jnp.float32),
      deg[ws + hop : ws + 4 * hop],
  ])
  ref_odd = ref[: ws + 4 * hop]
  # pylint: disable-next=protected-access
  build_specs = spectrogram._build_gammatone_spectrograms
  grad_fn = jax.jit(
      jax.grad(
          lambda d: jnp.sum(
              spectrogram.prepare_spectrograms(
                  *build_specs(
                      (ref_odd, spectrogram.scale_to_match_spl(ref_odd, d)),
                      16000,
                      use_speech_mode=True,
                  )[0]
              )[1]
          )
      )
  )
  grad_deg = grad_fn(deg_with_silence)
  assert grad_deg.dtype == jnp.float32
  assert bool(jnp.all(jnp.isfinite(grad_deg)))
  assert float(jnp.max(jnp.abs(grad_deg[ws + hop :]))) > 0.0

  with jax.enable_x64(True):
    ref64 = ref.astype(jnp.float64)
    deg64 = deg.astype(jnp.float64)
    assert spectrogram.scale_to_match_spl(ref64, deg64).dtype == jnp.float32
    sp, cf = spectrogram.build_gammatone_spectrogram(
        ref64, 16000, use_speech_mode=True
    )
    assert sp.dtype == cf.dtype == jnp.float32
    pr, pd = spectrogram.prepare_spectrograms(
        sp.astype(jnp.float64), sp.astype(jnp.float64)
    )
    assert pr.dtype == pd.dtype == jnp.float32
    prep_hlo = run_prep.lower(
        jax.ShapeDtypeStruct((n,), jnp.float32),
        jax.ShapeDtypeStruct((n,), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
    ).as_text()
    assert {int(w) for w in _FLOAT_WIDTH.findall(prep_hlo)} == {32}
    fb_hlo = (
        jax.jit(spectrogram.apply_gammatone_filterbank)
        .lower(
            jax.ShapeDtypeStruct((64,), jnp.float32),
            jax.ShapeDtypeStruct((21, 10), jnp.float32),
        )
        .as_text()
    )
    assert {int(w) for w in _FLOAT_WIDTH.findall(fb_hlo)} == {32}
