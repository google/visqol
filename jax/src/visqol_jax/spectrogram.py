"""SPL scaling, Slaney ERB gammatone filterbank, and spectrogram builder.

Ports ViSQOL v3.3.3 `misc_audio.cc`, `analysis_window.cc`,
`equivalent_rectangular_bandwidth.cc`, `gammatone_filterbank.cc`,
`signal_filter.cc`, `gammatone_spectrogram_builder.cc`, and `spectrogram.cc`.
"""

import array
import cmath
import functools
import math
from typing import SupportsFloat

import jax
import jax.numpy as jnp

from visqol_jax import _check
from visqol_jax import alignment

SPL_REFERENCE_POINT = 0.00002
NOISE_FLOOR_ABSOLUTE_DB = -45.0
NOISE_FLOOR_RELATIVE_TO_PEAK_DB = 45.0
SPEECH_MODE_NUM_BANDS = 21
AUDIO_MODE_NUM_BANDS = 32
MIN_FREQ = 50.0
SPEECH_MODE_MAX_FREQ = 8000.0
DEFAULT_WINDOW_DURATION = 0.08
DEFAULT_OVERLAP = 0.25

_EAR_Q = 9.26449
_MIN_BW = 24.7
_DOUBLE_EPS = math.ulp(1.0)
_INT32_MAX = 0x7FFFFFFF
# pylint: disable-next=protected-access
_valid_len = alignment._valid_len


def _as_finite_float(value: SupportsFloat, name: str) -> float:
  """Validate concrete finite real scalar `value` and return Python `float`."""
  msg = f"{name} must be a finite scalar, got {value!r}"
  if _check.is_bool(value) or isinstance(
      value, (complex, str, bytes, bytearray, jax.core.Tracer)
  ):
    raise ValueError(msg)
  if not isinstance(value, (int, float)):
    dtype = getattr(value, "dtype", None)
    if dtype is None or getattr(value, "ndim", 0) != 0:
      raise ValueError(msg)
    try:
      dt = jnp.dtype(dtype)
    except TypeError as exc:
      raise ValueError(msg) from exc
    if (
        dt == jnp.bool_
        or jnp.issubdtype(dt, jnp.complexfloating)
        or not jnp.issubdtype(dt, jnp.number)
    ):
      raise ValueError(msg)
  try:
    fv = float(value)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError(msg) from exc
  if not math.isfinite(fv):
    raise ValueError(msg)
  return 0.0 if fv == 0.0 else fv


def _as_bool(value: bool, name: str) -> bool:
  if not isinstance(value, bool) and not _check.is_bool(value):
    raise ValueError(f"{name} must be a bool, got {value!r}")
  return bool(value)


def calc_spl(
    signal: jax.typing.ArrayLike,
    valid_len: jax.typing.ArrayLike | None = None,
) -> jax.Array:
  """Compute sound pressure level in dB (`CalcSoundPressureLevel`)."""
  x = _check.asarray_f32(signal, "signal")
  _check.check_1d_nonempty(x, "signal")
  if valid_len is None:
    mean_sq = jnp.mean(x * x)
  else:
    n_valid = _valid_len(valid_len, x.shape[0], "valid_len")
    mask = jnp.arange(x.shape[0], dtype=jnp.int32) < n_valid
    denom = jnp.maximum(n_valid, jnp.int32(1)).astype(jnp.float32)
    mean_sq = jnp.sum(jnp.where(mask, x * x, jnp.float32(0.0))) / denom
  return jnp.float32(20.0) * jnp.log10(
      jnp.sqrt(mean_sq) / jnp.float32(SPL_REFERENCE_POINT)
  )


def scale_to_match_spl(
    reference: jax.typing.ArrayLike,
    degraded: jax.typing.ArrayLike,
    *,
    ref_len: jax.typing.ArrayLike | None = None,
    deg_len: jax.typing.ArrayLike | None = None,
) -> jax.Array:
  """Scale `degraded` to `reference` SPL (`ScaleToMatchSoundPressureLevel`)."""
  ref = _check.asarray_f32(reference, "reference")
  deg = _check.asarray_f32(degraded, "degraded")
  _check.check_1d_nonempty(ref, "reference")
  _check.check_1d_nonempty(deg, "degraded")
  n_ref = (
      jnp.int32(ref.shape[0])
      if ref_len is None
      else _valid_len(ref_len, ref.shape[0], "ref_len")
  )
  n_deg = (
      jnp.int32(deg.shape[0])
      if deg_len is None
      else _valid_len(deg_len, deg.shape[0], "deg_len")
  )
  ref_spl = calc_spl(ref, valid_len=None if ref_len is None else n_ref)
  deg_spl = calc_spl(deg, valid_len=None if deg_len is None else n_deg)
  zero, one = jnp.float32(0.0), jnp.float32(1.0)
  deg_finite = jnp.isfinite(deg_spl)
  raw_scale = jnp.power(
      jnp.float32(10.0),
      (ref_spl - jnp.where(deg_finite, deg_spl, zero)) / jnp.float32(20.0),
  )
  # Preserve exact 1.0 scale when ref and deg are bit-identical despite padding.
  m = min(ref.shape[0], deg.shape[0])
  exact_id = (
      (n_ref == n_deg)
      & (n_ref <= jnp.int32(m))
      & jnp.all(
          jnp.where(
              jnp.arange(m, dtype=jnp.int32) < n_ref, ref[:m] == deg[:m], True
          )
      )
  )
  scaled = deg * jnp.where(deg_finite & (~exact_id), raw_scale, one)
  if deg_len is not None:
    scaled = jnp.where(
        jnp.arange(deg.shape[0], dtype=jnp.int32) < n_deg, scaled, zero
    )
  return scaled


def analysis_window(
    sample_rate: int,
    overlap: float = DEFAULT_OVERLAP,
    window_duration: float = DEFAULT_WINDOW_DURATION,
) -> tuple[int, int, jax.Array]:
  """Return `(window_size, hop_size, hann)` matching C++ `AnalysisWindow`."""
  fs = _check.check_positive_int(sample_rate, "sample_rate")
  ov = _as_finite_float(overlap, "overlap")
  win_dur = _as_finite_float(window_duration, "window_duration")
  if win_dur <= 0.0 or ov <= 0.0:
    raise ValueError(
        "window_duration and overlap must be positive, got"
        f" {window_duration}, {overlap}"
    )
  prod = fs * win_dur
  if not math.isfinite(prod):
    raise ValueError(
        f"window_size overflow for sample_rate={fs}, window_duration={win_dur}"
    )
  # Match C++ std::round (half away from zero) without x + 0.5 double-rounding.
  frac, integ = math.modf(prod)
  window_size = int(integ) + int(frac >= 0.5)
  if window_size > _INT32_MAX:
    raise ValueError(
        f"window_size overflow for sample_rate={fs}, window_duration={win_dur}"
    )
  hop_prod = window_size * ov
  if not math.isfinite(hop_prod):
    raise ValueError(f"hop_size overflow for window_size={window_size}")
  hop_size = int(hop_prod)
  if window_size < 2 or hop_size < 1:
    raise ValueError(
        f"invalid window_size={window_size} or hop_size={hop_size} for fs={fs}"
    )
  idx = jnp.arange(window_size, dtype=jnp.float32)
  half = jnp.float32(0.5)
  hann = half - half * jnp.cos(
      jnp.float32(2.0 * math.pi) * idx / jnp.float32(window_size - 1)
  )
  return window_size, hop_size, hann


def num_spectrogram_frames(
    num_samples: jax.typing.ArrayLike,
    window_size: int,
    hop_size: int,
) -> jax.Array:
  """Return valid spectrogram frame count for `num_samples`."""
  ws = _check.check_positive_int(window_size, "window_size")
  hs = _check.check_positive_int(hop_size, "hop_size")
  if _check.is_bool(num_samples) or (
      isinstance(num_samples, (list, tuple))
      and any(
          _check.is_bool(v) or getattr(v, "dtype", None) == jnp.bool_
          for v in num_samples
      )
  ):
    raise ValueError(
        f"num_samples must have an integer type, got {num_samples!r}"
    )
  if isinstance(num_samples, int):
    frames = 0 if num_samples < ws else 1 + (num_samples - ws) // hs
    if frames > _INT32_MAX:
      raise ValueError(f"frame count overflows int32 for {num_samples}")
    return jnp.int32(frames)
  try:
    raw = jnp.asarray(num_samples)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError(
        f"num_samples must have an integer type, got {num_samples!r}"
    ) from exc
  if not jnp.issubdtype(raw.dtype, jnp.integer):
    raise ValueError(f"num_samples must have an integer dtype, got {raw.dtype}")
  work_dt = jnp.int32 if raw.dtype.itemsize < 4 else raw.dtype
  dt_max = int(jnp.iinfo(work_dt).max)
  if ws > dt_max:
    return jnp.zeros(raw.shape, dtype=jnp.int32)
  n = raw.astype(work_dt) if raw.dtype != work_dt else raw
  ws_dt = jnp.asarray(ws, dtype=work_dt)
  ge = n >= ws_dt
  if hs > dt_max:
    return jnp.where(ge, jnp.int32(1), jnp.int32(0))
  diff = jnp.maximum(n, ws_dt) - ws_dt
  q = diff // jnp.asarray(hs, dtype=work_dt)
  if work_dt != jnp.int32:
    q = jnp.minimum(q, jnp.asarray(_INT32_MAX - 1, dtype=work_dt))
  return jnp.where(ge, jnp.int32(1) + q.astype(jnp.int32), jnp.int32(0))


def _validate_erb_args(
    sample_rate: int,
    num_bands: int,
    low_freq: float,
    high_freq: float | None,
) -> tuple[int, int, float, float]:
  """Validate ERB filterbank parameters and clamp `high_freq` to Nyquist."""
  fs = _check.check_positive_int(sample_rate, "sample_rate")
  nb = _check.check_positive_int(num_bands, "num_bands")
  low_f = _as_finite_float(low_freq, "low_freq")
  nyquist = fs / 2.0
  raw_high = (
      nyquist
      if high_freq is None
      else _as_finite_float(high_freq, "high_freq")
  )
  eff_high = min(raw_high, nyquist)
  if low_f < 0.0 or eff_high < low_f:
    raise ValueError(
        f"invalid frequency range [{low_f}, {eff_high}] for sample_rate={fs}"
    )
  return fs, nb, low_f, eff_high


# Pure-stdlib float64 cache avoids jit tracer leaks and x5**4 cancellation.
@functools.cache
def _design_erb_filters(
    sample_rate: int,
    num_bands: int,
    low_freq: float,
    high_freq: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
  """Slaney ERB filter design in float64 ordered lowest-to-highest frequency."""
  clamped_high = min(float(high_freq), sample_rate / 2.0)
  a = -(_EAR_Q * _MIN_BW)
  b = -math.log(clamped_high + _EAR_Q * _MIN_BW)
  c = math.log(low_freq + _EAR_Q * _MIN_BW)
  d = clamped_high + _EAR_Q * _MIN_BW
  e = (b + c) / num_bands

  t = 1.0 / sample_rate
  p1 = 2.0**1.5
  s1, s2 = math.sqrt(3.0 - p1), math.sqrt(3.0 + p1)

  cfs: list[float] = []
  rows: list[tuple[float, ...]] = []
  for i in range(num_bands, 0, -1):
    cf = a + math.exp(i * e) * d
    bw = 1.019 * 2.0 * math.pi * ((cf / _EAR_Q) + _MIN_BW)
    bt = bw * t
    exp_bt = math.exp(bt)
    w = 2.0 * cf * math.pi * t
    cos_w, sin_w = math.cos(w), math.sin(w)
    b1 = -2.0 * cos_w / exp_bt
    b2 = math.exp(-2.0 * bt)

    sin_t = sin_w * t
    b_pos, b_neg = sin_t * 2.0 * s2, sin_t * 2.0 * s1
    a_cos = cos_w * 2.0 * t
    a11 = -(a_cos / exp_bt + b_pos / exp_bt) / 2.0
    a12 = -(a_cos / exp_bt - b_pos / exp_bt) / 2.0
    a13 = -(a_cos / exp_bt + b_neg / exp_bt) / 2.0
    a14 = -(a_cos / exp_bt - b_neg / exp_bt) / 2.0

    x_exp = cmath.exp(2.0j * w)
    x01 = -2.0 * x_exp * t
    x02 = 2.0 * cmath.exp(-bt + 1.0j * w) * t
    x1 = x01 + x02 * (cos_w - s1 * sin_w)
    x2 = x01 + x02 * (cos_w + s1 * sin_w)
    x3 = x01 + x02 * (cos_w - s2 * sin_w)
    x4 = x01 + x02 * (cos_w + s2 * sin_w)
    x5 = (
        (-2.0 / math.exp(2.0 * bt))
        - 2.0 * x_exp
        + (2.0 * (1.0 + x_exp)) / exp_bt
    )
    gain = abs((x1 * x2 * x3 * x4) / (x5**4))
    cfs.append(cf)
    rows.append((t, a11, a12, a13, a14, 0.0, 1.0, b1, b2, gain))
  return tuple(cfs), tuple(rows)


def _is_5smooth(n: int) -> bool:
  for p in (2, 3, 5):
    while n % p == 0:
      n //= p
  return n == 1


def _next_fft_size(n: int) -> int:
  """Smallest 5-smooth multiple of 128 >= n (<=16384), else next pow2."""
  if n < 128 or n > 16384:
    return alignment.next_pow_two(n)
  k = (n + 127) // 128
  while not _is_5smooth(k):
    k += 1
  return 128 * k


def _band_chunk_size(num_bands: int, fft_n: int) -> int:
  """Largest divisor of `num_bands` keeping FFT scratch memory bounded."""
  cap = 8 if fft_n <= 4096 else 4
  return max(d for d in range(1, min(num_bands, cap) + 1) if num_bands % d == 0)


@functools.cache
def _gammatone_impulse_f32_bytes(
    sample_rate: int,
    num_bands: int,
    low_freq: float,
    high_freq: float,
    window_size: int,
) -> bytes:
  """Exact float64 4-biquad gammatone impulse responses packed as float32."""
  _, rows = _design_erb_filters(sample_rate, num_bands, low_freq, high_freq)
  imp_f64 = array.array("d")
  for a0, a11, a12, a13, a14, a2, _, b1, b2, gain in rows:
    a10, a11_g, a12_g = a0 / gain, a11 / gain, a2 / gain
    s10 = s11 = s20 = s21 = s30 = s31 = s40 = s41 = 0.0
    band = [0.0] * window_size
    for m in range(window_size):
      x = 1.0 if m == 0 else 0.0
      y1 = a10 * x + s10
      s10, s11 = a11_g * x + s11 - b1 * y1, a12_g * x - b2 * y1
      y2 = a0 * y1 + s20
      s20, s21 = a12 * y1 + s21 - b1 * y2, a2 * y1 - b2 * y2
      y3 = a0 * y2 + s30
      s30, s31 = a13 * y2 + s31 - b1 * y3, a2 * y2 - b2 * y3
      y4 = a0 * y3 + s40
      s40, s41 = a14 * y3 + s41 - b1 * y4, a2 * y3 - b2 * y4
      band[m] = y4
    imp_f64.extend(band)
  return array.array("f", imp_f64).tobytes()


def make_erb_filters(
    sample_rate: int,
    num_bands: int,
    low_freq: float = MIN_FREQ,
    high_freq: float | None = None,
    *,
    reverse: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Compute Slaney ERB center frequencies and 4th-order gammatone coefficients.

  Args:
    sample_rate: Positive sample rate in Hz.
    num_bands: Positive ERB channel count.
    low_freq: Lowest band edge in Hz.
    high_freq: Highest band edge in Hz (defaults to Nyquist).
    reverse: Order bands lowest-to-highest when `True`, else highest-to-lowest.

  Returns:
    Tuple `(center_freqs, filter_coeffs)` of `float32` arrays with shapes
    `(num_bands,)` and `(num_bands, 10)`.
  """
  rev = _as_bool(reverse, "reverse")
  fs, nb, low_f, eff_high = _validate_erb_args(
      sample_rate, num_bands, low_freq, high_freq
  )
  cfs, rows = _design_erb_filters(fs, nb, low_f, eff_high)
  if not rev:
    cfs = cfs[::-1]
    rows = rows[::-1]
  return (
      jnp.asarray(cfs, dtype=jnp.float32),
      jnp.asarray(rows, dtype=jnp.float32),
  )


def _unpack_biquad4_coeffs(
    filter_coeffs: jax.Array,
) -> tuple[jax.Array, ...]:
  a0, a11, a12, a13, a14, a2, _, b1, b2, gain = filter_coeffs.T
  return a0, a0 / gain, a11 / gain, a2 / gain, a12, a13, a14, a2, b1, b2


def _biquad4_step(
    coeffs: tuple[jax.Array, ...],
    states: tuple[jax.Array, ...],
    x_t: jax.Array,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
  """One time step of the 4 cascaded DF2T gammatone biquads."""
  a0, a10, a11_g, a12_g, a12, a13, a14, a2, b1, b2 = coeffs
  s10, s11, s20, s21, s30, s31, s40, s41 = states
  y1 = a10 * x_t + s10
  s10, s11 = a11_g * x_t + s11 - b1 * y1, a12_g * x_t - b2 * y1
  y2 = a0 * y1 + s20
  s20, s21 = a12 * y1 + s21 - b1 * y2, a2 * y1 - b2 * y2
  y3 = a0 * y2 + s30
  s30, s31 = a13 * y2 + s31 - b1 * y3, a2 * y2 - b2 * y3
  y4 = a0 * y3 + s40
  s40, s41 = a14 * y3 + s41 - b1 * y4, a2 * y3 - b2 * y4
  return (s10, s11, s20, s21, s30, s31, s40, s41), y4


def apply_gammatone_filterbank(
    signal: jax.typing.ArrayLike,
    filter_coeffs: jax.typing.ArrayLike,
) -> jax.Array:
  """Filter 1D `signal` through 4 cascaded DF2T biquads (`ApplyFilter`)."""
  x = _check.asarray_f32(signal, "signal")
  coeffs = _check.asarray_f32(filter_coeffs, "filter_coeffs")
  _check.check_1d_nonempty(x, "signal")
  if coeffs.ndim != 2 or coeffs.shape[0] == 0 or coeffs.shape[1] != 10:
    raise ValueError(
        f"filter_coeffs must have shape (num_bands, 10), got {coeffs.shape}"
    )
  unpacked = _unpack_biquad4_coeffs(coeffs)
  z = jnp.zeros((coeffs.shape[0],), dtype=jnp.float32)
  _, ys = jax.lax.scan(
      lambda states, x_t: _biquad4_step(unpacked, states, x_t),
      (z,) * 8,
      x,
  )
  return ys.T


def _safe_sqrt(x: jax.Array) -> jax.Array:
  pos = x > jnp.float32(0.0)
  return jnp.where(
      pos, jnp.sqrt(jnp.where(pos, x, jnp.float32(1.0))), jnp.float32(0.0)
  )


def _filter_frames_rms_fft(
    frames: jax.Array, fft_n: int, h_spec: jax.Array
) -> jax.Array:
  """Causal float32 gammatone RMS via 2-frame complex-packed FFT convolution."""
  num_frames, window_size = frames.shape
  pad = num_frames % 2
  fp = (
      jnp.pad(frames, ((0, pad), (0, 0)), constant_values=jnp.float32(0.0))
      if pad
      else frames
  )
  # Pack adjacent frames into complex real/imag to halve FFT count on TPU/CPU.
  z_spec = jnp.fft.fft(jax.lax.complex(fp[0::2], fp[1::2]), n=fft_n, axis=-1)

  def one_band(h_b: jax.Array) -> jax.Array:
    y_b = jnp.fft.ifft(z_spec * h_b[None, :], n=fft_n, axis=-1)[:, :window_size]
    re = _safe_sqrt(jnp.mean(y_b.real * y_b.real, axis=-1))
    im = _safe_sqrt(jnp.mean(y_b.imag * y_b.imag, axis=-1))
    return jnp.stack([re, im], axis=-1).reshape(-1)[:num_frames]

  chunk = _band_chunk_size(h_spec.shape[0], fft_n)
  return jax.lax.map(one_band, h_spec, batch_size=chunk)


def _frame_signal(
    x: jax.Array, window_size: int, hop_size: int, hann: jax.Array
) -> jax.Array:
  num_cols = 1 + (x.shape[0] - window_size) // hop_size
  step = jnp.int32(0 if num_cols == 1 else hop_size)
  starts = jnp.arange(num_cols, dtype=jnp.int32) * step
  idx = starts[:, None] + jnp.arange(window_size, dtype=jnp.int32)[None, :]
  return x[idx] * hann[None, :]


def _build_gammatone_spectrograms(
    signals: tuple[jax.typing.ArrayLike, ...],
    sample_rate: int,
    *,
    use_speech_mode: bool = False,
    num_bands: int | None = None,
    min_freq: float = MIN_FREQ,
    max_freq: float | None = None,
    overlap: float = DEFAULT_OVERLAP,
    window_duration: float = DEFAULT_WINDOW_DURATION,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
  """Build linear gammatone spectrograms for one or more signals in one pass."""
  if not signals:
    raise ValueError("signals must be non-empty")
  speech = _as_bool(use_speech_mode, "use_speech_mode")
  xs = tuple(_check.asarray_f32(s, "signal") for s in signals)
  for x in xs:
    _check.check_1d_nonempty(x, "signal")
  fs = _check.check_positive_int(sample_rate, "sample_rate")
  window_size, hop_size, hann = analysis_window(
      fs, overlap=overlap, window_duration=window_duration
  )
  for x in xs:
    if x.shape[0] < window_size:
      raise ValueError(
          f"Too few samples ({x.shape[0]}) in signal to build spectrogram "
          f"(must be greater than {window_size})"
      )

  eff_bands = (
      (SPEECH_MODE_NUM_BANDS if speech else AUDIO_MODE_NUM_BANDS)
      if num_bands is None
      else num_bands
  )
  eff_max_in = (
      (SPEECH_MODE_MAX_FREQ if speech else fs / 2.0)
      if max_freq is None
      else max_freq
  )
  fs, nb, low_f, eff_max = _validate_erb_args(
      fs, eff_bands, min_freq, eff_max_in
  )
  cfs, _ = _design_erb_filters(fs, nb, low_f, eff_max)
  center_freqs = jnp.asarray(cfs, dtype=jnp.float32)

  frame_groups = [_frame_signal(x, window_size, hop_size, hann) for x in xs]
  counts = [f.shape[0] for f in frame_groups]
  # Pad each signal to even frames to isolate ref/deg real/imag FFT pairs.
  zero = jnp.float32(0.0)
  groups = [
      jnp.pad(f, ((0, f.shape[0] % 2), (0, 0)), constant_values=zero)
      if f.shape[0] % 2
      else f
      for f in frame_groups
  ]
  all_frames = (
      groups[0] if len(groups) == 1 else jnp.concatenate(groups, axis=0)
  )
  imp_bytes = _gammatone_impulse_f32_bytes(fs, nb, low_f, eff_max, window_size)
  fft_n = _next_fft_size(2 * window_size - 1)
  h_spec = jnp.fft.fft(
      jnp.frombuffer(imp_bytes, dtype=jnp.float32).reshape(nb, window_size),
      n=fft_n,
      axis=-1,
  )
  spec_all = _filter_frames_rms_fft(all_frames, fft_n, h_spec)

  if len(counts) == 1:
    return (spec_all[:, : counts[0]],), center_freqs
  specs = []
  offset = 0
  for n_cols, group in zip(counts, groups):
    specs.append(spec_all[:, offset : offset + n_cols])
    offset += group.shape[0]
  return tuple(specs), center_freqs


def build_gammatone_spectrogram(
    signal: jax.typing.ArrayLike,
    sample_rate: int,
    *,
    use_speech_mode: bool = False,
    num_bands: int | None = None,
    min_freq: float = MIN_FREQ,
    max_freq: float | None = None,
    overlap: float = DEFAULT_OVERLAP,
    window_duration: float = DEFAULT_WINDOW_DURATION,
) -> tuple[jax.Array, jax.Array]:
  """Build linear gammatone `(spectrogram, center_freq_bands)` (`Build`)."""
  (spec,), center_freqs = _build_gammatone_spectrograms(
      (signal,),
      sample_rate,
      use_speech_mode=use_speech_mode,
      num_bands=num_bands,
      min_freq=min_freq,
      max_freq=max_freq,
      overlap=overlap,
      window_duration=window_duration,
  )
  return spec, center_freqs


def convert_to_db(spectrogram: jax.typing.ArrayLike) -> jax.Array:
  """Convert amplitude values to dB (`Spectrogram::ConvertToDb`)."""
  x = _check.asarray_f32(spectrogram, "spectrogram")
  abs_x = jnp.abs(x)
  safe_x = jnp.where(
      abs_x == jnp.float32(0.0), jnp.float32(_DOUBLE_EPS), abs_x
  )
  return jnp.float32(10.0) * jnp.log10(safe_x)


def prepare_spectrograms(
    ref_spec: jax.typing.ArrayLike,
    deg_spec: jax.typing.ArrayLike,
    *,
    ref_valid_cols: jax.typing.ArrayLike | None = None,
    deg_valid_cols: jax.typing.ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Convert spectrograms to dB and apply floor normalization.

  Ports `MiscAudio::PrepareSpectrogramsForComparison`; optional valid column
  counts keep trailing zero-padded columns out of floor calculations.

  Args:
    ref_spec: 2D `(num_bands, T_ref)` linear reference spectrogram.
    deg_spec: 2D `(num_bands, T_deg)` linear degraded spectrogram.
    ref_valid_cols: Optional active column count for `ref_spec`.
    deg_valid_cols: Optional active column count for `deg_spec`.

  Returns:
    Tuple `(ref_db, deg_db)` of `float32` arrays shifted to a `0.0` dB floor.
  """
  ref = _check.asarray_f32(ref_spec, "ref_spec")
  deg = _check.asarray_f32(deg_spec, "deg_spec")
  _check.check_2d_nonempty(ref, "ref_spec")
  _check.check_2d_nonempty(deg, "deg_spec")
  if ref.shape[0] != deg.shape[0]:
    raise ValueError(
        f"band count mismatch: ref has {ref.shape[0]}, deg has {deg.shape[0]}"
    )

  abs_floor = jnp.float32(NOISE_FLOOR_ABSOLUTE_DB)
  ref = jnp.maximum(convert_to_db(ref), abs_floor)
  deg = jnp.maximum(convert_to_db(deg), abs_floor)

  t_ref_buf, t_deg_buf = ref.shape[1], deg.shape[1]
  n_ref = (
      jnp.int32(t_ref_buf)
      if ref_valid_cols is None
      else _valid_len(ref_valid_cols, t_ref_buf, "ref_valid_cols")
  )
  n_deg = (
      jnp.int32(t_deg_buf)
      if deg_valid_cols is None
      else _valid_len(deg_valid_cols, t_deg_buf, "deg_valid_cols")
  )
  m = min(t_ref_buf, t_deg_buf)
  pair_mask = jnp.arange(m, dtype=jnp.int32) < jnp.minimum(n_ref, n_deg)

  any_max = jnp.maximum(
      jnp.max(ref[:, :m], axis=0), jnp.max(deg[:, :m], axis=0)
  )
  floor_db = any_max - jnp.float32(NOISE_FLOOR_RELATIVE_TO_PEAK_DB)
  eff_floor = jnp.where(pair_mask, floor_db, jnp.float32(-jnp.inf))
  ref = ref.at[:, :m].set(jnp.maximum(ref[:, :m], eff_floor[None, :]))
  deg = deg.at[:, :m].set(jnp.maximum(deg[:, :m], eff_floor[None, :]))

  inf, zero = jnp.float32(jnp.inf), jnp.float32(0.0)
  ref_m = (
      None
      if ref_valid_cols is None
      else (jnp.arange(t_ref_buf, dtype=jnp.int32) < n_ref)[None, :]
  )
  deg_m = (
      None
      if deg_valid_cols is None
      else (jnp.arange(t_deg_buf, dtype=jnp.int32) < n_deg)[None, :]
  )
  ref_min = (
      jnp.min(ref) if ref_m is None else jnp.min(jnp.where(ref_m, ref, inf))
  )
  deg_min = (
      jnp.min(deg) if deg_m is None else jnp.min(jnp.where(deg_m, deg, inf))
  )
  lowest = jnp.minimum(ref_min, deg_min)
  lowest = jnp.where(jnp.isfinite(lowest), lowest, zero)

  ref, deg = ref - lowest, deg - lowest
  if ref_m is not None:
    ref = jnp.where(ref_m, ref, zero)
  if deg_m is not None:
    deg = jnp.where(deg_m, deg, zero)
  return ref, deg
