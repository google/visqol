"""Signal envelope, FFT cross-correlation, and global/patch alignment.

Ports ViSQOL v3.3.3 `envelope.cc`, `xcorr.cc`, `alignment.cc`,
`fast_fourier_transform.cc`, `fft_manager.cc`, and `MiscMath::NextPowTwo`.
"""

import math

import jax
import jax.numpy as jnp

from visqol_jax import _check

_MIN_FFT_SIZE = 32


def next_pow_two(n: int) -> int:
  """Smallest power of two >= `n` (`MiscMath::NextPowTwo`)."""
  iv = _check.check_nonneg_int(n, "n")
  return 0 if iv == 0 else 1 << (iv - 1).bit_length()


def fft_size(n: int) -> int:
  """PFFFT transform length for `n` samples (power of two >= 32)."""
  return max(next_pow_two(n), _MIN_FFT_SIZE)


def _as_sample_rate(sample_rate: float | jax.typing.ArrayLike) -> jax.Array:
  """Validate finite positive scalar `sample_rate` and convert to `float32`."""
  fs = _check.asarray_f32(sample_rate, "sample_rate")
  if fs.ndim != 0:
    raise ValueError(
        f"sample_rate must be a positive scalar, got {sample_rate!r}"
    )
  if not isinstance(fs, jax.core.Tracer):
    val = float(fs)
    if not math.isfinite(val) or val <= 0.0:
      raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
  return fs


def _valid_len(
    length: jax.typing.ArrayLike, buf_len: int, name: str
) -> jax.Array:
  """Validate scalar integer `length` and clamp to `[0, buf_len]`."""
  if _check.is_bool(length):
    raise ValueError(f"{name} must be an integer scalar, got {length!r}")
  if isinstance(length, int):
    return jnp.int32(min(max(length, 0), buf_len))
  try:
    arr = jnp.asarray(length)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError(
        f"{name} must be an integer scalar, got {length!r}"
    ) from exc
  if arr.ndim != 0 or not jnp.issubdtype(arr.dtype, jnp.integer):
    raise ValueError(f"{name} must be an integer scalar, got {length!r}")
  if arr.dtype.itemsize < 4:
    arr = arr.astype(jnp.int32)
  lo = jnp.zeros((), dtype=arr.dtype)
  hi = jnp.asarray(buf_len, dtype=arr.dtype)
  return jnp.clip(arr, lo, hi).astype(jnp.int32)


def _hilbert_scaling(fft_len: int, valid_len: jax.Array) -> jax.Array:
  """One-sided frequency scaling matching C++ `Envelope::Hilbert` quirks."""
  half = fft_len // 2
  # C++ sets bins [1..half-1] to 2.0, keeping Nyquist=0 unless N == fft_len.
  dc = jnp.where(valid_len == 1, jnp.float32(2.0), jnp.float32(1.0))
  nyquist = jnp.where(valid_len == fft_len, jnp.float32(1.0), jnp.float32(0.0))
  scale = jnp.full((half + 1,), jnp.float32(2.0), dtype=jnp.float32)
  return scale.at[0].set(dc).at[half].set(nyquist)


def calc_upper_env(
    signal: jax.typing.ArrayLike,
    valid_len: jax.typing.ArrayLike | None = None,
) -> jax.Array:
  """Compute upper signal envelope (`Envelope::CalcUpperEnv`).

  Preserves C++ PFFFT_REAL inverse transform on the scaled one-sided spectrum.
  When `valid_len` is given for static-shape padding, the transform uses
  `fft_size(len(signal))`, matching unpadded C++ for `valid_len` in the same
  FFT-size bucket.

  Args:
    signal: Non-empty 1D time-domain waveform.
    valid_len: Optional active sample count in `[0, len(signal)]`.

  Returns:
    1D `float32` upper envelope of length `len(signal)` (zero past `valid_len`).
  """
  x = _check.asarray_f32(signal, "signal")
  _check.check_1d_nonempty(x, "signal")
  n_buf = x.shape[0]
  zero = jnp.float32(0.0)

  if valid_len is None:
    mask = None
    n_valid = jnp.int32(n_buf)
    mean = jnp.mean(x)
    centered = x - mean
  else:
    n_valid = _valid_len(valid_len, n_buf, "valid_len")
    mask = jnp.arange(n_buf, dtype=jnp.int32) < n_valid
    denom = jnp.maximum(n_valid, jnp.int32(1)).astype(jnp.float32)
    mean = jnp.sum(jnp.where(mask, x, zero)) / denom
    centered = jnp.where(mask, x - mean, zero)

  fft_len = fft_size(n_buf)
  spec = jnp.fft.rfft(centered, n=fft_len)
  scaled = spec * _hilbert_scaling(fft_len, n_valid)
  inv = jnp.fft.irfft(scaled, n=fft_len)[:n_buf]
  env = jnp.abs(inv) + mean
  return env if mask is None else jnp.where(mask, env, zero)


def find_lowest_lag_index(
    signal_1: jax.typing.ArrayLike,
    signal_2: jax.typing.ArrayLike,
    *,
    len_1: jax.typing.ArrayLike | None = None,
    len_2: jax.typing.ArrayLike | None = None,
) -> jax.Array:
  """Sample lag maximizing cross-correlation (`XCorr::FindLowestLagIndex`).

  Positive lag means `signal_1` trails `signal_2` (`signal_2` should be delayed
  by prepending zeros); negative lag means `signal_2` trails `signal_1`.

  Args:
    signal_1: Non-empty 1D first (reference) signal.
    signal_2: Non-empty 1D second (degraded) signal.
    len_1: Optional active sample count for `signal_1`.
    len_2: Optional active sample count for `signal_2`.

  Returns:
    Scalar `int32` best sample lag in `[-max_lag, max_lag]`.
  """
  s1 = _check.asarray_f32(signal_1, "signal_1")
  s2 = _check.asarray_f32(signal_2, "signal_2")
  _check.check_1d_nonempty(s1, "signal_1")
  _check.check_1d_nonempty(s2, "signal_2")
  n1_buf, n2_buf = s1.shape[0], s2.shape[0]
  zero = jnp.float32(0.0)

  if len_1 is not None:
    n1 = _valid_len(len_1, n1_buf, "len_1")
    s1 = jnp.where(jnp.arange(n1_buf, dtype=jnp.int32) < n1, s1, zero)
  else:
    n1 = jnp.int32(n1_buf)
  if len_2 is not None:
    n2 = _valid_len(len_2, n2_buf, "len_2")
    s2 = jnp.where(jnp.arange(n2_buf, dtype=jnp.int32) < n2, s2, zero)
  else:
    n2 = jnp.int32(n2_buf)

  m_buf = max(n1_buf, n2_buf)
  max_lag_buf = m_buf - 1
  # Matches C++ frexp(2*M - 1, &exp); pow(2, exp).
  fft_points = 1 << (2 * m_buf - 1).bit_length()
  fft_len = max(fft_points, _MIN_FFT_SIZE)
  if n1_buf == n2_buf:
    s12 = jnp.fft.rfft(jnp.stack([s1, s2], axis=0), n=fft_len, axis=-1)
    prod = s12[0] * jnp.conj(s12[1])
  else:
    prod = jnp.fft.rfft(s1, n=fft_len) * jnp.conj(jnp.fft.rfft(s2, n=fft_len))
  res = jnp.fft.irfft(prod, n=fft_len)[:fft_points]
  # Slice from fft_points - max_lag_buf so max_lag_buf == 0 is empty.
  corrs = jnp.concatenate(
      [res[fft_points - max_lag_buf :], res[: max_lag_buf + 1]]
  )

  if len_1 is not None or len_2 is not None:
    i0 = jnp.int32(0)
    max_lag_eff = jnp.where(
        (n1 > i0) & (n2 > i0), jnp.maximum(n1, n2) - jnp.int32(1), i0
    )
    lags = jnp.arange(-max_lag_buf, max_lag_buf + 1, dtype=jnp.int32)
    corrs = jnp.where(
        jnp.abs(lags) <= max_lag_eff, corrs, jnp.float32(-jnp.inf)
    )

  return jax.lax.argmax(corrs, 0, jnp.int32) - jnp.int32(max_lag_buf)


def globally_align(
    reference: jax.typing.ArrayLike,
    degraded: jax.typing.ArrayLike,
    sample_rate: float | jax.typing.ArrayLike,
    *,
    ref_len: jax.typing.ArrayLike | None = None,
    deg_len: jax.typing.ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Align `degraded` to `reference` (`Alignment::GloballyAlign`).

  Args:
    reference: Non-empty 1D reference waveform.
    degraded: Non-empty 1D degraded waveform.
    sample_rate: Positive sample rate in Hz.
    ref_len: Optional active sample count for `reference`.
    deg_len: Optional active sample count for `degraded`.

  Returns:
    Tuple `(aligned_degraded, valid_len, lag_s)` where `aligned_degraded` has
    fixed length `len(degraded) + len(reference) // 2` with zero padding past
    `valid_len`.
  """
  fs = _as_sample_rate(sample_rate)
  ref = _check.asarray_f32(reference, "reference")
  deg = _check.asarray_f32(degraded, "degraded")
  _check.check_1d_nonempty(ref, "reference")
  _check.check_1d_nonempty(deg, "degraded")

  n_ref_buf, n_deg_buf = ref.shape[0], deg.shape[0]
  n_ref = (
      jnp.int32(n_ref_buf)
      if ref_len is None
      else _valid_len(ref_len, n_ref_buf, "ref_len")
  )
  n_deg = (
      jnp.int32(n_deg_buf)
      if deg_len is None
      else _valid_len(deg_len, n_deg_buf, "deg_len")
  )

  if n_ref_buf == n_deg_buf:
    pair = jnp.stack([ref, deg], axis=0)
    if ref_len is None and deg_len is None:
      envs = jax.vmap(calc_upper_env)(pair)
    else:
      envs = jax.vmap(calc_upper_env)(pair, jnp.stack([n_ref, n_deg], axis=0))
    env_ref, env_deg = envs[0], envs[1]
  else:
    env_ref = calc_upper_env(ref, valid_len=None if ref_len is None else n_ref)
    env_deg = calc_upper_env(deg, valid_len=None if deg_len is None else n_deg)
  best_lag = find_lowest_lag_index(
      env_ref,
      env_deg,
      len_1=None if ref_len is None else n_ref,
      len_2=None if deg_len is None else n_deg,
  )
  # Integer |best_lag| > n_ref / 2.0 is identical to |best_lag| > n_ref // 2.
  i0 = jnp.int32(0)
  effective_lag = jnp.where(
      jnp.abs(best_lag) > (n_ref // jnp.int32(2)), i0, best_lag
  )

  src_idx = (
      jnp.arange(n_deg_buf + n_ref_buf // 2, dtype=jnp.int32) - effective_lag
  )
  in_bounds = (src_idx >= i0) & (src_idx < n_deg)
  aligned_deg = jnp.where(
      in_bounds,
      deg[jnp.clip(src_idx, i0, jnp.int32(n_deg_buf - 1))],
      jnp.float32(0.0),
  )
  valid_len = jnp.maximum(n_deg + effective_lag, i0)
  return aligned_deg, valid_len, effective_lag.astype(jnp.float32) / fs


def align_and_truncate(
    reference: jax.typing.ArrayLike,
    degraded: jax.typing.ArrayLike,
    sample_rate: float | jax.typing.ArrayLike,
    *,
    ref_len: jax.typing.ArrayLike | None = None,
    deg_len: jax.typing.ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Align `degraded` and truncate both (`Alignment::AlignAndTruncate`).

  Args:
    reference: Non-empty 1D reference waveform.
    degraded: Non-empty 1D degraded waveform.
    sample_rate: Positive sample rate in Hz.
    ref_len: Optional active sample count for `reference`.
    deg_len: Optional active sample count for `degraded`.

  Returns:
    Tuple `(aligned_ref, aligned_deg, valid_len, lag_s)` where both arrays have
    fixed length `len(reference)` with zeros past `valid_len`.
  """
  fs = _as_sample_rate(sample_rate)
  ref = _check.asarray_f32(reference, "reference")
  deg = _check.asarray_f32(degraded, "degraded")
  aligned_deg, n_deg, lag_s = globally_align(
      ref, deg, fs, ref_len=ref_len, deg_len=deg_len
  )

  n_ref_buf = ref.shape[0]
  n_ref = (
      jnp.int32(n_ref_buf)
      if ref_len is None
      else _valid_len(ref_len, n_ref_buf, "ref_len")
  )
  i0 = jnp.int32(0)
  pos_start = jnp.maximum(jnp.trunc(lag_s * fs).astype(jnp.int32), i0)
  start = jnp.where(n_ref < n_deg, pos_start, i0)
  valid_len = jnp.maximum(jnp.minimum(n_ref, n_deg) - start, i0)

  idx = jnp.arange(n_ref_buf, dtype=jnp.int32)
  src_idx = idx + start
  valid = idx < valid_len
  zero = jnp.float32(0.0)
  out_ref = jnp.where(
      valid, ref[jnp.clip(src_idx, i0, jnp.int32(n_ref_buf - 1))], zero
  )
  out_deg = jnp.where(
      valid,
      aligned_deg[jnp.clip(src_idx, i0, jnp.int32(aligned_deg.shape[0] - 1))],
      zero,
  )
  return out_ref, out_deg, valid_len, lag_s
