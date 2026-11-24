"""2D boundary convolution and Neurogram Similarity Index Measure (NSIM).

Ports ViSQOL v3.3.3 `convolution_2d.cc` and
`neurogram_similiarity_index_measure.cc`, including edge-replicated padding,
sample standard deviation (`ddof=1`, `0.0` when `valid_cols < 2`), and
static-shape column masking for fine patch realignment.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from visqol_jax import _check
from visqol_jax import alignment

# 3x3 symmetric Gaussian window from neurogram_similiarity_index_measure.cc.
GAUSSIAN_WINDOW: tuple[tuple[float, float, float], ...] = (
    (0.0113033910173052, 0.0838251475442633, 0.0113033910173052),
    (0.0838251475442633, 0.619485845753726, 0.0838251475442633),
    (0.0113033910173052, 0.0838251475442633, 0.0113033910173052),
)
INTENSITY_RANGE = 1.0
_K1 = 0.01
_K2 = 0.03
# pylint: disable-next=protected-access
_valid_len = alignment._valid_len


class PatchSimilarity(NamedTuple):
  """Per-patch NSIM similarity and frequency-band statistics."""

  similarity: jax.Array
  freq_band_means: jax.Array
  freq_band_stddevs: jax.Array
  freq_band_deg_energy: jax.Array


class PatchNsimStats(NamedTuple):
  """Precomputed per-patch edge-padded tensor and local Gaussian moments."""

  padded: jax.Array
  mu: jax.Array
  mu_sq: jax.Array
  sigma_sq: jax.Array


def _safe_sqrt(x: jax.Array) -> jax.Array:
  le_zero = x <= jnp.float32(0.0)
  return jnp.where(
      le_zero,
      jnp.float32(0.0),
      jnp.sqrt(jnp.where(le_zero, jnp.float32(1.0), x)),
  )


def _check_2d_min(x: jax.Array, name: str) -> None:
  if x.ndim < 2 or x.shape[-2] == 0 or x.shape[-1] == 0:
    raise ValueError(
        f"{name} must be at least 2D with non-empty last two axes, got shape"
        f" {x.shape}"
    )


def _as_intensity_range(
    intensity_range: float | jax.typing.ArrayLike,
) -> jax.Array:
  """Validate finite positive scalar `intensity_range` and convert to f32."""
  ir = _check.asarray_f32(intensity_range, "intensity_range")
  if ir.ndim != 0:
    raise ValueError(
        f"intensity_range must be a positive scalar, got {intensity_range!r}"
    )
  if not isinstance(ir, jax.core.Tracer):
    val = float(ir)
    if not math.isfinite(val) or val <= 0.0:
      raise ValueError(
          f"intensity_range must be positive, got {intensity_range!r}"
      )
  return ir


def _edge_pad_2d(
    x: jax.Array, valid_cols: jax.typing.ArrayLike | None = None
) -> jax.Array:
  """Replicate 1-pixel borders (`Convolution2D::AddMatrixBoundary`)."""
  if valid_cols is not None:
    n_valid = _valid_len(valid_cols, x.shape[-1], "valid_cols")
    max_idx = jnp.maximum(n_valid - jnp.int32(1), jnp.int32(0))
    x = x[..., jnp.minimum(jnp.arange(x.shape[-1], dtype=jnp.int32), max_idx)]
  pad_width = ((0, 0),) * (x.ndim - 2) + ((1, 1), (1, 1))
  return jnp.pad(x, pad_width, mode="edge")


def _conv2d_general(padded: jax.Array, k: jax.Array) -> jax.Array:
  lead = padded.shape[:-2]
  flat = padded.reshape((-1, 1, padded.shape[-2], padded.shape[-1]))
  out = jax.lax.conv_general_dilated(
      flat,
      k[None, None, :, :],
      window_strides=(1, 1),
      padding="VALID",
      dimension_numbers=("NCHW", "OIHW", "NCHW"),
      precision=jax.lax.Precision.HIGHEST,
  )
  return out.reshape(lead + out.shape[-2:])


def _conv2d_stencil(padded: jax.Array, k: jax.Array) -> jax.Array:
  """Evaluate valid 2D convolution via an unrolled spatial slice stencil."""
  kh, kw = k.shape
  h, w = padded.shape[-2] - kh + 1, padded.shape[-1] - kw + 1
  rows = []
  for di in range(kh):
    row = k[di, 0] * padded[..., di : di + h, :w]
    for dj in range(1, kw):
      row = row + k[di, dj] * padded[..., di : di + h, dj : dj + w]
    rows.append(row)
  acc = rows[0]
  for row in rows[1:]:
    acc = acc + row
  return acc


def _valid_conv2d(padded: jax.Array, fir_filter: jax.Array) -> jax.Array:
  """Valid 2D convolution on an already-padded `(..., H_p, W_p)` tensor."""
  kh, kw = fir_filter.shape
  # Flip filter along both spatial axes to match C++ filter_index-- convolution.
  k = fir_filter[::-1, ::-1]
  if kh * kw <= 9:
    # CPU prefers Eigen conv; TPU/GPU prefer a fused stencil over C=1 MXU conv.
    return jax.lax.platform_dependent(
        padded, k, cpu=_conv2d_general, default=_conv2d_stencil
    )
  return _conv2d_general(padded, k)


def valid_2d_conv_with_boundary(
    fir_filter: jax.typing.ArrayLike,
    input_matrix: jax.typing.ArrayLike,
    *,
    valid_cols: jax.typing.ArrayLike | None = None,
) -> jax.Array:
  """2D valid convolution with 1-pixel edge pad (`Valid2DConvWithBoundary`)."""
  filt = _check.asarray_f32(fir_filter, "fir_filter")
  x = _check.asarray_f32(input_matrix, "input_matrix")
  _check.check_2d_nonempty(filt, "fir_filter")
  _check_2d_min(x, "input_matrix")
  if x.shape[-2] + 2 < filt.shape[0] or x.shape[-1] + 2 < filt.shape[1]:
    raise ValueError(
        f"padded input {(x.shape[-2] + 2, x.shape[-1] + 2)} smaller than filter"
        f" {filt.shape}"
    )
  return _valid_conv2d(_edge_pad_2d(x, valid_cols=valid_cols), filt)


def _centered_moment_3x3(
    a_pad: jax.Array, a_mu: jax.Array, b_pad: jax.Array, b_mu: jax.Array
) -> jax.Array:
  """Evaluate 3x3 Gaussian-weighted local covariance around `(a_mu, b_mu)`."""
  h, w = a_mu.shape[-2], a_mu.shape[-1]
  acc = jnp.zeros(jnp.broadcast_shapes(a_mu.shape, b_mu.shape), jnp.float32)
  for di in range(3):
    for dj in range(3):
      da = a_pad[..., di : di + h, dj : dj + w] - a_mu
      db = b_pad[..., di : di + h, dj : dj + w] - b_mu
      acc = acc + jnp.float32(GAUSSIAN_WINDOW[2 - di][2 - dj]) * (da * db)
  return acc


def precompute_patch_stats(
    patches: jax.typing.ArrayLike,
    *,
    valid_cols: jax.typing.ArrayLike | None = None,
) -> PatchNsimStats:
  """Precompute edge-padded tensor and 3x3 Gaussian moments for `patches`."""
  x = _check.asarray_f32(patches, "patches")
  _check_2d_min(x, "patches")
  window = jnp.asarray(GAUSSIAN_WINDOW, dtype=jnp.float32)
  padded = _edge_pad_2d(x, valid_cols=valid_cols)
  mu = _valid_conv2d(padded, window)
  mu_sq = mu * mu
  sigma_sq = jnp.maximum(
      _centered_moment_3x3(padded, mu, padded, mu), jnp.float32(0.0)
  )
  return PatchNsimStats(padded=padded, mu=mu, mu_sq=mu_sq, sigma_sq=sigma_sq)


def _unpack_stats(
    stats: PatchNsimStats, prefix: str
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Validate `PatchNsimStats` fields and shapes in `float32`."""
  if not isinstance(stats, PatchNsimStats) or len(stats) != 4:
    raise ValueError(f"{prefix} must be a PatchNsimStats, got {stats!r}")
  pad = _check.asarray_f32(stats.padded, f"{prefix}.padded")
  mu = _check.asarray_f32(stats.mu, f"{prefix}.mu")
  mu_sq = _check.asarray_f32(stats.mu_sq, f"{prefix}.mu_sq")
  sig_sq = _check.asarray_f32(stats.sigma_sq, f"{prefix}.sigma_sq")
  _check_2d_min(pad, f"{prefix}.padded")
  _check_2d_min(mu, f"{prefix}.mu")
  want_pad = mu.shape[:-2] + (mu.shape[-2] + 2, mu.shape[-1] + 2)
  if (
      pad.shape != want_pad
      or mu_sq.shape != mu.shape
      or sig_sq.shape != mu.shape
  ):
    raise ValueError(
        f"patch spatial shape mismatch in {prefix}: padded {pad.shape}, mu"
        f" {mu.shape}, mu_sq {mu_sq.shape}, sigma_sq {sig_sq.shape}"
    )
  return pad, mu, mu_sq, sig_sq


def sim_map_from_stats(
    ref_stats: PatchNsimStats,
    deg_stats: PatchNsimStats,
    *,
    intensity_range: float | jax.typing.ArrayLike = INTENSITY_RANGE,
) -> jax.Array:
  """Compute per-pixel NSIM similarity map from precomputed patch statistics."""
  ir = _as_intensity_range(intensity_range)
  r_pad, r_mu, r_mu_sq, r_sig_sq = _unpack_stats(ref_stats, "ref_stats")
  d_pad, d_mu, d_mu_sq, d_sig_sq = _unpack_stats(deg_stats, "deg_stats")
  if r_mu.shape[-2:] != d_mu.shape[-2:]:
    raise ValueError(
        f"patch spatial shape mismatch: padded {r_pad.shape[-2:]} vs"
        f" {d_pad.shape[-2:]}, mu {r_mu.shape[-2:]} vs {d_mu.shape[-2:]}"
    )
  jnp.broadcast_shapes(r_mu.shape, d_mu.shape)

  k1_ir = jnp.float32(_K1) * ir
  k2_ir = jnp.float32(_K2) * ir
  c1 = k1_ir * k1_ir
  c3 = (k2_ir * k2_ir) * jnp.float32(0.5)
  one = jnp.float32(1.0)

  mu_r_mu_d = r_mu * d_mu
  sigma_r_d = _centered_moment_3x3(r_pad, r_mu, d_pad, d_mu)
  intensity = jnp.minimum(
      (mu_r_mu_d * jnp.float32(2.0) + c1) / (r_mu_sq + d_mu_sq + c1), one
  )
  denom = _safe_sqrt(r_sig_sq) * _safe_sqrt(d_sig_sq) + c3
  structure = jnp.minimum((sigma_r_d + c3) / denom, one)
  return intensity * structure


def measure_patch_similarity(
    ref_patch: jax.typing.ArrayLike,
    deg_patch: jax.typing.ArrayLike,
    *,
    intensity_range: float | jax.typing.ArrayLike = INTENSITY_RANGE,
    valid_cols: jax.typing.ArrayLike | None = None,
) -> PatchSimilarity:
  """Measure 3x3 NSIM patch similarity (`MeasurePatchSimilarity`)."""
  ref = _check.asarray_f32(ref_patch, "ref_patch")
  deg = _check.asarray_f32(deg_patch, "deg_patch")
  _check_2d_min(ref, "ref_patch")
  _check_2d_min(deg, "deg_patch")
  if ref.shape != deg.shape:
    raise ValueError(
        f"ref_patch shape {ref.shape} != deg_patch shape {deg.shape}"
    )
  ir = _as_intensity_range(intensity_range)

  w_max = ref.shape[-1]
  n_valid = (
      None
      if valid_cols is None
      else _valid_len(valid_cols, w_max, "valid_cols")
  )
  sim_map = sim_map_from_stats(
      precompute_patch_stats(ref, valid_cols=n_valid),
      precompute_patch_stats(deg, valid_cols=n_valid),
      intensity_range=ir,
  )

  zero = jnp.float32(0.0)
  if n_valid is None:
    fdeg = jnp.mean(deg, axis=-1)
    fbm = jnp.mean(sim_map, axis=-1)
    if w_max < 2:
      fstd = jnp.zeros_like(fbm)
    else:
      diff = sim_map - fbm[..., None]
      var = jnp.sum(diff * diff, axis=-1) / jnp.float32(w_max - 1)
      fstd = _safe_sqrt(var)
  else:
    col_mask = jnp.arange(w_max, dtype=jnp.int32) < n_valid
    denom = jnp.maximum(n_valid, jnp.int32(1)).astype(jnp.float32)
    fdeg = jnp.sum(jnp.where(col_mask, deg, zero), axis=-1) / denom
    fbm = jnp.sum(jnp.where(col_mask, sim_map, zero), axis=-1) / denom
    diff = jnp.where(col_mask, sim_map - fbm[..., None], zero)
    var_denom = jnp.maximum(n_valid - jnp.int32(1), jnp.int32(1)).astype(
        jnp.float32
    )
    var = jnp.where(
        n_valid >= jnp.int32(2), jnp.sum(diff * diff, axis=-1) / var_denom, zero
    )
    fstd = _safe_sqrt(var)

  return PatchSimilarity(
      similarity=jnp.mean(fbm, axis=-1),
      freq_band_means=fbm,
      freq_band_stddevs=fstd,
      freq_band_deg_energy=fdeg,
  )
