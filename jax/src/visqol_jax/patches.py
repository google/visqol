"""Uniform and RMS-VAD spectrogram patch extraction.

Ports ViSQOL v3.3.3 `image_patch_creator.cc`, `rms_vad.cc`,
`vad_patch_creator.cc`, and `MiscMath::Normalize` in `misc_math.cc`, with
static-shape compaction for `jax.jit` in VAD speech mode.
"""

import math
import operator
from typing import SupportsIndex

import jax
import jax.numpy as jnp

from visqol_jax import _check
from visqol_jax import spectrogram

AUDIO_PATCH_SIZE = 30
SPEECH_PATCH_SIZE = 20
SILENT_CHUNK_COUNT = 3
RMS_THRESHOLD = 5000.0
VOICE_ACTIVITY_PRESENT = 1.0
VOICE_ACTIVITY_ABSENT = 0.0
FRAMES_WITH_VA_THRESHOLD = 1.0

_INT16_SCALE = 32768.0
_INT16_MIN = -32768.0
_INT16_MAX = 32767.0
_INT32_MAX = 0x7FFFFFFF


def _check_patch_size(patch_size: SupportsIndex, *, min_value: int) -> int:
  """Validate integer `patch_size >= min_value`."""
  if (
      _check.is_bool(patch_size)
      or getattr(patch_size, "dtype", None) == jnp.bool_
  ):
    raise ValueError(f"patch_size must be >= {min_value}, got {patch_size!r}")
  try:
    ps = operator.index(patch_size)
  except TypeError as exc:
    raise ValueError(
        f"patch_size must be >= {min_value}, got {patch_size!r}"
    ) from exc
  if ps < min_value:
    raise ValueError(f"patch_size must be >= {min_value}, got {patch_size!r}")
  return ps


def _normalize_signal(signal: jax.Array) -> jax.Array:
  """Normalize by signed max (`MiscMath::Normalize`), avoiding 0/0 NaNs."""
  max_val = jnp.max(signal)
  zero, one = jnp.float32(0.0), jnp.float32(1.0)
  safe_denom = jnp.where(max_val == zero, one, max_val)
  # Match C++ std::min(32767, 0/0)->32767 and std::max(-32768, neg/0)->-32768.
  fallback = jnp.where(signal < zero, -one, one)
  return jnp.where(max_val == zero, fallback, signal / safe_denom)


def _float_to_int16_samples(signal: jax.Array) -> jax.Array:
  """Scale normalized floats to truncated int16 range (`GetVoiceActivity`)."""
  clamped = jnp.clip(
      signal * jnp.float32(_INT16_SCALE),
      jnp.float32(_INT16_MIN),
      jnp.float32(_INT16_MAX),
  )
  # C++ vector<int16_t>::emplace_back truncates toward zero.
  return jnp.trunc(clamped)


def create_uniform_patch_indices(
    spectrum_length: int,
    patch_size: int = AUDIO_PATCH_SIZE,
) -> jax.Array:
  """Create uniform 0-based patch start columns (`CreateRefPatchIndices`)."""
  ps = _check_patch_size(patch_size, min_value=2)
  spec_len = _check.check_nonneg_int(spectrum_length, "spectrum_length")
  if spec_len > _INT32_MAX:
    raise ValueError(
        f"spectrum_length ({spec_len}) exceeds int32 limit ({_INT32_MAX})"
    )
  init_patch_index = ps // 2
  minimum_patch_size = ps + init_patch_index
  if spec_len < minimum_patch_size:
    raise ValueError(
        f"Reference spectrum size ({spec_len}) smaller than minimum patch "
        f"size ({minimum_patch_size})."
    )
  max_index = (
      spec_len - ps
      if init_patch_index < (spec_len - ps)
      else init_patch_index + 1
  )
  return jnp.arange(init_patch_index - 1, max_index - 1, ps, dtype=jnp.int32)


def calc_chunk_rms(chunks: jax.typing.ArrayLike) -> jax.Array:
  """Compute root-mean-square over the last axis (`CalcRootMeanSquare`)."""
  c = _check.asarray_f32(chunks, "chunks")
  if c.ndim == 0 or c.shape[-1] == 0:
    raise ValueError(
        "chunks must be at least 1D with non-empty last axis, got shape"
        f" {c.shape}"
    )
  ms = jnp.mean(c * c, axis=-1)
  pos = ms > jnp.float32(0.0)
  return jnp.where(
      pos, jnp.sqrt(jnp.where(pos, ms, jnp.float32(1.0))), jnp.float32(0.0)
  )


def calc_rms_vad(chunks: jax.typing.ArrayLike) -> jax.Array:
  """Per-chunk RMS voice activity with 3-chunk silence hangover (`RmsVad`)."""
  c = _check.asarray_f32(chunks, "chunks")
  if c.ndim != 2 or c.shape[1] == 0:
    raise ValueError(
        "chunks must be a 2D array with non-empty chunk size, got shape"
        f" {c.shape}"
    )
  n = c.shape[0]
  present = jnp.float32(VOICE_ACTIVITY_PRESENT)
  absent = jnp.float32(VOICE_ACTIVITY_ABSENT)
  head_len = min(n, SILENT_CHUNK_COUNT - 1)
  head = jnp.full((head_len,), present, dtype=jnp.float32)
  if n <= SILENT_CHUNK_COUNT - 1:
    return head
  active = calc_chunk_rms(c) >= jnp.float32(RMS_THRESHOLD)
  silent_tail = (~active[2:]) & (~active[1:-1]) & (~active[:-2])
  tail = jnp.where(silent_tail, absent, present)
  return jnp.concatenate([head, tail])


def get_voice_activity(
    signal: jax.typing.ArrayLike,
    start_sample: int,
    total_samples: int,
    frame_len: int,
) -> jax.Array:
  """Run RMS-VAD over framed float audio (`GetVoiceActivity`)."""
  x = _check.asarray_f32(signal, "signal")
  _check.check_1d_nonempty(x, "signal")
  try:
    start_i = _check.check_nonneg_int(start_sample, "start_sample")
    total_i = _check.check_positive_int(total_samples, "total_samples")
    frame_i = _check.check_positive_int(frame_len, "frame_len")
  except ValueError as exc:
    raise ValueError(
        f"invalid VAD slice parameters: start_sample={start_sample!r}, "
        f"total_samples={total_samples!r}, frame_len={frame_len!r}"
    ) from exc
  if start_i + total_i > x.shape[0]:
    raise ValueError(
        f"signal length ({x.shape[0]}) smaller than requested VAD range "
        f"[{start_i}, {start_i + total_i})"
    )
  num_frames = total_i // frame_i
  segment = x[start_i : start_i + num_frames * frame_i]
  chunks = _float_to_int16_samples(segment).reshape(num_frames, frame_i)
  return calc_rms_vad(chunks)


def create_vad_patch_indices(
    spectrum_length: int,
    ref_signal: jax.typing.ArrayLike,
    sample_rate: int,
    *,
    patch_size: int = SPEECH_PATCH_SIZE,
    overlap: float = spectrogram.DEFAULT_OVERLAP,
    window_duration: float = spectrogram.DEFAULT_WINDOW_DURATION,
) -> tuple[jax.Array, jax.Array]:
  """Select active speech patch indices via RMS-VAD (`CreateRefPatchIndices`).

  Preserves C++ quirks (`MiscMath::Normalize` divides by signed max; frame index
  `first_patch_idx` is passed as `start_sample`) and compacts active indices to
  `[:num_active]` of a fixed `(patch_count,)` array for `jax.jit`.

  Args:
    spectrum_length: Reference spectrogram frame count.
    ref_signal: Non-empty 1D reference waveform.
    sample_rate: Positive sample rate in Hz.
    patch_size: Patch width in spectrogram frames (`>= 2`).
    overlap: Window step fraction (`DEFAULT_OVERLAP`).
    window_duration: Window duration in seconds (`DEFAULT_WINDOW_DURATION`).

  Returns:
    Tuple `(indices_padded, num_active)` of `int32` arrays with shapes
    `(patch_count,)` and `()`.
  """
  ps = _check_patch_size(patch_size, min_value=2)
  spec_len = _check.check_nonneg_int(spectrum_length, "spectrum_length")
  sig = _check.asarray_f32(ref_signal, "ref_signal")
  _check.check_1d_nonempty(sig, "ref_signal")
  window_size, frame_size, _ = spectrogram.analysis_window(
      sample_rate, overlap=overlap, window_duration=window_duration
  )

  first_patch_idx = ps // 2 - 1
  min_spectrum_len = first_patch_idx + ps
  if spec_len < min_spectrum_len:
    raise ValueError(
        f"Reference spectrum size ({spec_len}) smaller than minimum VAD "
        f"patch size ({min_spectrum_len})."
    )
  patch_count = (spec_len - first_patch_idx) // ps
  # Match C++ size_t patch_sample_len = patch_size_ * (size * overlap).
  raw_patch_samples = ps * (window_size * float(overlap))
  if not math.isfinite(raw_patch_samples):
    raise ValueError(f"patch_sample_len overflow for patch_size={ps}")
  patch_sample_len = int(raw_patch_samples)
  total_sample_count = patch_count * patch_sample_len

  # C++ quirk: passes frame index `first_patch_idx` directly as `start_sample`.
  vad = get_voice_activity(
      _normalize_signal(sig), first_patch_idx, total_sample_count, frame_size
  )
  frames_with_va = jnp.sum(
      vad[: patch_count * ps].reshape(patch_count, ps), axis=-1
  )
  keep = frames_with_va >= jnp.float32(FRAMES_WITH_VA_THRESHOLD)

  patch_ids = jnp.arange(patch_count, dtype=jnp.int32)
  candidate_indices = jnp.int32(first_patch_idx) + patch_ids * jnp.int32(ps)
  num_active = jnp.sum(keep, dtype=jnp.int32)
  compacted = candidate_indices[
      jnp.argsort((~keep).astype(jnp.int32), stable=True)
  ]
  indices_padded = jnp.where(patch_ids < num_active, compacted, jnp.int32(0))
  return indices_padded, num_active


def create_patches_from_indices(
    spectrogram_data: jax.typing.ArrayLike,
    patch_indices: jax.typing.ArrayLike,
    patch_size: int,
) -> jax.Array:
  """Gather 2D `(P, num_bands, patch_size)` spectrogram patches."""
  spec = _check.asarray_f32(spectrogram_data, "spectrogram_data")
  _check.check_2d_nonempty(spec, "spectrogram_data")
  ps = _check_patch_size(patch_size, min_value=1)
  if spec.shape[1] < ps:
    raise ValueError(
        f"spectrogram_data has {spec.shape[1]} columns, fewer than"
        f" patch_size={ps}"
    )
  max_start = spec.shape[1] - ps
  if isinstance(patch_indices, (list, tuple)):
    if not patch_indices:
      raw_indices = jnp.zeros((0,), dtype=jnp.int32)
    else:
      for item in patch_indices:
        if _check.is_bool(item) or getattr(item, "dtype", None) == jnp.bool_:
          raise ValueError("patch_indices must have an integer dtype, got bool")
        if isinstance(item, int) and (item < 0 or item > max_start):
          raise ValueError(
              f"patch_indices must be in [0, {max_start}], got {patch_indices}"
          )
      try:
        raw_indices = jnp.asarray(patch_indices)
      except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"patch_indices must be a 1D integer array, got {patch_indices!r}"
        ) from exc
  else:
    if _check.is_bool(patch_indices):
      raise ValueError("patch_indices must be a 1D array, got bool")
    try:
      raw_indices = jnp.asarray(patch_indices)
    except (TypeError, ValueError, OverflowError) as exc:
      raise ValueError(
          f"patch_indices must be a 1D integer array, got {patch_indices!r}"
      ) from exc
  if raw_indices.ndim != 1:
    raise ValueError(
        f"patch_indices must be a 1D array, got shape {raw_indices.shape}"
    )
  if not jnp.issubdtype(raw_indices.dtype, jnp.integer):
    raise ValueError(
        f"patch_indices must have an integer dtype, got {raw_indices.dtype}"
    )
  if not isinstance(raw_indices, jax.core.Tracer) and raw_indices.shape[0] > 0:
    lo_i = int(jnp.min(raw_indices))
    hi_i = int(jnp.max(raw_indices))
    if lo_i < 0 or hi_i > max_start:
      raise ValueError(
          f"patch_indices must be in [0, {max_start}], got {raw_indices}"
      )
  if raw_indices.dtype == jnp.int32 or raw_indices.dtype.itemsize < 4:
    idx_i32 = raw_indices.astype(jnp.int32)
  else:
    lo = jnp.zeros((), dtype=raw_indices.dtype)
    hi = jnp.asarray(max_start, dtype=raw_indices.dtype)
    idx_i32 = jnp.clip(raw_indices, lo, hi).astype(jnp.int32)
  return jax.vmap(
      lambda start_col: jax.lax.dynamic_slice_in_dim(
          spec, start_col, ps, axis=1
      )
  )(idx_i32)
