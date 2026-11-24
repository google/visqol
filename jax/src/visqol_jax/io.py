"""Pure-stdlib/JAX WAV reader and file-based ViSQOL v3.3.3 entry point.

Ports ViSQOL v3.3.3 `wav_reader.cc`, `misc_audio.cc`, and `misc_math.cc`.
"""

import array
import logging
import math
import os
import struct
import sys
from typing import BinaryIO

import jax
import jax.numpy as jnp

from visqol_jax import _check
from visqol_jax.visqol import visqol
from visqol_jax.visqol import VisqolOptions
from visqol_jax.visqol import VisqolResult

_LOGGER = logging.getLogger(__name__)
_WAV_FORMAT_PCM = 0x0001
_WAV_FORMAT_EXTENSIBLE = 0xFFFE
_PCM_FMT_SIZE = 16
_EXTENSIBLE_EXTRA_MIN = 24
_BYTES_PER_SAMPLE = 2
# KSDATAFORMAT_SUBTYPE_PCM GUID (00000001-0000-0010-8000-00aa00389b71).
_PCM_SUBFORMAT_GUID = (
    b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
)


def _read_exact(stream: BinaryIO, n: int) -> bytes:
  data = stream.read(n)
  if len(data) != n:
    raise ValueError(
        f"Unexpected EOF while reading WAV header (expected {n} bytes, got"
        f" {len(data)})."
    )
  return data


def load_wav_as_mono(path: str | os.PathLike[str]) -> tuple[jax.Array, int]:
  """Load a 16-bit PCM or extensible WAV file as a mono float32 waveform.

  Supports PCM (`0x0001`) and `WAVE_FORMAT_EXTENSIBLE` (`0xfffe`), skipping
  optional `fmt ` extension bytes, `fact`, and other non-`data` chunks until
  `data`. Normalizes `int16` samples matching C++ `misc_math.cc` and averages
  channels to mono matching `MiscAudio::ToMono`.

  Args:
    path: Path to a 16-bit WAV file.

  Returns:
    `(samples, sample_rate)` with 1D `float32` `jax.Array` mono samples and
    `int` Hz.
  """
  wav_path = _check.check_nonempty_path(path, "path")
  with open(wav_path, "rb") as stream:
    stream.seek(0, os.SEEK_END)
    file_size = stream.tell()
    stream.seek(0)

    if file_size < 12:
      raise ValueError(
          "Invalid WAV header: file smaller than 12-byte RIFF header."
      )
    riff_id, _, wave_id = struct.unpack("<4sI4s", _read_exact(stream, 12))
    if riff_id != b"RIFF" or wave_id != b"WAVE":
      raise ValueError(
          "Invalid WAV header: expected RIFF/WAVE, "
          f"got {riff_id!r}/{wave_id!r}."
      )

    pos = 12
    fmt: tuple[int, int] | None = None
    data_size: int | None = None

    while pos + 8 <= file_size:
      chunk_id, chunk_size = struct.unpack("<4sI", _read_exact(stream, 8))
      pos += 8

      if chunk_id == b"fmt ":
        if fmt is not None:
          raise ValueError("Invalid WAV file: duplicate 'fmt ' chunk.")
        if chunk_size < _PCM_FMT_SIZE or (
            chunk_size != _PCM_FMT_SIZE and chunk_size < _PCM_FMT_SIZE + 2
        ):
          raise ValueError(f"Invalid WAV 'fmt ' chunk size: {chunk_size}.")
        if pos + chunk_size > file_size:
          raise ValueError("Truncated WAV 'fmt ' chunk.")
        (
            format_tag,
            num_channels,
            sample_rate,
            _,
            _,
            bits_per_sample,
        ) = struct.unpack("<HHIIHH", _read_exact(stream, _PCM_FMT_SIZE))
        extra = chunk_size - _PCM_FMT_SIZE
        # RIFF chunks are word-aligned when not ending at EOF.
        pad = chunk_size & 1 if pos + chunk_size < file_size else 0

        if format_tag not in (_WAV_FORMAT_PCM, _WAV_FORMAT_EXTENSIBLE):
          raise ValueError(
              f"Unsupported WAV format tag 0x{format_tag:04x}; "
              "expected 0x0001 (PCM) or 0xfffe (EXTENSIBLE)."
          )
        if format_tag == _WAV_FORMAT_EXTENSIBLE:
          if extra < _EXTENSIBLE_EXTRA_MIN:
            raise ValueError(
                f"Invalid extensible WAV 'fmt ' chunk size: {chunk_size}."
            )
          ext_bytes = _read_exact(stream, _EXTENSIBLE_EXTRA_MIN)
          cb_size, valid_bits, _ = struct.unpack("<HHI", ext_bytes[:8])
          subformat = ext_bytes[8:24]
          if cb_size < 22 or subformat != _PCM_SUBFORMAT_GUID:
            raise ValueError(
                "Unsupported extensible WAV subformat; expected PCM."
            )
          if valid_bits not in (0, 16):
            raise ValueError(
                f"Unsupported WAV valid bit depth: {valid_bits} "
                "(expected 16-bit)."
            )
          rem = extra - _EXTENSIBLE_EXTRA_MIN + pad
          if rem > 0:
            stream.seek(rem, os.SEEK_CUR)
        elif extra + pad > 0:
          stream.seek(extra + pad, os.SEEK_CUR)
        pos += chunk_size + pad

        if num_channels < 1:
          raise ValueError(f"Invalid WAV channel count: {num_channels}.")
        if sample_rate < 1:
          raise ValueError(f"Invalid WAV sample rate: {sample_rate}.")
        if bits_per_sample != 16:
          raise ValueError(
              f"Unsupported WAV bit depth: {bits_per_sample} (expected 16-bit)."
          )
        fmt = (num_channels, sample_rate)
      elif chunk_id == b"data":
        if fmt is None:
          raise ValueError(
              "Invalid WAV file: 'data' chunk appeared before 'fmt '."
          )
        data_size = chunk_size
        break
      else:
        if pos + chunk_size > file_size:
          raise ValueError("Could not find 'data' chunk in WAV file header.")
        pad = chunk_size & 1 if pos + chunk_size < file_size else 0
        stream.seek(chunk_size + pad, os.SEEK_CUR)
        pos += chunk_size + pad

    if fmt is None:
      raise ValueError("Invalid WAV file: missing 'fmt ' chunk.")
    if data_size is None:
      raise ValueError("Invalid WAV file: missing 'data' chunk.")
    if data_size == 0:
      raise ValueError("Invalid WAV file: 'data' chunk is empty.")
    if data_size % _BYTES_PER_SAMPLE != 0:
      raise ValueError(
          f"Invalid WAV 'data' chunk size ({data_size} bytes): "
          "must be even for 16-bit PCM."
      )

    num_channels, sample_rate = fmt
    num_total_samples = data_size // _BYTES_PER_SAMPLE
    if num_total_samples % num_channels != 0:
      raise ValueError(
          f"Invalid WAV 'data' chunk: {num_total_samples} samples is not "
          f"divisible by {num_channels} channels."
      )

    # Bound read to remaining file bytes to avoid huge allocs on corrupt sizes.
    raw = stream.read(min(data_size, max(0, file_size - pos)))

  num_samples_read = len(raw) // _BYTES_PER_SAMPLE
  if num_samples_read < num_total_samples:
    _LOGGER.warning(
        "Number of samples read (%d) was less than the expected number (%d).",
        num_samples_read,
        num_total_samples,
    )
  usable_samples = (num_samples_read // num_channels) * num_channels
  if usable_samples == 0:
    raise ValueError(
        "Invalid WAV file: 'data' chunk contains no readable frames."
    )

  pcm = array.array("h")
  pcm.frombytes(raw[: usable_samples * _BYTES_PER_SAMPLE])
  if sys.byteorder != "little":
    pcm.byteswap()
  # Match C++ misc_math.cc:46-47: float32 division by 32768.0f.
  normalized = jnp.frombuffer(pcm, dtype=jnp.int16).astype(
      jnp.float32
  ) / jnp.float32(32768.0)
  if num_channels == 1:
    return normalized, sample_rate

  # Match C++ MiscAudio::ToMono channel accumulation order.
  frames = normalized.reshape(-1, num_channels)
  mono = frames[:, 0]
  for ch in range(1, num_channels):
    mono = mono + frames[:, ch]
  return mono / jnp.float32(num_channels), sample_rate


def visqol_files(
    reference_path: str | os.PathLike[str],
    degraded_path: str | os.PathLike[str],
    options: VisqolOptions | None = None,
) -> VisqolResult:
  """Run ViSQOL v3.3.3 on a pair of 16-bit WAV files.

  Args:
    reference_path: Path to the reference WAV file.
    degraded_path: Path to the degraded WAV file.
    options: Optional `VisqolOptions` (defaults to audio mode).

  Raises:
    ValueError: If WAV files are invalid, sample rates differ, options reject
      the input, or the pair is unscorable after alignment (`vnsim` is `NaN`).

  Returns:
    `VisqolResult` computed on the mono `float32` downmixed waveforms.
  """
  if options is not None and not isinstance(options, VisqolOptions):
    raise ValueError(
        f"options must be a VisqolOptions or None, got {options!r}"
    )
  ref, ref_sr = load_wav_as_mono(reference_path)
  deg, deg_sr = load_wav_as_mono(degraded_path)
  if ref_sr != deg_sr:
    raise ValueError(
        "Input audio signals have different sample rates: "
        f"reference is {ref_sr} Hz, degraded is {deg_sr} Hz."
    )

  result = visqol(ref, deg, ref_sr, options)
  if math.isnan(float(result.vnsim)):
    raise ValueError(
        "Audio pair could not be scored after alignment "
        "(no valid degraded frames or patches)."
    )
  return result
