"""Unit and conformance tests for io.py."""

import array
from collections.abc import Sequence
import functools
import logging
import math
import os
import pathlib
import struct
import sys
import tracemalloc

import jax
import jax.numpy as jnp
import pytest

import visqol_jax
from visqol_jax import io as io_mod
from visqol_jax.visqol import VisqolOptions

# KSDATAFORMAT_SUBTYPE_PCM GUID (00000001-0000-0010-8000-00aa00389b71).
_PCM_SUBFORMAT_GUID = (
    b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
)
# KSDATAFORMAT_SUBTYPE_IEEE_FLOAT GUID (00000003-0000-0010-8000-00aa00389b71).
_FLOAT_SUBFORMAT_GUID = (
    b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
)


@functools.cache
def _find_repo_file(relpath: str) -> pathlib.Path:
  """Locate `testdata/...` across Blaze runfiles and repository layouts."""
  this_file = pathlib.Path(__file__)
  for base in (this_file, this_file.resolve()):
    for parent in base.parents:
      candidate = parent / relpath
      if candidate.is_file():
        return candidate
  for env_name in ("RUNFILES_DIR", "TEST_SRCDIR"):
    env_val = os.environ.get(env_name)
    if env_val:
      root = pathlib.Path(env_val)
      if root.is_dir():
        for match in root.rglob(relpath):
          if match.is_file():
            return match
  raise FileNotFoundError(f"Could not locate {relpath}")


def _chunk(chunk_id: bytes, payload: bytes, *, pad: bool = True) -> bytes:
  out = struct.pack("<4sI", chunk_id, len(payload)) + payload
  if pad and len(payload) % 2 == 1:
    out += b"\x00"
  return out


def _pack_pcm16(
    samples: jax.Array | Sequence[int] | Sequence[Sequence[int]],
) -> tuple[bytes, int]:
  flat: list[int] = []
  if isinstance(samples, jax.Array):
    num_channels = 1 if samples.ndim == 1 else samples.shape[1]
    flat = [int(v) for v in samples.reshape(-1).tolist()]
  else:
    seq = list(samples)
    if seq and isinstance(seq[0], Sequence):
      num_channels = len(seq[0])
      for frame in seq:
        assert isinstance(frame, Sequence)
        flat.extend(int(v) for v in frame)
    else:
      num_channels = 1
      for v in seq:
        assert isinstance(v, int)
        flat.append(v)
  pcm = array.array("h", flat)
  if sys.byteorder != "little":
    pcm.byteswap()
  return pcm.tobytes(), num_channels


def _build_wav(
    samples: jax.Array | Sequence[int] | Sequence[Sequence[int]],
    sample_rate: int = 16000,
    *,
    format_tag: int = 0x0001,
    bits_per_sample: int = 16,
    fmt_extra: bytes = b"",
    pre_chunks: tuple[tuple[bytes, bytes], ...] = (),
    mid_chunks: tuple[tuple[bytes, bytes], ...] = (),
    trailing: bytes = b"",
    declared_data_size: int | None = None,
) -> bytes:
  pcm_bytes, num_channels = _pack_pcm16(samples)
  block_align = num_channels * (bits_per_sample // 8)
  avg_bps = sample_rate * block_align
  fmt_body = (
      struct.pack(
          "<HHIIHH",
          format_tag,
          num_channels,
          sample_rate,
          avg_bps,
          block_align,
          bits_per_sample,
      )
      + fmt_extra
  )
  data_len = (
      len(pcm_bytes) if declared_data_size is None else declared_data_size
  )
  body = bytearray(b"WAVE")
  for cid, cdata in pre_chunks:
    body.extend(_chunk(cid, cdata))
  body.extend(_chunk(b"fmt ", fmt_body))
  for cid, cdata in mid_chunks:
    body.extend(_chunk(cid, cdata))
  body.extend(struct.pack("<4sI", b"data", data_len))
  body.extend(pcm_bytes)
  body.extend(trailing)
  return struct.pack("<4sI", b"RIFF", len(body)) + bytes(body)


def _lcg_f32(
    seed: int, n: int, lo: float = -1.0, hi: float = 1.0
) -> jax.Array:
  state = seed & 0xFFFFFFFF
  vals = [0.0] * n
  scale = (hi - lo) / float(0x100000000)
  for i in range(n):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    vals[i] = lo + state * scale
  return jnp.asarray(vals, dtype=jnp.float32)


def _synth_tone(
    sample_rate: int, duration_s: float = 1.5, *, seed: int = 0
) -> jax.Array:
  n = int(sample_rate * duration_s)
  t = jnp.arange(n, dtype=jnp.float32) / jnp.float32(sample_rate)
  sig = (
      jnp.float32(0.35) * jnp.sin(jnp.float32(2.0 * math.pi * 440.0) * t)
      + jnp.float32(0.20) * jnp.sin(jnp.float32(2.0 * math.pi * 1320.0) * t)
      + _lcg_f32(seed, n, lo=-0.05, hi=0.05)
  )
  return jnp.clip(
      jnp.round(sig * jnp.float32(32767.0)),
      jnp.float32(-32768.0),
      jnp.float32(32767.0),
  ).astype(jnp.int16)


def _assert_array_equal(actual: jax.Array, expected: jax.Array) -> None:
  assert isinstance(actual, jax.Array)
  assert actual.dtype == expected.dtype
  assert actual.shape == expected.shape
  assert bool(jnp.array_equal(actual, expected))


def test_public_reexports():
  assert visqol_jax.load_wav_as_mono is io_mod.load_wav_as_mono
  assert visqol_jax.visqol_files is io_mod.visqol_files
  assert "load_wav_as_mono" in visqol_jax.__all__
  assert "visqol_files" in visqol_jax.__all__


def test_load_wav_pcm16_mono_exact_normalization(tmp_path: pathlib.Path):
  raw = [-32768, -16384, -1, 0, 1, 16384, 32767]
  wav_path = tmp_path / "mono.wav"
  wav_path.write_bytes(_build_wav(raw, sample_rate=16000))

  audio, sr = io_mod.load_wav_as_mono(wav_path)
  assert sr == 16000
  assert audio.dtype == jnp.float32
  assert audio.ndim == 1
  expected = jnp.asarray(raw, dtype=jnp.float32) / jnp.float32(32768.0)
  _assert_array_equal(audio, expected)


def test_load_wav_stereo_and_multichannel_to_mono(tmp_path: pathlib.Path):
  stereo = [[-32768, 32767], [1000, -1000], [1234, 5678], [0, 0]]
  stereo_path = tmp_path / "stereo.wav"
  stereo_path.write_bytes(
      _build_wav(
          stereo,
          sample_rate=48000,
          fmt_extra=struct.pack("<H", 0),
          mid_chunks=((b"LIST", b"INFO" + b"\x00" * 10),),
          trailing=b"JUNK\x04\x00\x00\x00abcd",
      )
  )
  audio, sr = io_mod.load_wav_as_mono(stereo_path)
  assert sr == 48000
  assert audio.dtype == jnp.float32
  norm = jnp.asarray(stereo, dtype=jnp.float32) / jnp.float32(32768.0)
  _assert_array_equal(audio, (norm[:, 0] + norm[:, 1]) / jnp.float32(2.0))

  # 3-channel WAV verifies sequential C++ ToMono summation order.
  tri = [[1000, 2000, 3001], [-5000, 1234, 7777]]
  tri_path = tmp_path / "tri.wav"
  tri_path.write_bytes(_build_wav(tri, sample_rate=48000))
  tri_audio, tri_sr = io_mod.load_wav_as_mono(tri_path)
  assert tri_sr == 48000
  tri_norm = jnp.asarray(tri, dtype=jnp.float32) / jnp.float32(32768.0)
  expected_tri = ((tri_norm[:, 0] + tri_norm[:, 1]) + tri_norm[:, 2]) / (
      jnp.float32(3.0)
  )
  _assert_array_equal(tri_audio, expected_tri)


def test_load_wav_extensible_with_fact_and_extra_chunks(tmp_path: pathlib.Path):
  stereo = [[4096, -2048], [8192, 8192], [-16384, 0]]
  ext = struct.pack("<HHI", 22, 16, 0x3) + _PCM_SUBFORMAT_GUID
  fact = struct.pack("<I", len(stereo))
  wav_path = tmp_path / "extensible.wav"
  wav_path.write_bytes(
      _build_wav(
          stereo,
          sample_rate=48000,
          format_tag=0xFFFE,
          fmt_extra=ext,
          pre_chunks=((b"JUNK", b"odd"),),
          mid_chunks=(
              (b"fact", fact),
              (b"LIST", b"INFO123456"),
              (b"DISP", b"abc"),
          ),
      )
  )

  audio, sr = io_mod.load_wav_as_mono(wav_path)
  assert sr == 48000
  assert audio.dtype == jnp.float32
  norm = jnp.asarray(stereo, dtype=jnp.float32) / jnp.float32(32768.0)
  _assert_array_equal(audio, (norm[:, 0] + norm[:, 1]) / jnp.float32(2.0))

  # Extensible WAV without a fact chunk is also accepted.
  no_fact_path = tmp_path / "extensible_nofact.wav"
  no_fact_path.write_bytes(
      _build_wav(stereo, sample_rate=16000, format_tag=0xFFFE, fmt_extra=ext)
  )
  audio2, sr2 = io_mod.load_wav_as_mono(no_fact_path)
  assert sr2 == 16000
  _assert_array_equal(audio2, audio)


def test_load_wav_truncated_payload_warns(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
):
  samples = list(range(10, 22))
  wav_path = tmp_path / "truncated.wav"
  wav_path.write_bytes(
      _build_wav(samples, sample_rate=16000, declared_data_size=40)
  )

  with caplog.at_level(logging.WARNING, logger=io_mod.__name__):
    audio, sr = io_mod.load_wav_as_mono(wav_path)
  assert sr == 16000
  assert audio.shape == (12,)
  expected = jnp.asarray(samples, dtype=jnp.float32) / jnp.float32(32768.0)
  _assert_array_equal(audio, expected)
  assert any("less than the expected" in r.getMessage() for r in caplog.records)

  # Stereo payload truncated mid-frame drops the incomplete trailing sample.
  stereo_partial, _ = _pack_pcm16([1000, 3000, -2000, -4000, 9999])
  wav_stereo = tmp_path / "truncated_stereo.wav"
  body = (
      b"WAVE"
      + _chunk(b"fmt ", struct.pack("<HHIIHH", 1, 2, 16000, 64000, 4, 16))
      + struct.pack("<4sI", b"data", 16)
      + stereo_partial
  )
  wav_stereo.write_bytes(struct.pack("<4sI", b"RIFF", len(body)) + body)
  caplog.clear()
  with caplog.at_level(logging.WARNING, logger=io_mod.__name__):
    audio_st, _ = io_mod.load_wav_as_mono(wav_stereo)
  assert audio_st.shape == (2,)
  assert any("less than the expected" in r.getMessage() for r in caplog.records)

  # Corrupt 4 GiB data_size stays bounded to actual remaining file bytes.
  huge_hdr = tmp_path / "huge_declared_size.wav"
  huge_hdr.write_bytes(
      _build_wav(samples, sample_rate=16000, declared_data_size=0xFFFFFFFC)
  )
  caplog.clear()
  tracemalloc.start()
  try:
    with caplog.at_level(logging.WARNING, logger=io_mod.__name__):
      audio_huge, _ = io_mod.load_wav_as_mono(huge_hdr)
    _, peak_bytes = tracemalloc.get_traced_memory()
  finally:
    tracemalloc.stop()
  _assert_array_equal(audio_huge, expected)
  assert peak_bytes < 1_000_000
  assert any("less than the expected" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("name", "raw_bytes", "match"),
    [
        ("too_short", b"RIFF\x00\x00", "12-byte"),
        ("bad_riff", b"FORM\x24\x00\x00\x00WAVE" + b"\x00" * 32, "RIFF/WAVE"),
        ("bad_wave", b"RIFF\x24\x00\x00\x00AVI " + b"\x00" * 32, "RIFF/WAVE"),
        (
            "data_before_fmt",
            struct.pack("<4sI4s", b"RIFF", 16, b"WAVE")
            + _chunk(b"data", b"\x00\x00\x00\x00"),
            "before 'fmt '",
        ),
        (
            "duplicate_fmt",
            _build_wav(
                [0, 0, 0, 0],
                mid_chunks=((
                    b"fmt ",
                    struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16),
                ),),
            ),
            "duplicate 'fmt '",
        ),
        (
            "missing_fmt",
            struct.pack("<4sI4s", b"RIFF", 12, b"WAVE")
            + _chunk(b"JUNK", b"abcd"),
            "missing 'fmt '",
        ),
        (
            "short_fmt",
            struct.pack("<4sI4s", b"RIFF", 20, b"WAVE")
            + _chunk(b"fmt ", b"\x00" * 12),
            "'fmt ' chunk size",
        ),
        (
            "fmt_size_17",
            struct.pack("<4sI4s", b"RIFF", 26, b"WAVE")
            + _chunk(b"fmt ", b"\x00" * 17),
            "'fmt ' chunk size",
        ),
        (
            "truncated_fmt",
            struct.pack("<4sI4s", b"RIFF", 40, b"WAVE")
            + struct.pack("<4sI", b"fmt ", 40)
            + b"\x00" * 16,
            "Truncated WAV 'fmt '",
        ),
        (
            "ieee_float",
            _build_wav([0, 0, 0, 0], format_tag=0x0003),
            "format tag",
        ),
        (
            "extensible_short_fmt",
            _build_wav(
                [0, 0, 0, 0],
                format_tag=0xFFFE,
                fmt_extra=struct.pack("<H", 0),
            ),
            "extensible WAV 'fmt '",
        ),
        (
            "extensible_float_subformat",
            _build_wav(
                [0, 0, 0, 0],
                format_tag=0xFFFE,
                fmt_extra=struct.pack("<HHI", 22, 16, 0x4)
                + _FLOAT_SUBFORMAT_GUID,
            ),
            "subformat",
        ),
        (
            "extensible_valid_bits_12",
            _build_wav(
                [0, 0, 0, 0],
                format_tag=0xFFFE,
                fmt_extra=struct.pack("<HHI", 22, 12, 0x4)
                + _PCM_SUBFORMAT_GUID,
            ),
            "valid bit depth",
        ),
        (
            "24bit",
            _build_wav([0, 0, 0, 0], bits_per_sample=24),
            "bit depth",
        ),
        (
            "8bit",
            _build_wav([0, 0, 0, 0], bits_per_sample=8),
            "bit depth",
        ),
        (
            "zero_channels",
            struct.pack("<4sI4s", b"RIFF", 36, b"WAVE")
            + _chunk(b"fmt ", struct.pack("<HHIIHH", 1, 0, 16000, 0, 0, 16))
            + _chunk(b"data", b"\x00\x00"),
            "channel count",
        ),
        (
            "zero_sample_rate",
            struct.pack("<4sI4s", b"RIFF", 36, b"WAVE")
            + _chunk(b"fmt ", struct.pack("<HHIIHH", 1, 1, 0, 0, 2, 16))
            + _chunk(b"data", b"\x00\x00"),
            "sample rate",
        ),
        (
            "missing_data",
            struct.pack("<4sI4s", b"RIFF", 28, b"WAVE")
            + _chunk(
                b"fmt ", struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
            ),
            "missing 'data'",
        ),
        (
            "chunk_past_eof",
            struct.pack("<4sI4s", b"RIFF", 40, b"WAVE")
            + _chunk(
                b"fmt ", struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
            )
            + struct.pack("<4sI", b"LIST", 9999)
            + b"abcd",
            "Could not find 'data'",
        ),
        (
            "empty_data",
            _build_wav([]),
            "empty",
        ),
        (
            "odd_data_size",
            _build_wav([0, 0], declared_data_size=3),
            "even",
        ),
        (
            "stereo_indivisible_data_size",
            _build_wav([[0, 0], [0, 0]], declared_data_size=6),
            "not divisible",
        ),
        (
            "zero_readable_bytes",
            _build_wav([], declared_data_size=4),
            "no readable frames",
        ),
    ],
)
def test_load_wav_malformed_errors(
    tmp_path: pathlib.Path, name: str, raw_bytes: bytes, match: str
):
  wav_path = tmp_path / f"{name}.wav"
  wav_path.write_bytes(raw_bytes)
  with pytest.raises(ValueError, match=match):
    io_mod.load_wav_as_mono(wav_path)


def test_load_wav_invalid_path_types():
  for bad_path in ("", "clip\x00.wav", b"clip.wav", 0):
    with pytest.raises(ValueError, match="non-empty str or PathLike"):
      io_mod.load_wav_as_mono(bad_path)  # type: ignore[arg-type]


def test_visqol_files_dtype_sample_rate_and_unscorable(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
):
  ref_16k = tmp_path / "ref_16k.wav"
  deg_16k = tmp_path / "deg_16k.wav"
  ref_48k = tmp_path / "ref_48k.wav"
  deg_48k = tmp_path / "deg_48k.wav"

  sig_16k = _synth_tone(16000, duration_s=1.5, seed=1)
  sig_48k = _synth_tone(48000, duration_s=1.5, seed=2)
  ref_16k.write_bytes(_build_wav(sig_16k, sample_rate=16000))
  deg_16k.write_bytes(_build_wav(sig_16k, sample_rate=16000))
  ref_48k.write_bytes(_build_wav(sig_48k, sample_rate=48000))
  deg_48k.write_bytes(_build_wav(sig_48k, sample_rate=48000))

  res = io_mod.visqol_files(
      ref_16k, deg_16k, VisqolOptions(use_speech_scoring=True)
  )
  assert res.moslqo.dtype == jnp.float32
  assert res.vnsim.dtype == jnp.float32
  assert float(res.vnsim) == pytest.approx(1.0, abs=1e-6)

  # Even with jax_enable_x64=True, WAV loader and visqol_files return float32.
  with jax.enable_x64(True):
    wav_x64, _ = io_mod.load_wav_as_mono(ref_16k)
    res_x64 = io_mod.visqol_files(
        ref_16k, deg_16k, VisqolOptions(use_speech_scoring=True)
    )
  assert wav_x64.dtype == jnp.float32
  assert res_x64.moslqo.dtype == jnp.float32
  assert res_x64.vnsim.dtype == jnp.float32

  with pytest.raises(ValueError, match="VisqolOptions or None"):
    io_mod.visqol_files(
        "/missing/a.wav", "/missing/b.wav", "speech"  # type: ignore[arg-type]
    )

  # Sample-rate mismatch raises ValueError.
  with pytest.raises(ValueError, match="different sample rates"):
    io_mod.visqol_files(ref_48k, deg_16k)

  # Non-48 kHz audio fails by default and succeeds with override.
  with pytest.raises(ValueError, match="48000 Hz"):
    io_mod.visqol_files(ref_16k, deg_16k)
  caplog.clear()
  with caplog.at_level(logging.WARNING, logger=visqol_jax.visqol.__module__):
    res_non48 = io_mod.visqol_files(
        ref_16k,
        deg_16k,
        VisqolOptions(allow_unsupported_sample_rates=True),
    )
  assert float(res_non48.vnsim) == pytest.approx(1.0, abs=1e-6)
  assert any("48 kHz" in r.getMessage() for r in caplog.records)

  # Audio mode at 48 kHz succeeds with default options.
  res_audio = io_mod.visqol_files(ref_48k, deg_48k)
  assert float(res_audio.vnsim) == pytest.approx(1.0, abs=1e-6)

  # Negative global lag shrinking degraded below window_size raises ValueError.
  idx = jnp.arange(300, dtype=jnp.float32)
  hann = jnp.float32(0.5) - jnp.float32(0.5) * jnp.cos(
      jnp.float32(2.0 * math.pi) * idx / jnp.float32(299.0)
  )
  burst = jnp.sin(jnp.float32(2.0 * math.pi * 600.0) * idx / jnp.float32(16000))
  burst_i16 = jnp.round(burst * hann * jnp.float32(30000.0)).astype(jnp.int16)
  unscorable_ref = jnp.zeros(20000, dtype=jnp.int16).at[100:400].set(burst_i16)
  unscorable_deg = jnp.zeros(1400, dtype=jnp.int16).at[600:900].set(burst_i16)
  u_ref_wav = tmp_path / "unscorable_ref.wav"
  u_deg_wav = tmp_path / "unscorable_deg.wav"
  u_ref_wav.write_bytes(_build_wav(unscorable_ref, sample_rate=16000))
  u_deg_wav.write_bytes(_build_wav(unscorable_deg, sample_rate=16000))
  with pytest.raises(ValueError, match="could not be scored"):
    io_mod.visqol_files(
        u_ref_wav, u_deg_wav, VisqolOptions(use_speech_scoring=True)
    )


def test_conformance_wavs():
  ca01_ref = _find_repo_file("testdata/clean_speech/CA01_01.wav")
  ca01_deg = _find_repo_file("testdata/clean_speech/transcoded_CA01_01.wav")
  audio, sr = io_mod.load_wav_as_mono(ca01_ref)
  assert sr == 48000
  assert audio.shape == (131444,)
  assert audio.dtype == jnp.float32

  # Reference values from src/include/conformance.h.
  res_speech_lat = io_mod.visqol_files(
      ca01_ref, ca01_deg, VisqolOptions(use_speech_scoring=True)
  )
  assert float(res_speech_lat.moslqo) == pytest.approx(
      3.3129234313964844, abs=1e-2
  )

  res_speech_exp = io_mod.visqol_files(
      ca01_ref,
      ca01_deg,
      VisqolOptions(use_speech_scoring=True, use_lattice_model=False),
  )
  assert float(res_speech_exp.moslqo) == pytest.approx(
      3.374505555111911, abs=1e-2
  )

  res_ca01_audio = io_mod.visqol_files(ca01_ref, ca01_deg)
  assert float(res_ca01_audio.moslqo) == pytest.approx(
      1.7658378752958486, abs=2e-3
  )

  guitar_ref = _find_repo_file(
      "testdata/conformance_testdata_subset/guitar48_stereo.wav"
  )
  guitar_deg = _find_repo_file(
      "testdata/conformance_testdata_subset/guitar48_stereo_64kbps_aac.wav"
  )
  g_audio, g_sr = io_mod.load_wav_as_mono(guitar_ref)
  assert g_sr == 48000
  assert g_audio.shape == (597784,)
  assert g_audio.dtype == jnp.float32
  res_guitar = io_mod.visqol_files(guitar_ref, guitar_deg)
  assert float(res_guitar.moslqo) == pytest.approx(4.349722308064298, abs=2e-3)
