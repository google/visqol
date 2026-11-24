"""Unit and conformance tests for cli.py."""

import argparse
import array
import csv
import dataclasses
import functools
import gzip
import json
import logging
import math
import os
import pathlib
import struct
import sys
from typing import Literal

import jax
import jax.numpy as jnp
import pytest

import visqol_jax
from visqol_jax import cli
from visqol_jax import io as io_mod
from visqol_jax.selector import PatchSimilarityResult
from visqol_jax.visqol import VisqolOptions
from visqol_jax.visqol import VisqolResult


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


def _write_pcm16_wav(
    path: pathlib.Path, samples: jax.Array, sample_rate: int
) -> pathlib.Path:
  num_channels = 1 if samples.ndim == 1 else samples.shape[1]
  block_align = num_channels * 2
  pcm_arr = array.array("h", [int(v) for v in samples.reshape(-1).tolist()])
  if sys.byteorder != "little":
    pcm_arr.byteswap()
  pcm = pcm_arr.tobytes()
  fmt = struct.pack(
      "<4sIHHIIHH",
      b"fmt ",
      16,
      1,
      num_channels,
      sample_rate,
      sample_rate * block_align,
      block_align,
      16,
  )
  data = struct.pack("<4sI", b"data", len(pcm)) + pcm
  body = b"WAVE" + fmt + data
  path.write_bytes(struct.pack("<4sI", b"RIFF", len(body)) + body)
  return path


def _lcg_f32(seed: int, n: int, lo: float = -1.0, hi: float = 1.0) -> jax.Array:
  state = seed & 0xFFFFFFFF
  vals = [0.0] * n
  scale = (hi - lo) / float(0x100000000)
  for i in range(n):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    vals[i] = lo + state * scale
  return jnp.asarray(vals, dtype=jnp.float32)


def _synth_signal(
    sample_rate: int, duration_s: float = 1.6, *, seed: int = 0
) -> jax.Array:
  n = int(sample_rate * duration_s)
  t = jnp.arange(n, dtype=jnp.float32) / jnp.float32(sample_rate)
  # Modulated envelope gives global alignment a single distinct lag peak.
  env = jnp.float32(0.5) + jnp.float32(0.5) * jnp.sin(
      jnp.float32(2.0 * math.pi * 2.5) * t
  )
  sig = env * (
      jnp.float32(0.35) * jnp.sin(jnp.float32(2.0 * math.pi * 440.0) * t)
      + jnp.float32(0.20) * jnp.sin(jnp.float32(2.0 * math.pi * 1100.0) * t)
      + _lcg_f32(seed, n, lo=-0.04, hi=0.04)
  )
  return jnp.clip(
      jnp.round(sig * jnp.float32(32767.0)),
      jnp.float32(-32768.0),
      jnp.float32(32767.0),
  ).astype(jnp.int16)


def _parse_concatenated_json(text: str) -> list[cli.DebugDict]:
  decoder = json.JSONDecoder()
  pos, out, n = 0, [], len(text)
  while pos < n:
    while pos < n and text[pos].isspace():
      pos += 1
    if pos >= n:
      break
    obj, pos = decoder.raw_decode(text, pos)
    out.append(obj)
  return out


def _assert_bool_flags(ns: argparse.Namespace, **expected: bool) -> None:
  for key, want in expected.items():
    val = getattr(ns, key)
    assert isinstance(val, bool) and val == want, f"{key}: {val!r} != {want!r}"


def test_absl_boolean_flag_forms():
  parser = cli.build_parser()
  help_text = parser.format_help()
  assert ".txt.gz" in help_text
  assert ".npz" in help_text
  assert ".tflite" not in help_text

  defaults = parser.parse_args([])
  _assert_bool_flags(
      defaults,
      verbose=False,
      use_speech_mode=False,
      use_lattice_model=True,
      use_unscaled_speech_mos_mapping=False,
      skip_mos_mapping=False,
      disable_global_alignment=False,
      disable_realignment=False,
  )
  assert defaults.search_window_radius == 60

  a = parser.parse_args([
      "--verbose",
      "--use_speech_mode=true",
      "--nouse_lattice_model",
      "--use_unscaled_speech_mos_mapping=1",
      "--skip_mos_mapping=FALSE",
      "--disable_global_alignment=0",
      "--disable_realignment=True",
      "--search_window_radius=30",
  ])
  _assert_bool_flags(
      a,
      verbose=True,
      use_speech_mode=True,
      use_lattice_model=False,
      use_unscaled_speech_mos_mapping=True,
      skip_mos_mapping=False,
      disable_global_alignment=False,
      disable_realignment=True,
  )
  assert a.search_window_radius == 30

  b = parser.parse_args([
      "--use_lattice_model=false",
      "--use_speech_mode",
      "--nouse_speech_mode",
      "--verbose=0",
  ])
  _assert_bool_flags(
      b,
      use_lattice_model=False,
      use_speech_mode=False,
      verbose=False,
  )

  c = parser.parse_args(["--nouse_lattice_model", "--use_lattice_model=1"])
  _assert_bool_flags(c, use_lattice_model=True)

  for bad_flag in (
      "--use_speech_mode=notabool",
      "--unknown_flag",
      "--verb",
      "--reference_fi=a.wav",
  ):
    with pytest.raises(SystemExit) as exc_info:
      parser.parse_args([bad_flag])
    assert exc_info.value.code == 2


def test_cli_single_pair_verbose_and_modes(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
  sig_16k = _synth_signal(16000, duration_s=1.6, seed=10)
  sig_48k = _synth_signal(48000, duration_s=1.6, seed=11)
  ref_16k = _write_pcm16_wav(tmp_path / "ref16.wav", sig_16k, 16000)
  deg_16k = _write_pcm16_wav(tmp_path / "deg16.wav", sig_16k, 16000)
  ref_48k = _write_pcm16_wav(tmp_path / "ref48.wav", sig_48k, 48000)
  deg_48k = _write_pcm16_wav(tmp_path / "deg48.wav", sig_48k, 48000)

  # Audio mode non-verbose.
  rc = cli.main([f"--reference_file={ref_48k}", f"--degraded_file={deg_48k}"])
  assert rc == 0
  out = capsys.readouterr().out
  assert out.startswith(
      "ViSQOL conformance version: 333\n"
      "Audio mode, support vector regression model\n"
      "MOS-LQO:\t\t"
  )

  # Audio mode with custom gzipped SVR model (.txt.gz).
  cpp_svr = _find_repo_file("testdata/test_model/cpp_model.txt")
  svr_gz = tmp_path / "custom_svr.txt.gz"
  svr_gz.write_bytes(gzip.compress(cpp_svr.read_bytes(), mtime=0))
  rc = cli.main([
      f"--reference_file={ref_48k}",
      f"--degraded_file={deg_48k}",
      f"--similarity_to_quality_model={svr_gz}",
  ])
  assert rc == 0
  assert "MOS-LQO:\t\t" in capsys.readouterr().out

  # Speech lattice mode verbose.
  rc = cli.main([
      f"--reference_file={ref_16k}",
      f"--degraded_file={deg_16k}",
      "--use_speech_mode",
      "--verbose",
  ])
  assert rc == 0
  out = capsys.readouterr().out
  assert "ViSQOL conformance version: 333\nSpeech mode, lattice model\n" in out
  assert f"Reference Filepath:\t{ref_16k}\n" in out
  assert f"Degraded Filepath:\t{deg_16k}\n" in out
  assert "MOS-LQO:\t\t" in out
  assert "|  FVNSIM  |  FVNSIM10  |  Freq Band  |" in out
  assert (
      "| Patch Idx | Similarity | Ref Patch: Start - End "
      "| Deg Patch: Start - End |"
  ) in out
  # FVNSIM and patch table borders match C++ widths (39 and 76).
  out_lines = out.splitlines()
  assert out_lines.count("-" * 39) == 3
  assert out_lines.count("-" * 76) == 3
  fvnsim_rows = [
      line
      for line in out_lines
      if line.startswith("|") and line.endswith("Hz |")
  ]
  assert len(fvnsim_rows) == 21
  assert all(len(row) == 39 for row in fvnsim_rows)
  patch_rows = [
      line
      for line in out_lines
      if line.startswith("|") and len(line) == 76 and "Patch Idx" not in line
  ]
  assert len(patch_rows) > 0
  assert patch_rows[0].startswith("|         0 |")

  # Speech exponential mode.
  rc = cli.main([
      f"--reference_file={ref_16k}",
      f"--degraded_file={deg_16k}",
      "--use_speech_mode=true",
      "--nouse_lattice_model",
      "--use_unscaled_speech_mos_mapping",
  ])
  assert rc == 0
  out = capsys.readouterr().out
  assert "Speech mode, exponential model\nMOS-LQO:\t\t" in out


def test_cli_non_48k_audio_warns_not_errors(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
  sig_16k = _synth_signal(16000, duration_s=1.6, seed=12)
  ref = _write_pcm16_wav(tmp_path / "ref16.wav", sig_16k, 16000)
  deg = _write_pcm16_wav(tmp_path / "deg16.wav", sig_16k, 16000)

  with caplog.at_level(logging.WARNING, logger=visqol_jax.__name__):
    rc = cli.main([f"--reference_file={ref}", f"--degraded_file={deg}"])
  assert rc == 0
  assert "MOS-LQO:\t\t" in capsys.readouterr().out
  assert any("48 kHz" in r.getMessage() for r in caplog.records)


def test_cli_skip_mos_mapping_ignores_nonexistent_model(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
  sig_16k = _synth_signal(16000, duration_s=1.6, seed=13)
  ref = _write_pcm16_wav(tmp_path / "ref.wav", sig_16k, 16000)
  deg = _write_pcm16_wav(tmp_path / "deg.wav", sig_16k, 16000)
  dbg = tmp_path / "debug.json"
  res_csv = tmp_path / "results.csv"

  rc = cli.main([
      f"--reference_file={ref}",
      f"--degraded_file={deg}",
      "--use_speech_mode",
      "--skip_mos_mapping",
      "--similarity_to_quality_model=/does/not/exist/model.txt",
      f"--output_debug={dbg}",
      f"--results_csv={res_csv}",
  ])
  assert rc == 0
  out = capsys.readouterr().out
  assert "VNSIM:\t\t" in out
  assert "MOS-LQO:" not in out

  parsed = json.loads(dbg.read_text(encoding="utf-8"))
  assert "moslqo" not in parsed
  assert parsed["vnsim"] == pytest.approx(1.0, abs=1e-6)

  rows = list(csv.reader(res_csv.read_text(encoding="utf-8").splitlines()))
  assert len(rows) == 2
  assert rows[1][2] == "0"


def test_cli_batch_csv_crlf_append_and_debug_json_schema(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
  sig_a = _synth_signal(16000, duration_s=1.6, seed=20)
  # Shift by 400 samples (25 ms) so global alignment lag is non-zero.
  sig_b = jnp.concatenate([jnp.zeros(400, dtype=jnp.int16), sig_a[:-400]])
  ref = _write_pcm16_wav(tmp_path / "ref.wav", sig_a, 16000)
  deg_same = _write_pcm16_wav(tmp_path / "deg_same.wav", sig_a, 16000)
  deg_lag = _write_pcm16_wav(tmp_path / "deg_lag.wav", sig_b, 16000)

  batch_csv = tmp_path / "batch.csv"
  batch_csv.write_bytes(
      f"reference,degraded\r\n{ref},{deg_same}\r\n\r\n{ref},{deg_lag}\r\n"
      .encode("utf-8")
  )
  res_csv = tmp_path / "results.csv"
  dbg_json = tmp_path / "debug.json"

  # --batch_input_csv overrides --reference_file and --degraded_file.
  rc = cli.main([
      "--reference_file=/nonexistent/ref.wav",
      "--degraded_file=/nonexistent/deg.wav",
      f"--batch_input_csv={batch_csv}",
      f"--results_csv={res_csv}",
      f"--output_debug={dbg_json}",
      "--use_speech_mode",
  ])
  assert rc == 0
  capsys.readouterr()

  # Second call appends row 3 without repeating header and appends JSON obj 3.
  rc2 = cli.main([
      f"--reference_file={ref}",
      f"--degraded_file={deg_lag}",
      f"--results_csv={res_csv}",
      f"--output_debug={dbg_json}",
      "--use_speech_mode",
      "--disable_global_alignment",
  ])
  assert rc2 == 0
  capsys.readouterr()

  csv_rows = list(csv.reader(res_csv.read_text(encoding="utf-8").splitlines()))
  assert len(csv_rows) == 4
  n_bands = 21
  expected_header = [
      "reference",
      "degraded",
      "moslqo",
      *(f"fvnsim{i}" for i in range(n_bands)),
      *(f"fvnsim10_{i}" for i in range(n_bands)),
      *(f"fstdnsim{i}" for i in range(n_bands)),
      *(f"fvdegenergy{i}" for i in range(n_bands)),
  ]
  assert csv_rows[0] == expected_header
  assert csv_rows[1][0] == str(ref)
  assert csv_rows[1][1] == str(deg_same)
  assert csv_rows[2][1] == str(deg_lag)
  assert csv_rows[3][1] == str(deg_lag)
  assert all(len(r) == len(expected_header) for r in csv_rows[1:])

  # Verify :.9g formatting matches computed result values.
  ref_res = io_mod.visqol_files(
      ref, deg_same, VisqolOptions(use_speech_scoring=True)
  )
  assert csv_rows[1][2] == f"{float(ref_res.moslqo):.9g}"
  assert csv_rows[1][3] == f"{float(ref_res.fvnsim[0]):.9g}"

  objs = _parse_concatenated_json(dbg_json.read_text(encoding="utf-8"))
  assert len(objs) == 3

  # Object 0: identical clip has alignment_lag_s == 0.0 -> omitted.
  obj0 = objs[0]
  assert "alignmentLagS" not in obj0
  assert obj0["referenceFilepath"] == str(ref)
  assert obj0["degradedFilepath"] == str(deg_same)
  assert obj0["moslqo"] == pytest.approx(float(ref_res.moslqo), abs=1e-7)
  assert obj0["vnsim"] == pytest.approx(float(ref_res.vnsim), abs=1e-7)
  band_keys: tuple[
      Literal[
          "fvnsim", "fvnsim10", "fstdnsim", "fvdegenergy", "centerFreqBands"
      ],
      ...,
  ] = (
      "fvnsim",
      "fvnsim10",
      "fstdnsim",
      "fvdegenergy",
      "centerFreqBands",
  )
  for key in band_keys:
    assert len(obj0[key]) == n_bands
  n_patches = int(ref_res.patch_sims.num_patches)
  assert len(obj0["patchSims"]) == n_patches
  for patch in obj0["patchSims"]:
    assert set(patch.keys()) == {
        "similarity",
        "freqBandMeans",
        "refPatchStartTime",
        "refPatchEndTime",
        "degPatchStartTime",
        "degPatchEndTime",
    }
    assert len(patch["freqBandMeans"]) == n_bands

  # Object 1: shifted clip has non-zero alignmentLagS (-25 ms).
  assert "alignmentLagS" in objs[1]
  assert objs[1]["alignmentLagS"] == pytest.approx(-0.025, abs=0.005)

  # Object 2: --disable_global_alignment forces lag == 0.0 -> omitted.
  assert "alignmentLagS" not in objs[2]


def test_cli_errors_and_batch_partial_failure(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
  sig_16k = _synth_signal(16000, duration_s=1.6, seed=30)
  sig_48k = _synth_signal(48000, duration_s=1.6, seed=31)
  ref_16k = _write_pcm16_wav(tmp_path / "ref16.wav", sig_16k, 16000)
  deg_16k = _write_pcm16_wav(tmp_path / "deg16.wav", sig_16k, 16000)
  deg_48k = _write_pcm16_wav(tmp_path / "deg48.wav", sig_48k, 48000)

  # Missing input files, malformed 1-column batch CSV, or csv.Error.
  assert cli.main([]) == 1
  assert cli.main(["--batch_input_csv=/nonexistent/batch.csv"]) == 1
  bad_csv = tmp_path / "bad.csv"
  bad_csv.write_text(f"reference,degraded\n{ref_16k}\n", encoding="utf-8")
  assert cli.main([f"--batch_input_csv={bad_csv}"]) == 1

  old_limit = csv.field_size_limit(4)
  try:
    assert cli.main([f"--batch_input_csv={bad_csv}"]) == 1
  finally:
    csv.field_size_limit(old_limit)

  # Nonexistent model when MOS mapping is enabled.
  assert (
      cli.main([
          f"--reference_file={ref_16k}",
          f"--degraded_file={deg_16k}",
          "--similarity_to_quality_model=/nonexistent/svr.txt",
      ])
      == 1
  )

  # Negative search window radius.
  assert (
      cli.main([
          f"--reference_file={ref_16k}",
          f"--degraded_file={deg_16k}",
          "--search_window_radius=-1",
      ])
      == 1
  )

  # Unwritable or NUL-containing output path returns exit code 1.
  for out_flag in (
      f"--results_csv={tmp_path / 'missing_dir' / 'out.csv'}",
      "--results_csv=bad\x00.csv",
      "--output_debug=bad\x00.json",
  ):
    assert (
        cli.main([
            f"--reference_file={ref_16k}",
            f"--degraded_file={deg_16k}",
            "--use_speech_mode",
            out_flag,
        ])
        == 1
    )
  capsys.readouterr()

  # Batch with 1 failing pair (mismatched SR) + 1 valid pair continues -> 1.
  batch_csv = tmp_path / "partial.csv"
  batch_csv.write_text(
      f"reference,degraded\n{ref_16k},{deg_48k}\n{ref_16k},{deg_16k}\n",
      encoding="utf-8",
  )
  dbg = tmp_path / "partial_dbg.json"
  with caplog.at_level(logging.ERROR, logger=cli.__name__):
    rc = cli.main([
        f"--batch_input_csv={batch_csv}",
        f"--output_debug={dbg}",
        "--use_speech_mode",
    ])
  assert rc == 1
  assert "MOS-LQO:\t\t" in capsys.readouterr().out
  objs = _parse_concatenated_json(dbg.read_text(encoding="utf-8"))
  assert len(objs) == 1
  assert objs[0]["degradedFilepath"] == str(deg_16k)


def test_cli_conformance(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
  ca01_ref = _find_repo_file("testdata/clean_speech/CA01_01.wav")
  ca01_deg = _find_repo_file("testdata/clean_speech/transcoded_CA01_01.wav")
  guitar_ref = _find_repo_file(
      "testdata/conformance_testdata_subset/guitar48_stereo.wav"
  )
  guitar_deg = _find_repo_file(
      "testdata/conformance_testdata_subset/guitar48_stereo_64kbps_aac.wav"
  )

  dbg_speech = tmp_path / "ca01_dbg.json"
  csv_speech = tmp_path / "ca01.csv"
  rc = cli.main([
      f"--reference_file={ca01_ref}",
      f"--degraded_file={ca01_deg}",
      "--use_speech_mode",
      "--verbose",
      f"--output_debug={dbg_speech}",
      f"--results_csv={csv_speech}",
  ])
  assert rc == 0
  out = capsys.readouterr().out
  assert "ViSQOL conformance version: 333\nSpeech mode, lattice model\n" in out
  speech_obj = json.loads(dbg_speech.read_text(encoding="utf-8"))
  # Match kConformanceSpeechCA01TranscodedLattice in src/include/conformance.h.
  assert speech_obj["moslqo"] == pytest.approx(3.3129234313964844, abs=1e-2)
  assert len(speech_obj["fvnsim"]) == 21
  assert len(speech_obj["patchSims"]) == 5

  dbg_audio = tmp_path / "guitar_dbg.json"
  rc = cli.main([
      f"--reference_file={guitar_ref}",
      f"--degraded_file={guitar_deg}",
      f"--output_debug={dbg_audio}",
  ])
  assert rc == 0
  capsys.readouterr()
  audio_obj = json.loads(dbg_audio.read_text(encoding="utf-8"))
  assert audio_obj["moslqo"] == pytest.approx(4.349722308064298, abs=2e-3)
  assert len(audio_obj["fvnsim"]) == 32


def test_proto3_debug_json_omits_zero_defaults_and_normalizes_neg_zero(
    tmp_path: pathlib.Path,
):
  # Patch 0 starts at 0.0s in deg; patch 1 is a packet-loss skip (all 0.0).
  ps = PatchSimilarityResult(
      similarity=jnp.array([0.875, -0.0], dtype=jnp.float32),
      freq_band_means=jnp.array([[0.875, -0.0], [0.0, 0.0]], dtype=jnp.float32),
      freq_band_stddevs=jnp.zeros((2, 2), dtype=jnp.float32),
      freq_band_deg_energy=jnp.zeros((2, 2), dtype=jnp.float32),
      ref_patch_start_time=jnp.array([0.125, 0.5], dtype=jnp.float32),
      ref_patch_end_time=jnp.array([0.5, 1.0], dtype=jnp.float32),
      deg_patch_start_time=jnp.array([-0.0, 0.0], dtype=jnp.float32),
      deg_patch_end_time=jnp.array([0.375, 0.0], dtype=jnp.float32),
      num_patches=jnp.int32(2),
  )
  synth = VisqolResult(
      moslqo=jnp.float32(2.5),
      vnsim=jnp.float32(-0.0),
      fvnsim=jnp.array([-0.0, 0.25], dtype=jnp.float32),
      fvnsim10=jnp.array([0.0, 0.125], dtype=jnp.float32),
      fstdnsim=jnp.array([0.0, 0.0625], dtype=jnp.float32),
      fvdegenergy=jnp.array([1.0, 2.0], dtype=jnp.float32),
      alignment_lag_s=jnp.float32(-0.0),
      center_freq_bands=jnp.array([50.0, 120.0], dtype=jnp.float32),
      patch_sims=ps,
  )
  dbg = cli.build_debug_dict(synth, "", "", skip_mos_mapping=False)
  assert list(dbg.keys()) == [
      "moslqo",
      "fvnsim",
      "centerFreqBands",
      "patchSims",
      "fstdnsim",
      "fvdegenergy",
      "fvnsim10",
  ]
  assert list(dbg["patchSims"][0].keys()) == [
      "similarity",
      "freqBandMeans",
      "refPatchStartTime",
      "refPatchEndTime",
      "degPatchEndTime",
  ]
  assert dbg["patchSims"][1] == {
      "freqBandMeans": [0.0, 0.0],
      "refPatchStartTime": 0.5,
      "refPatchEndTime": 1.0,
  }
  dbg_file = tmp_path / "synth_dbg.json"
  csv_file = tmp_path / "synth.csv"
  cli.write_debug_json(dbg_file, synth, "r.wav", "d.wav", skip_mos_mapping=True)
  cli.write_results_csv(
      csv_file, synth, "r.wav", "d.wav", skip_mos_mapping=True
  )
  assert "-0.0" not in dbg_file.read_text(encoding="utf-8")
  assert "-0" not in csv_file.read_text(encoding="utf-8")
  console = cli.format_console_output(
      synth,
      "r.wav",
      "d.wav",
      verbose=True,
      use_speech_mode=True,
      use_lattice_model=True,
      skip_mos_mapping=True,
  )
  assert "-0" not in console
  expected_fvnsim = (
      f"{'-' * 39}\n"
      "|  FVNSIM  |  FVNSIM10  |  Freq Band  |\n"
      f"{'-' * 39}\n"
      "| 0.000000 |   0.000000 |      50.0Hz |\n"
      "| 0.250000 |   0.125000 |     120.0Hz |\n"
      f"{'-' * 39}\n"
  )
  expected_patches = (
      f"{'-' * 76}\n"
      "| Patch Idx | Similarity | Ref Patch: Start - End "
      "| Deg Patch: Start - End |\n"
      f"{'-' * 76}\n"
      "|         0 |   0.875000 |       0.125  -   0.500 "
      "|       0.000  -   0.375 |\n"
      "|         1 |   0.000000 |       0.500  -   1.000 "
      "|       0.000  -   0.000 |\n"
      f"{'-' * 76}\n"
  )
  assert expected_fvnsim in console
  assert expected_patches in console

  mismatched = dataclasses.replace(
      synth, center_freq_bands=jnp.array([50.0], dtype=jnp.float32)
  )
  with pytest.raises(ValueError):
    cli.format_console_output(
        mismatched,
        "r.wav",
        "d.wav",
        verbose=True,
        use_speech_mode=True,
        use_lattice_model=True,
        skip_mos_mapping=True,
    )
