"""Command-line interface matching C++ ViSQOL v3.3.3 flags and outputs.

Ports ViSQOL v3.3.3 `main.cc`, `commandline_parser.cc`, and
`sim_results_writer.h`.
"""

import argparse
from collections.abc import Sequence
import csv
import json
import logging
import math
import pathlib
import sys
from typing import TypedDict

import jax

from visqol_jax import io as io_mod
from visqol_jax import lattice
from visqol_jax import selector
from visqol_jax import spectrogram
from visqol_jax import svr
from visqol_jax.visqol import VisqolOptions
from visqol_jax.visqol import VisqolResult

# Matches C++ conformance.h: `#define kVisqolConformanceNumber (333)`.
CONFORMANCE_NUMBER = 333

_LOGGER = logging.getLogger(__name__)
_FVNSIM_RULE = "-" * 39
_PATCH_RULE = "-" * 76


def _parse_bool(value: str) -> bool:
  low = value.strip().lower()
  if low in ("true", "t", "1", "yes", "y"):
    return True
  if low in ("false", "f", "0", "no", "n"):
    return False
  raise argparse.ArgumentTypeError(
      f"Invalid boolean value {value!r}; expected true/false or 1/0."
  )


def _add_bool_flag(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
  """Register an absl-compatible boolean flag (`--flag`/`--noflag`)."""
  parser.add_argument(
      f"--{name}",
      dest=name,
      nargs="?",
      const=True,
      default=default,
      type=_parse_bool,
      metavar="BOOL",
      help=help_text,
  )
  parser.add_argument(
      f"--no{name}",
      dest=name,
      action="store_false",
      default=argparse.SUPPRESS,
      help=argparse.SUPPRESS,
  )


_BOOL_FLAGS: tuple[tuple[str, bool, str], ...] = (
    ("verbose", False, "Enables verbose output in the terminal."),
    (
        "use_speech_mode",
        False,
        "Use wideband speech mode (21 bands <= 8 kHz, RMS-VAD).",
    ),
    (
        "use_lattice_model",
        True,
        "Use deep lattice model in speech mode (default: true).",
    ),
    (
        "use_unscaled_speech_mos_mapping",
        False,
        "Disable 5.0 scaling in speech exponential MOS mapping.",
    ),
    ("skip_mos_mapping", False, "Skip mapping similarity features to MOS-LQO."),
    ("disable_global_alignment", False, "Disable global envelope alignment."),
    (
        "disable_realignment",
        False,
        "Disable per-patch fine time-domain realignment.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
  """Build the argument parser matching C++ ViSQOL v3.3.3 flags."""
  parser = argparse.ArgumentParser(
      prog="visqol-jax",
      description=(
          "Perceptual quality estimator for speech and audio "
          "(ViSQOL v3.3.3 in JAX)."
      ),
      allow_abbrev=False,
  )
  parser.add_argument(
      "--reference_file",
      default="",
      help="The wav file path used as the reference audio.",
  )
  parser.add_argument(
      "--degraded_file",
      default="",
      help="The wav file path used as the degraded audio.",
  )
  parser.add_argument(
      "--batch_input_csv",
      default="",
      help=(
          "Path to a CSV with header 'reference,degraded' listing pairs to "
          "score. Overrides --reference_file and --degraded_file."
      ),
  )
  parser.add_argument(
      "--results_csv",
      default="",
      help="Path to append CSV similarity and MOS-LQO results.",
  )
  parser.add_argument(
      "--output_debug",
      default="",
      help="Path to append proto3-compatible SimilarityResultMsg debug JSON.",
  )
  parser.add_argument(
      "--similarity_to_quality_model",
      default="",
      help=(
          "Optional model path (.txt or .txt.gz libsvm SVR in audio mode; "
          ".npz lattice archive in speech lattice mode)."
      ),
  )
  parser.add_argument(
      "--search_window_radius",
      type=int,
      default=selector.DEFAULT_SEARCH_WINDOW_RADIUS,
      help="Patch search radius (default: 60).",
  )
  for name, default, help_text in _BOOL_FLAGS:
    _add_bool_flag(parser, name, default=default, help_text=help_text)
  return parser


def _clean_zero(value: float) -> float:
  return 0.0 if value == 0.0 else value


def _clean_vec(values: jax.Array | Sequence[float]) -> list[float]:
  raw = values.tolist() if isinstance(values, jax.Array) else values
  return [_clean_zero(float(x)) for x in raw]


def _clean_mat(
    values: jax.Array | Sequence[Sequence[float]],
) -> list[list[float]]:
  raw = values.tolist() if isinstance(values, jax.Array) else values
  return [[_clean_zero(float(x)) for x in row] for row in raw]


def _read_batch_csv(path: pathlib.Path) -> list[tuple[str, str]]:
  """Read `(reference, degraded)` pairs from a batch CSV after its header."""
  pairs: list[tuple[str, str]] = []
  try:
    with path.open("r", encoding="utf-8", newline="") as handle:
      reader = csv.reader(handle)
      next(reader, None)
      for line_no, row in enumerate(reader, start=2):
        if not row or (len(row) == 1 and not row[0].strip()):
          continue
        if len(row) < 2:
          raise ValueError(
              f"Malformed batch CSV row {line_no} in {path}: "
              f"expected 2 columns, got {len(row)}."
          )
        pairs.append((row[0], row[1]))
  except csv.Error as exc:
    raise ValueError(f"Malformed batch CSV {path}: {exc}") from exc
  return pairs


def _format_fvnsim_table(result: VisqolResult) -> str:
  """Format per-band FVNSIM table matching `FormatFVNSIM`."""
  fvnsim = _clean_vec(result.fvnsim)
  fvnsim10 = _clean_vec(result.fvnsim10)
  cfb = _clean_vec(result.center_freq_bands)
  lines = [
      _FVNSIM_RULE,
      "|  FVNSIM  |  FVNSIM10  |  Freq Band  |",
      _FVNSIM_RULE,
  ]
  for v, v10, cf in zip(fvnsim, fvnsim10, cfb, strict=True):
    lines.append(f"| {v:8.6f} | {v10:10.6f} | {cf:9.1f}Hz |")
  lines.append(_FVNSIM_RULE)
  return "\n".join(lines) + "\n\n"


def _format_patch_similarity_table(result: VisqolResult) -> str:
  """Format per-patch similarity table matching `FormatPatchSimilarity`."""
  ps = result.patch_sims
  n = int(ps.num_patches)
  sims = _clean_vec(ps.similarity)[:n]
  ref_s = _clean_vec(ps.ref_patch_start_time)[:n]
  ref_e = _clean_vec(ps.ref_patch_end_time)[:n]
  deg_s = _clean_vec(ps.deg_patch_start_time)[:n]
  deg_e = _clean_vec(ps.deg_patch_end_time)[:n]
  lines = [
      _PATCH_RULE,
      (
          "| Patch Idx | Similarity | Ref Patch: Start - End "
          "| Deg Patch: Start - End |"
      ),
      _PATCH_RULE,
  ]
  for i in range(n):
    lines.append(
        f"| {i:>9d} | {sims[i]:>10.6f} "
        f"| {ref_s[i]:>11.3f}  - {ref_e[i]:>7.3f} "
        f"| {deg_s[i]:>11.3f}  - {deg_e[i]:>7.3f} |"
    )
  lines.append(_PATCH_RULE)
  return "\n".join(lines) + "\n\n"


def format_console_output(
    result: VisqolResult,
    ref_path: str,
    deg_path: str,
    *,
    verbose: bool,
    use_speech_mode: bool,
    use_lattice_model: bool,
    skip_mos_mapping: bool,
) -> str:
  """Format console text matching `SimilarityResultsWriter::WriteToConsole`."""
  moslqo = _clean_zero(float(result.moslqo))
  vnsim = _clean_zero(float(result.vnsim))
  has_moslqo = not skip_mos_mapping and not math.isnan(moslqo) and moslqo != 0.0

  parts = [
      f"ViSQOL conformance version: {CONFORMANCE_NUMBER}\n",
      "Speech mode, " if use_speech_mode else "Audio mode, ",
  ]
  if has_moslqo:
    if use_speech_mode:
      parts.append(
          "lattice model\n" if use_lattice_model else "exponential model\n"
      )
    else:
      parts.append("support vector regression model\n")
  if verbose:
    parts.append("\n")
    parts.append(f"Reference Filepath:\t{ref_path}\n")
    parts.append(f"Degraded Filepath:\t{deg_path}\n")
  if has_moslqo:
    parts.append(f"MOS-LQO:\t\t{moslqo:.6g}\n")
  else:
    parts.append(f"VNSIM:\t\t{vnsim:.6g}\n")
  if verbose:
    parts.append("\n")
    parts.append(_format_fvnsim_table(result))
    parts.append("\n")
    parts.append(_format_patch_similarity_table(result))
    parts.append("\n")
  return "".join(parts)


PatchDebugDict = TypedDict(
    "PatchDebugDict",
    {
        "similarity": float,
        "freqBandMeans": list[float],
        "refPatchStartTime": float,
        "refPatchEndTime": float,
        "degPatchStartTime": float,
        "degPatchEndTime": float,
    },
    total=False,
)

DebugDict = TypedDict(
    "DebugDict",
    {
        "moslqo": float,
        "vnsim": float,
        "fvnsim": list[float],
        "centerFreqBands": list[float],
        "patchSims": list[PatchDebugDict],
        "referenceFilepath": str,
        "degradedFilepath": str,
        "fstdnsim": list[float],
        "fvdegenergy": list[float],
        "alignmentLagS": float,
        "fvnsim10": list[float],
    },
    total=False,
)


def _build_patch_debug_dict(
    sim: float,
    fbm: list[float],
    ref_s: float,
    ref_e: float,
    deg_s: float,
    deg_e: float,
) -> PatchDebugDict:
  """Serialize one `PatchSimilarityMsg` omitting proto3 `0.0` defaults."""
  out: PatchDebugDict = {}
  if sim != 0.0:
    out["similarity"] = sim
  if fbm:
    out["freqBandMeans"] = fbm
  if ref_s != 0.0:
    out["refPatchStartTime"] = ref_s
  if ref_e != 0.0:
    out["refPatchEndTime"] = ref_e
  if deg_s != 0.0:
    out["degPatchStartTime"] = deg_s
  if deg_e != 0.0:
    out["degPatchEndTime"] = deg_e
  return out


def build_debug_dict(
    result: VisqolResult,
    ref_path: str,
    deg_path: str,
    *,
    skip_mos_mapping: bool,
) -> DebugDict:
  """Build a proto3-JSON dict matching C++ `SimilarityResultMsg`."""
  moslqo = _clean_zero(float(result.moslqo))
  vnsim = _clean_zero(float(result.vnsim))
  lag_s = _clean_zero(float(result.alignment_lag_s))
  has_moslqo = not skip_mos_mapping and not math.isnan(moslqo) and moslqo != 0.0

  ps = result.patch_sims
  n = int(ps.num_patches)
  sims = _clean_vec(ps.similarity)[:n]
  fbms = _clean_mat(ps.freq_band_means[:n])
  ref_s = _clean_vec(ps.ref_patch_start_time)[:n]
  ref_e = _clean_vec(ps.ref_patch_end_time)[:n]
  deg_s = _clean_vec(ps.deg_patch_start_time)[:n]
  deg_e = _clean_vec(ps.deg_patch_end_time)[:n]

  patch_sims = [
      _build_patch_debug_dict(
          sims[i], fbms[i], ref_s[i], ref_e[i], deg_s[i], deg_e[i]
      )
      for i in range(n)
  ]
  fvnsim = _clean_vec(result.fvnsim)
  cfb = _clean_vec(result.center_freq_bands)
  fstdnsim = _clean_vec(result.fstdnsim)
  fvdegenergy = _clean_vec(result.fvdegenergy)
  fvnsim10 = _clean_vec(result.fvnsim10)

  # Match proto3 MessageToJsonString field order (tags 1..11) and 0/empty omit.
  payload: DebugDict = {}
  if has_moslqo:
    payload["moslqo"] = moslqo
  if vnsim != 0.0:
    payload["vnsim"] = vnsim
  if fvnsim:
    payload["fvnsim"] = fvnsim
  if cfb:
    payload["centerFreqBands"] = cfb
  if patch_sims:
    payload["patchSims"] = patch_sims
  if ref_path:
    payload["referenceFilepath"] = ref_path
  if deg_path:
    payload["degradedFilepath"] = deg_path
  if fstdnsim:
    payload["fstdnsim"] = fstdnsim
  if fvdegenergy:
    payload["fvdegenergy"] = fvdegenergy
  if lag_s != 0.0:
    payload["alignmentLagS"] = lag_s
  if fvnsim10:
    payload["fvnsim10"] = fvnsim10
  return payload


def write_debug_json(
    path: pathlib.Path,
    result: VisqolResult,
    ref_path: str,
    deg_path: str,
    *,
    skip_mos_mapping: bool,
) -> None:
  """Append proto3 JSON matching `SimilarityResultsWriter::WriteDebugJSON`."""
  payload = build_debug_dict(
      result, ref_path, deg_path, skip_mos_mapping=skip_mos_mapping
  )
  encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
  with path.open("a", encoding="utf-8") as handle:
    handle.write(encoded)


def write_results_csv(
    path: pathlib.Path,
    result: VisqolResult,
    ref_path: str,
    deg_path: str,
    *,
    skip_mos_mapping: bool,
) -> None:
  """Append CSV row matching `SimilarityResultsWriter::WriteResultsToCSV`."""
  write_header = not path.exists()
  fvnsim = _clean_vec(result.fvnsim)
  fvnsim10 = _clean_vec(result.fvnsim10)
  fstdnsim = _clean_vec(result.fstdnsim)
  fvdegenergy = _clean_vec(result.fvdegenergy)
  moslqo = _clean_zero(float(result.moslqo))
  moslqo_val = 0.0 if (skip_mos_mapping or math.isnan(moslqo)) else moslqo

  with path.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    if write_header:
      header = [
          "reference",
          "degraded",
          "moslqo",
          *(f"fvnsim{i}" for i in range(len(fvnsim))),
          *(f"fvnsim10_{i}" for i in range(len(fvnsim10))),
          *(f"fstdnsim{i}" for i in range(len(fstdnsim))),
          *(f"fvdegenergy{i}" for i in range(len(fvdegenergy))),
      ]
      writer.writerow(header)
    row = [
        ref_path,
        deg_path,
        f"{moslqo_val:.9g}",
        *(f"{v:.9g}" for v in fvnsim),
        *(f"{v:.9g}" for v in fvnsim10),
        *(f"{v:.9g}" for v in fstdnsim),
        *(f"{v:.9g}" for v in fvdegenergy),
    ]
    writer.writerow(row)


def _resolve_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
  """Resolve `(reference, degraded)` file pairs from batch CSV or CLI flags."""
  if args.batch_input_csv:
    batch_path = pathlib.Path(args.batch_input_csv)
    if not batch_path.is_file():
      raise FileNotFoundError(f"File not found: {args.batch_input_csv}")
    return _read_batch_csv(batch_path)

  missing = [
      (flag, val)
      for flag, val in (
          ("--reference_file", args.reference_file),
          ("--degraded_file", args.degraded_file),
      )
      if not val or not pathlib.Path(val).is_file()
  ]
  if missing:
    detail = ", ".join(
        f"{flag} ({val!r})" if val else f"{flag} (not provided)"
        for flag, val in missing
    )
    raise FileNotFoundError(f"Missing or unreadable input file(s): {detail}")
  return [(args.reference_file, args.degraded_file)]


def _init_options_and_model(
    args: argparse.Namespace,
) -> VisqolOptions:
  """Build `VisqolOptions` and load mapper weights (`VisqolManager::Init`)."""
  model_path = (
      None
      if args.skip_mos_mapping or not args.similarity_to_quality_model
      else args.similarity_to_quality_model
  )
  options = VisqolOptions(
      use_speech_scoring=args.use_speech_mode,
      use_lattice_model=args.use_lattice_model,
      use_unscaled_speech_mos_mapping=args.use_unscaled_speech_mos_mapping,
      search_window_radius=args.search_window_radius,
      skip_mos_mapping=args.skip_mos_mapping,
      allow_unsupported_sample_rates=True,
      disable_global_alignment=args.disable_global_alignment,
      disable_realignment=args.disable_realignment,
      svr_model_path=model_path,
  )
  # Validate mapper weights upfront to match VisqolManager::Init.
  if not options.skip_mos_mapping:
    if not options.use_speech_scoring:
      if options.use_lattice_model:
        _LOGGER.warning(
            "Lattice models are not yet supported for audio mode, falling back "
            "to SVR model."
        )
      svr.load_svr_model(
          options.svr_model_path, num_features=spectrogram.AUDIO_MODE_NUM_BANDS
      )
    elif options.use_lattice_model:
      lattice.load_lattice_model(
          options.svr_model_path, num_bands=spectrogram.SPEECH_MODE_NUM_BANDS
      )
  return options


def _optional_out_path(raw: str, flag: str) -> pathlib.Path | None:
  if not raw:
    return None
  if "\x00" in raw:
    raise ValueError(f"Invalid {flag} path: embedded NUL byte.")
  return pathlib.Path(raw)


def main(argv: Sequence[str] | None = None) -> int:
  """Entry point for `visqol-jax` CLI."""
  parser = build_parser()
  args = parser.parse_args(argv)

  try:
    pairs = _resolve_pairs(args)
    options = _init_options_and_model(args)
    debug_path = _optional_out_path(args.output_debug, "--output_debug")
    csv_path = _optional_out_path(args.results_csv, "--results_csv")
  except (OSError, ValueError) as exc:
    _LOGGER.error("%s", exc)
    return 1

  had_error = False

  for ref_path, deg_path in pairs:
    try:
      result = io_mod.visqol_files(ref_path, deg_path, options)
    except (OSError, ValueError) as exc:
      _LOGGER.error("Error executing ViSQOL: %s.", exc)
      had_error = True
      continue

    sys.stdout.write(
        format_console_output(
            result,
            ref_path,
            deg_path,
            verbose=args.verbose,
            use_speech_mode=options.use_speech_scoring,
            use_lattice_model=options.use_lattice_model,
            skip_mos_mapping=options.skip_mos_mapping,
        )
    )
    try:
      if debug_path is not None:
        write_debug_json(
            debug_path,
            result,
            ref_path,
            deg_path,
            skip_mos_mapping=options.skip_mos_mapping,
        )
      if csv_path is not None:
        write_results_csv(
            csv_path,
            result,
            ref_path,
            deg_path,
            skip_mos_mapping=options.skip_mos_mapping,
        )
    except (OSError, ValueError) as exc:
      _LOGGER.error("Error writing output file: %s.", exc)
      had_error = True

  return 1 if had_error else 0


if __name__ == "__main__":
  raise SystemExit(main())
