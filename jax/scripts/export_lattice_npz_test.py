"""Tests for scripts/export_lattice_npz.py against upstream ViSQOL .tflite."""

import importlib.util
import os
import pathlib
from typing import Any

from click import testing
import numpy as np

_LATTICE_TFLITE_REL = (
    "model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_"
    "ep2400_train1_7_raw.tflite"
)


def _load_converter_module() -> Any:
  script_path = pathlib.Path(__file__).with_name("export_lattice_npz.py")
  spec = importlib.util.spec_from_file_location(
      "export_lattice_npz", script_path
  )
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _find_repo_file(rel_path: str) -> pathlib.Path:
  this_file = pathlib.Path(__file__)
  for base in (this_file, this_file.resolve()):
    for parent in base.parents:
      candidate = parent / rel_path
      if candidate.is_file():
        return candidate
  for env_name in ("RUNFILES_DIR", "TEST_SRCDIR"):
    env_val = os.environ.get(env_name)
    if env_val:
      root = pathlib.Path(env_val)
      if root.is_dir():
        for match in root.rglob(rel_path):
          if match.is_file():
            return match
  raise FileNotFoundError(f"Could not locate {rel_path}")


def _find_bundled_npz() -> pathlib.Path:
  this_file = pathlib.Path(__file__)
  for base in (this_file, this_file.resolve()):
    jax_root = base.parent.parent
    for rel in (
        "_model_data/speech_lattice.npz",
        "src/visqol_jax/_model_data/speech_lattice.npz",
    ):
      candidate = jax_root / rel
      if candidate.is_file():
        return candidate
  raise FileNotFoundError("Could not locate bundled speech_lattice.npz")


def test_export_lattice_npz_matches_bundled_and_tflite(
    tmp_path: pathlib.Path,
):
  converter = _load_converter_module()
  tflite_path = _find_repo_file(_LATTICE_TFLITE_REL)
  bundled_path = _find_bundled_npz()
  out_path = tmp_path / "exported_speech_lattice.npz"

  runner = testing.CliRunner()
  result = runner.invoke(
      converter.main, [str(tflite_path), str(out_path), "--verify"]
  )
  assert result.exit_code == 0, result.output
  assert out_path.is_file()

  with (
      np.load(out_path, allow_pickle=False) as got_npz,
      np.load(bundled_path, allow_pickle=False) as want_npz,
  ):
    assert set(got_npz.files) == set(want_npz.files)
    assert len(want_npz.files) == 12
    for key in want_npz.files:
      assert got_npz[key].dtype == want_npz[key].dtype, key
      assert np.array_equal(got_npz[key], want_npz[key]), key
