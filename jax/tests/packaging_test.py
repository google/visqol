"""Tests for bundled weights, runtime import hygiene, modules, and wheel."""

import ast
import importlib.metadata
import importlib.resources
import importlib.resources.abc
import os
import pathlib
import re

import pytest

import visqol_jax
from visqol_jax import cli
from visqol_jax import io
from visqol_jax import lattice
from visqol_jax import mappers
from visqol_jax import svr

_EXPECTED_RUNTIME_MODULES = frozenset({
    "__init__.py",
    "_check.py",
    "alignment.py",
    "cli.py",
    "io.py",
    "lattice.py",
    "mappers.py",
    "nsim.py",
    "patches.py",
    "selector.py",
    "spectrogram.py",
    "svr.py",
    "visqol.py",
})


def _in_site_packages() -> bool:
  return (
      visqol_jax.__file__ is not None
      and "site-packages" in pathlib.Path(visqol_jax.__file__).parts
  )


def _package_files() -> importlib.resources.abc.Traversable:
  return importlib.resources.files(visqol_jax)


def test_models_and_public_modules_in_package():
  """Bundled .txt.gz/.npz weights, py.typed, io, and cli ship in package."""
  pkg_root = _package_files()
  models = pkg_root.joinpath("_model_data")
  assert models.is_dir()
  model_names = {entry.name for entry in models.iterdir()}
  assert model_names == {"speech_lattice.npz", "svr_model.txt.gz"}
  runtime_py = {
      entry.name
      for entry in pkg_root.iterdir()
      if entry.name.endswith(".py") and not entry.name.endswith("_test.py")
  }
  assert runtime_py == _EXPECTED_RUNTIME_MODULES
  assert pkg_root.joinpath("py.typed").is_file()
  assert pkg_root.joinpath("io.py").is_file()
  assert pkg_root.joinpath("cli.py").is_file()
  assert mappers.MODELS == models
  assert visqol_jax.load_wav_as_mono is io.load_wav_as_mono
  assert visqol_jax.visqol_files is io.visqol_files
  assert callable(cli.main)


def test_no_heavy_runtime_imports_across_package():
  """Runtime modules do not import TensorFlow, TFLite, SciPy, or Click."""
  forbidden = frozenset({
      "ai_edge_litert",
      "click",
      "scipy",
      "tensorflow",
      "tflite",
      "tflite_runtime",
  })
  sources: dict[str, str] = {}
  for entry in _package_files().iterdir():
    if (
        entry.is_file()
        and entry.name.endswith(".py")
        and not entry.name.endswith("_test.py")
    ):
      sources[f"pkg:{entry.name}"] = entry.read_text(encoding="utf-8")

  assert len(sources) >= 11
  offenders: list[str] = []
  for label, text in sorted(sources.items()):
    tree = ast.parse(text, filename=label)
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          if alias.name.split(".")[0] in forbidden:
            offenders.append(f"{label}:{node.lineno}:import {alias.name}")
      elif isinstance(node, ast.ImportFrom):
        if (node.module or "").split(".")[0] in forbidden:
          offenders.append(f"{label}:{node.lineno}:from {node.module}")
  assert not offenders, f"Found forbidden runtime imports: {offenders}"


def test_model_loaders_read_from_package_resources():
  assert mappers.MODELS.name == "_model_data"
  lat = lattice.load_lattice_model()
  assert lat.in_keypoints.shape[0] == 85
  assert lat.ensemble_weights.shape == (60,)
  svr_model = svr.load_svr_model()
  assert svr_model.support_vectors.shape == (317, 32)
  assert svr_model.sv_coef.shape == (317,)


def test_installed_wheel_layout_and_entry_point():
  """In site-packages, wheel ships io/cli/models and requires jax + numpy."""
  assert visqol_jax.__file__ is not None
  if not _in_site_packages():
    if (
        os.environ.get("CI", "").lower() in ("1", "true", "yes")
        and "." not in visqol_jax.__name__
    ):
      pytest.fail(
          "Expected visqol_jax to be imported from site-packages in CI, "
          f"got {visqol_jax.__file__}"
      )
    pytest.skip(
        f"visqol_jax is not installed in site-packages ({visqol_jax.__file__})"
    )
  pkg_root = _package_files()
  names = {
      entry.name for entry in pkg_root.iterdir() if entry.name != "__pycache__"
  }
  assert names == _EXPECTED_RUNTIME_MODULES | {"_model_data", "py.typed"}
  assert not any(name.endswith("_test.py") for name in names)

  reqs = importlib.metadata.requires("visqol-jax")
  assert reqs is not None
  core_reqs = {
      match.group(0).lower()
      for req in reqs
      if ";" not in req and (match := re.search(r"^[A-Za-z0-9_.-]+", req))
  }
  assert core_reqs == {"jax", "numpy"}
  assert not any("tensorflow" in req.lower() for req in reqs)

  eps = importlib.metadata.entry_points(group="console_scripts")
  expected_ep = f"{visqol_jax.__name__}.cli:main"
  assert expected_ep == "visqol" + "_jax.cli:main"
  assert any(ep.name == "visqol-jax" and ep.value == expected_ep for ep in eps)
