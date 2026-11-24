import os
import shutil
import subprocess
import sys

from setuptools import Distribution, setup
from setuptools.command.build_ext import build_ext

_BAZEL_TARGETS = [
    "//:similarity_result_py_pb2",
    "//:visqol_config_py_pb2",
    "//python:visqol_lib_py.so",
]
_MODEL_FILES = [
    "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite",
    "libsvm_nu_svr_model.txt",
]


class BazelBuildExt(build_ext):
  """Builds the extension with Bazel and stages it into the wheel."""

  def run(self):
    # Build against the interpreter running this build, not whatever is on PATH.
    env = dict(os.environ, PYTHON_BIN_PATH=sys.executable)
    bazel = ["bazel", "build", "-c", "opt"]
    subprocess.check_call(bazel + _BAZEL_TARGETS, env=env)
    bin_dir = subprocess.check_output(
        ["bazel", "info", "-c", "opt", "bazel-bin"], env=env, text=True
    ).strip()

    pkg = os.path.join(self.build_lib, "visqol")
    for sub in ("pb2", "model"):
      os.makedirs(os.path.join(pkg, sub), exist_ok=True)
    shutil.copy(os.path.join(bin_dir, "python", "visqol_lib_py.so"), pkg)
    for name in ("similarity_result_pb2.py", "visqol_config_pb2.py"):
      shutil.copy(os.path.join(bin_dir, name), os.path.join(pkg, "pb2"))
    for name in _MODEL_FILES:
      shutil.copy(os.path.join("model", name), os.path.join(pkg, "model"))


class BinaryDistribution(Distribution):
  """Forces a platform-specific wheel tag for the native extension."""

  def has_ext_modules(self):
    return True


setup(
    name="visqol",
    version="3.3.3",
    url="https://github.com/google/visqol",
    description="An objective, full-reference metric for perceived audio quality.",
    python_requires=">=3.9",
    install_requires=["numpy", "protobuf>=4.21,<6"],
    packages=[],
    cmdclass={"build_ext": BazelBuildExt},
    distclass=BinaryDistribution,
)
