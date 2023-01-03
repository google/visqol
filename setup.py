import os
from setuptools import setup

os.system("bazel build -c opt //:similarity_result_py_pb2")
os.system("bazel build -c opt //:visqol_config_py_pb2")
os.system("bazel build -c opt //python:visqol_lib_py.so")

setup(
    name="visqol",
    version="3.3.3",
    url="https://github.com/google/visqol",
    description="An objective, full-reference metric for perceived audio quality.",
    packages=["visqol", "visqol.model", "visqol.pb2"],
    package_dir={
        "visqol": "bazel-bin/python",
        "visqol.model": "model",
        "visqol.pb2": "bazel-bin",
        "external": "bazel-bin/external"
    },
    package_data={
        "visqol": ["visqol_lib_py.so"],
        "visqol.model": [
            "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite",
            "libsvm_nu_svr_model.txt"
        ]
    }
)
