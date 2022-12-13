import os
from setuptools import setup

os.system("./setup.sh")

setup(
    name="visqol_lib_py",
    version="3.3.3",
    url="https://github.com/google/visqol",
    description="An objective, full-reference metric for perceived audio quality.",
    package_dir={"": "visqol_lib_py"},
)
