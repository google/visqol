VISQOL_LIB_PY_DIR=visqol_lib_py

bazel build -c opt //:similarity_result_py_pb2
bazel build -c opt //:visqol_config_py_pb2
bazel build -c opt //python:visqol_lib_py.so

mkdir -rf $VISQOL_LIB_PY_DIR
mkdir $VISQOL_LIB_PY_DIR

mv bazel-bin/*_pb2.py $VISQOL_LIB_PY_DIR
mv bazel-bin/python/visqol_lib_py.so $VISQOL_LIB_PY_DIR
mv bazel-bin/python/visqol_lib_py.so.runfiles/__main__/model/* $VISQOL_LIB_PY_DIR
