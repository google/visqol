VISQOL_LIB_PY_DIR=visqol_lib_py

bazel build -c opt //:similarity_result_py_pb2
bazel build -c opt //:visqol_config_py_pb2
bazel build -c opt //python:visqol_lib_py.so

mkdir -rf $VISQOL_LIB_PY_DIR
mkdir $VISQOL_LIB_PY_DIR

cp bazel-bin/*_pb2.py $VISQOL_LIB_PY_DIR
cp bazel-bin/python/visqol_lib_py.so $VISQOL_LIB_PY_DIR
cp model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite $VISQOL_LIB_PY_DIR
cp model/libsvm_nu_svr_model.txt $VISQOL_LIB_PY_DIR
