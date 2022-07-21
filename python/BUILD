load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

licenses(["notice"])

# Python Binding Extension
pybind_extension(
    name = "visqol_lib_py",
    srcs = ["visqol_lib_py.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//:similarity_result_cc_proto",
        "//:visqol_config_cc_proto",
        "//:visqol_lib",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_test(
    name = "visqol_lib_py_test",
    srcs = ["visqol_lib_py_test.py"],
    data = [
        ":visqol_lib_py.so",
        "//testdata:clean_speech/CA01_01.wav",
        "//testdata:clean_speech/transcoded_CA01_01.wav",
    ],
    python_version = "PY3",
    deps = [
        "//:similarity_result_py_pb2",
        "//:visqol_config_py_pb2",
        "@com_google_protobuf//:protobuf_python",
    ],
)
