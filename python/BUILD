load("//third_party/pybind11/:build_defs.bzl", "pybind_extension")

licenses(["notice"])

# Python Binding Extension
pybind_extension(
    name = "visqol_lib_py",
    srcs = ["visqol_lib_py.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@pybind11_bazel//:pybind11_eigen",
        "@pybind11_abseil//:absl_casters",
        "@pybind11_abseil//:absl_numpy_span_caster",
        "@pybind11_abseil//:status_casters",
        "@pybind11_protobuf//:proto_casters",
        "//third_party/visqol:similarity_result_cc_proto",
        "//:visqol_config_cc_proto",
        "//:visqol_lib",
    ],
)

py_test(
    name = "visqol_lib_py_test",
    srcs = ["visqol_lib_py_test.py"],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:clean_speech/CA01_01.wav",
        "//testdata:clean_speech/transcoded_CA01_01.wav",
    ],
    python_version = "PY3",
    deps = [
        ":visqol_lib_py",
        "//testing/pybase",
        "@numpy//:numpy",
    ],
)
