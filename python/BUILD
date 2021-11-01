load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

licenses(["notice"])

# Python Binding Extension
pybind_extension(
    name = "pyvisqol",
    srcs = ["visqol_lib_py.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//:visqol_lib",
    ],
)

py_test(
    name = "pyvisqol_test",
    srcs = ["pyvisqol_test.py"],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:clean_speech/CA01_01.wav",
        "//testdata:clean_speech/transcoded_CA01_01.wav",
    ],
    python_version = "PY3",
    deps = [
        ":pyvisqol",
        "//testing/pybase",
        "@numpy//:numpy",
    ],
)
