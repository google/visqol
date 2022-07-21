load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

licenses(["notice"])  # BSD

py_binary(
    name = "create_conformance_scores",
    srcs = ["create_conformance_scores.py"],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:alignment/degraded.wav",
        "//testdata:alignment/reference.wav",
        "//testdata:clean_speech/CA01_01.wav",
        "//testdata:clean_speech/transcoded_CA01_01.wav",
        "//testdata:short_duration/5_second/guitar48_stereo_5_sec.wav",
        "//testdata/conformance_testdata_subset:castanets48_stereo.wav",
        "//testdata/conformance_testdata_subset:contrabassoon48_stereo.wav",
        "//testdata/conformance_testdata_subset:contrabassoon48_stereo_24kbps_aac.wav",
        "//testdata/conformance_testdata_subset:glock48_stereo.wav",
        "//testdata/conformance_testdata_subset:glock48_stereo_48kbps_aac.wav",
        "//testdata/conformance_testdata_subset:guitar48_stereo.wav",
        "//testdata/conformance_testdata_subset:guitar48_stereo_64kbps_aac.wav",
        "//testdata/conformance_testdata_subset:harpsichord48_stereo.wav",
        "//testdata/conformance_testdata_subset:harpsichord48_stereo_96kbps_mp3.wav",
        "//testdata/conformance_testdata_subset:moonlight48_stereo.wav",
        "//testdata/conformance_testdata_subset:moonlight48_stereo_128kbps_aac.wav",
        "//testdata/conformance_testdata_subset:ravel48_stereo.wav",
        "//testdata/conformance_testdata_subset:ravel48_stereo_128kbps_opus.wav",
        "//testdata/conformance_testdata_subset:sopr48_stereo.wav",
        "//testdata/conformance_testdata_subset:sopr48_stereo_256kbps_aac.wav",
        "//testdata/conformance_testdata_subset:steely48_stereo.wav",
        "//testdata/conformance_testdata_subset:steely48_stereo_lp7.wav",
        "//testdata/conformance_testdata_subset:strauss48_stereo.wav",
        "//testdata/conformance_testdata_subset:strauss48_stereo_lp35.wav",
    ],
    python_version = "PY3",
    deps = [
        "//pyglib:resources",
        "//third_party/py/absl:app",
        "//third_party/visqol/python:visqol_lib_py",
    ],
)

py_binary(
    name = "fit_nsim_to_mos_poly",
    srcs = ["fit_nsim_to_mos_poly.py"],
    deps = ["@numpy"],
)

pytype_strict_binary(
    name = "train_lattice",
    srcs = ["train_lattice.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    test_lib = True,
    deps = [
        "//file/colossus/public:cns",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/attr",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow:tensorflow_compat_v1_estimator",
        "//third_party/py/tensorflow:tensorflow_estimator",
        "//third_party/py/tensorflow_lattice",
        "@numpy",
    ],
)

py_binary(
    name = "make_svm_train_file",
    srcs = ["make_svm_train_file.py"],
)

py_strict_test(
    name = "train_lattice_test",
    srcs = ["train_lattice_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_lattice.testonly_lib",
        "//testing/pybase",
        "//testing/pybase:parameterized",
        "//third_party/py/pandas",
        "@numpy",
    ],
)

pytype_strict_binary(
    name = "convert_saved_model_to_tflite",
    srcs = ["convert_saved_model_to_tflite.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//file/colossus/public:cns",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/tensorflow",
    ],
)
