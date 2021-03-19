package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

exports_files(["LICENSE"])

# Libraries
# =========================================================

proto_library(
    name = "similarity_result",
    srcs = ["src/proto/similarity_result.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "visqol_config",
    srcs = ["src/proto/visqol_config.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "similarity_result_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":similarity_result"],
)

cc_proto_library(
    name = "visqol_config_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":visqol_config"],
)

cc_library(
    name = "visqol_lib",
    srcs = glob(
        [
            "src/*.cc",
            "src/proto/*.cc",
            "src/proto/*.h",
            "src/svr_training/*.cc",
            "src/svr_training/*.h",
            "src/include/*.h",
        ],
        exclude = ["**/main.cc"],
    ),
    hdrs = glob(["src/include/*.h"]),
    copts = select({
        "@bazel_tools//src/conditions:windows": [
            # Windows Compile Opts
            "/D_USE_MATH_DEFINES",
        ],
        "//conditions:default": [
            # Mac/Linux Compile Opts
        ],
    }),
    includes = [
        "src/include",
        "src/proto",
        "src/svr_training",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":similarity_result_cc_proto",
        ":visqol_config_cc_proto",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@armadillo_headers//:armadillo_header",
        "@svm_lib//:libsvm",
        "@pffft_lib//:pffft_lib",
        "@boost//:filesystem",
        "@boost//:system",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

# Application
# =========================================================
cc_binary(
    name = "visqol",
    srcs = ["src/main.cc"],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//model:tcdvoip_nu.568_c5.31474325639_g3.17773760038_model.txt",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":visqol_lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

# Tests
# =========================================================

# This test suite defines all unit tests, excluding the conformance tests,
# SVR training tests and edge case tests.
test_suite(
    name = "all_unit_tests",
    tests = [
        "alignment_test",
        "analysis_window_test",
        "commandline_parser_test",
        "comparison_patches_selector_test",
        "convolution_2d_test",
        "fast_fourier_transform_test",
        "gammatone_filterbank_test",
        "gammatone_spectrogram_builder_test",
        "misc_audio_test",
        "misc_math_test",
        "rms_vad_test",
        "spectrogram_test",
        "test_utility_test",
        "vad_patch_creator_test",
        "visqol_api_test",
        "visqol_manager_test",
        "xcorr_test",
    ],
)

# This test suite defines all edge case tests.
test_suite(
    name = "edge_case_tests",
    tests = [
        "long_duration_test",
        "mismatched_duration_test",
        "multithreading_test",
        "short_duration_test",
    ],
)

cc_library(
    name = "test_utility",
    testonly = True,
    hdrs = ["tests/test_utility.h"],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_test(
    name = "visqol_manager_test",
    size = "large",
    timeout = "long",
    srcs = [
        "tests/visqol_manager_test.cc",
    ],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:clean_speech/CA01_01.wav",
        "//testdata:clean_speech/transcoded_CA01_01.wav",
        "//testdata:filtered_freqs/guitar48_stereo_10k_filtered_freqs.wav",
        "//testdata:mismatched_duration/guitar48_stereo_middle_50ms_cut.wav",
        "//testdata:non_48k_sample_rate/guitar48_stereo_44100Hz.wav",
        "//testdata/conformance_testdata_subset:guitar48_stereo.wav",
        "//testdata/conformance_testdata_subset:guitar48_stereo_64kbps_aac.wav",
    ],
    shard_count = 15,
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_test(
    name = "vad_patch_creator_test",
    srcs = ["tests/vad_patch_creator_test.cc"],
    data = [
        "//testdata:clean_speech/CA01_01.wav",
    ],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "visqol_api_test",
    size = "medium",
    timeout = "long",
    srcs = ["tests/visqol_api_test.cc"],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:clean_speech/CA01_01.wav",
        "//testdata:clean_speech/transcoded_CA01_01.wav",
        "//testdata/conformance_testdata_subset:contrabassoon48_stereo.wav",
        "//testdata/conformance_testdata_subset:contrabassoon48_stereo_24kbps_aac.wav",
    ],
    shard_count = 15,
    deps = [
        ":similarity_result_cc_proto",
        ":visqol_config_cc_proto",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "commandline_parser_test",
    size = "small",
    srcs = [
        "tests/commandline_parser_test.cc",
    ],
    data = [
        "//testdata:example_batch/batch_input.csv",
    ],
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rms_vad_test",
    srcs = ["tests/rms_vad_test.cc"],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "svr_model_training_test",
    size = "medium",
    srcs = [
        "src/svr_training/training_data_file_reader.h",
        "tests/svr_model_training_test.cc",
    ],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:svr_training/training_mat_tcdaudio14_aacvopus15_fvnsims.txt",
        "//testdata:svr_training/training_mat_tcdaudio14_aacvopus15_moslqs.txt",
    ],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_test(
    name = "comparison_patches_selector_test",
    srcs = ["tests/comparison_patches_selector_test.cc"],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "conformance_test",
    size = "large",
    timeout = "long",
    srcs = [
        "tests/conformance_test.cc",
    ],
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
    shard_count = 15,
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fast_fourier_transform_test",
    size = "small",
    srcs = [
        "tests/fast_fourier_transform_test.cc",
    ],
    data = [
        "//testdata:clean_speech/CA01_01.wav",
    ],
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "convolution_2d_test",
    size = "small",
    srcs = [
        "tests/convolution_2d_test.cc",
    ],
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "test_utility_test",
    size = "small",
    srcs = [
        "tests/test_utility_test.cc",
    ],
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "misc_math_test",
    size = "small",
    srcs = ["tests/misc_math_test.cc"],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "spectrogram_test",
    size = "small",
    srcs = [
        "tests/spectrogram_test.cc",
    ],
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "misc_audio_test",
    size = "small",
    srcs = ["tests/misc_audio_test.cc"],
    data = [
        "//testdata:clean_speech/CA01_01.wav",
        "//testdata/conformance_testdata_subset:guitar48_stereo.wav",
    ],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mismatched_duration_test",
    size = "large",
    srcs = [
        "tests/mismatched_duration_test.cc",
    ],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:mismatched_duration/guitar48_stereo_middle_2sec_cut.wav",
        "//testdata:mismatched_duration/guitar48_stereo_middle_50ms_cut.wav",
        "//testdata:mismatched_duration/guitar48_stereo_x2.wav",
        "//testdata/conformance_testdata_subset:guitar48_stereo.wav",
    ],
    shard_count = 15,
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "analysis_window_test",
    size = "small",
    srcs = ["tests/analysis_window_test.cc"],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gammatone_filterbank_test",
    size = "small",
    srcs = ["tests/gammatone_filterbank_test.cc"],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gammatone_spectrogram_builder_test",
    size = "medium",
    srcs = ["tests/gammatone_spectrogram_builder_test.cc"],
    data = [
        "//testdata/conformance_testdata_subset:contrabassoon48_stereo.wav",
        "//testdata/conformance_testdata_subset:contrabassoon48_stereo_24kbps_aac.wav",
        "//testdata/conformance_testdata_subset:guitar48_stereo.wav",
    ],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "xcorr_test",
    size = "small",
    srcs = ["tests/xcorr_test.cc"],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "alignment_test",
    size = "small",
    srcs = ["tests/alignment_test.cc"],
    deps = [
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "multithreading_test",
    size = "medium",
    timeout = "long",
    srcs = [
        "tests/multithreading_test.cc",
    ],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:test_model/cpp_model.txt",
        "//testdata/conformance_testdata_subset:glock48_stereo.wav",
        "//testdata/conformance_testdata_subset:glock48_stereo_48kbps_aac.wav",
        "//testdata/conformance_testdata_subset:guitar48_stereo.wav",
        "//testdata/conformance_testdata_subset:guitar48_stereo_64kbps_aac.wav",
    ],
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_test(
    name = "long_duration_test",
    size = "large",
    srcs = [
        "tests/long_duration_test.cc",
    ],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:long_duration/1_min/guitar48_stereo_deg_25s.wav",
        "//testdata:long_duration/1_min/guitar48_stereo_ref_25s.wav",
    ],
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "short_duration_test",
    size = "large",
    srcs = [
        "tests/short_duration_test.cc",
    ],
    data = [
        "//model:libsvm_nu_svr_model.txt",
        "//testdata:short_duration/10000_sample/guitar48_stereo_10000_sample.wav",
        "//testdata:short_duration/1000_sample/guitar48_stereo_1000_sample.wav",
        "//testdata:short_duration/100_sample/guitar48_stereo_100_sample.wav",
        "//testdata:short_duration/10_sample/guitar48_stereo_10_sample.wav",
        "//testdata:short_duration/1_sample/guitar48_stereo_1_sample.wav",
        "//testdata:short_duration/1_second/guitar48_stereo_1_sec.wav",
        "//testdata:short_duration/5_second/guitar48_stereo_5_sec.wav",
    ],
    deps = [
        ":test_utility",
        ":visqol_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
    ],
)
