########################
# Platform Independent #
########################

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# GoogleTest/GoogleMock framework.
git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.10.0",
)

# proto_library, cc_proto_library, and java_proto_library rules implicitly
# depend on @com_google_protobuf for protoc and proto runtimes.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-3.11.1",
    url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.11.1/protobuf-all-3.11.1.tar.gz",
    sha256 = "761bfffc7d53cd01514fa237ca0d3aba5a3cfd8832a71808c0ccc447174fd0da",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Google Abseil Libs
git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp.git",
    tag = "20200923",
)

# LIBSVM
http_archive(
    name = "svm_lib",
    strip_prefix = "libsvm-324",
    # Depending on a recent stable version.
    urls = ["https://github.com/cjlin1/libsvm/archive/v324.zip"],
    sha256 = "401a60bd828bce8870b9eebf5023602028c7751b0db928a6a3bc351560b8b618",
    build_file_content = """
cc_library(
    name = "libsvm",
    srcs = glob(["svm.cpp"]),
    hdrs = glob(["svm.h"]),
    visibility = ["//visibility:public"],
)
""",
)

# Armadillo Headers
http_archive(
    name = "armadillo_headers",
    strip_prefix = "armadillo-9.860.2",
    urls = ["http://sourceforge.net/projects/arma/files/armadillo-9.860.2.tar.xz"],
    sha256 = "d856ea58c18998997bcae6689784d2d3eeb5daf1379d569fddc277fe046a996b",
    build_file_content = """
cc_library(
    name = "armadillo_header",
    hdrs = glob(["include/armadillo", "include/armadillo_bits/*.hpp"]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
)
""",
)

http_archive(
    name = "bazel_skylib",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

# Python bindings
git_repository(
    name = "pybind11_bazel",
    remote = "https://github.com/pybind/pybind11_bazel.git",
    branch = "master",
)

http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.5.0",
  urls = ["https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

# pybind11_abseil does not have an abseil.
new_git_repository(
    name = "pybind11_abseil",
    remote = "https://github.com/pybind/pybind11_abseil.git",
    branch = "experimental",
    build_file_content = """
pybind_library(
    name = "absl_casters",
    hdrs = ["absl_casters.h"],
    data = ["//third_party/py/dateutil"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)


pybind_library(
    name = "absl_numpy_span_caster",
    hdrs = ["absl_numpy_span_caster.h"],
    deps = [
        "@com_google_absl//absl/types:span",
#        "//third_party/py/numpy:headers",  # buildcleaner: keep
#        "//third_party/py/numpy:multiarray",  # buildcleaner: keep
    ],
)

pybind_library(
    name = "status_utils",
    srcs = ["status_utils.cc"],
    hdrs = ["status_utils.h"],
    deps = [
        ":absl_casters",
        ":status_not_ok_exception",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

pybind_library(
    name = "status_casters",
    hdrs = ["status_casters.h"],
    deps = [
        ":status_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
"""
)

new_git_repository(
    name = "pybind11_protobuf",
    remote = "https://github.com/pybind/pybind11_protobuf.git",
    branch = "master",
    build_file_content = """
pybind_library(
    name = "proto_casters",
)
"""
)

# PFFFT
new_git_repository(
    name = "pffft_lib",
    remote = "https://bitbucket.org/jpommier/pffft.git",
    branch = "master",
    build_file_content = """
cc_library(
    name = "pffft_lib",
    srcs = glob(["pffft.c"]),
    hdrs = glob(["pffft.h"]),
	copts = select({
    "@bazel_tools//src/conditions:windows": [
        "/D_USE_MATH_DEFINES",
        "/W0",
	],
    "//conditions:default": [
    ]}),
    visibility = ["//visibility:public"],
)
""",
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "9f9fb8b2f0213989247c9d5c0e814a8451d18d7f",
    remote = "https://github.com/nelhage/rules_boost",
    shallow_since = "1570056263 -0700",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()
