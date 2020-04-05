########################
# Platform Independent #
########################

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# GoogleTest/GoogleMock framework.
http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-4e4df226fc197c0dda6e37f5c8c3845ca1e73a49",
    # Google Test has not produced a recent release, so instead depending on a recent stable commit.
    url = "https://github.com/google/googletest/archive/4e4df226fc197c0dda6e37f5c8c3845ca1e73a49.zip",
    sha256 = "d4179caf54410968d1fff0b869e7d74803dd30209ee6645ccf1ca65ab6cf5e5a",
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
http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-20190808",
    url = "https://github.com/abseil/abseil-cpp/archive/20190808.zip",
    sha256 = "0b62fc2d00c2b2bc3761a892a17ac3b8af3578bd28535d90b4c914b0a7460d4e",
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
    strip_prefix = "armadillo-9.860.1",
    urls = ["http://sourceforge.net/projects/arma/files/armadillo-9.860.1.tar.xz"],
    sha256 = "1603888ab73b7f0588df1a37a464436eb0ff6b1372a9962ee1424b4329f165a9",
    build_file_content = """
cc_library(
    name = "armadillo_header",
    hdrs = glob(["include/armadillo", "include/armadillo_bits/*.hpp"]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
)
""",
)


##################
# Platform Linux #
##################
# PFFFT - Linux
http_archive(
    name = "pffft_lib_linux",
    strip_prefix = "jpommier-pffft-29e4f76ac53b",
    urls = ["https://bitbucket.org/jpommier/pffft/get/29e4f76ac53b.zip"],
    sha256 = "bb10afba127904a0c6c553fa445082729b7d72373511bda1b12a5be0e03f318a",
    build_file_content = """
cc_library(
    name = "pffft_linux",
    srcs = glob(["pffft.c"]),
    hdrs = glob(["pffft.h"]),
    visibility = ["//visibility:public"],
)
""",
)

####################
# Platform Windows #
####################
# Boost Headers
new_local_repository(
    name = "boost_headers_windows",
    path = "C:\\boost",
    build_file_content = """
cc_library(
    name = "boost_header",
    hdrs = glob(["boost/**/*.hpp","boost/**/*.h"]),
    visibility = ["//visibility:public"],
)
""",
)

# PFFFT - Windows
http_archive(
    name = "pffft_lib_win",
    strip_prefix = "jpommier-pffft-29e4f76ac53b",
    urls = ["https://bitbucket.org/jpommier/pffft/get/29e4f76ac53b.zip"],
    sha256 = "bb10afba127904a0c6c553fa445082729b7d72373511bda1b12a5be0e03f318a",
    build_file_content = """
cc_library(
    name = "pffft_win",
    srcs = glob(["pffft.c"]),
    hdrs = glob(["pffft.h"]),
	copts = [
		"/D_USE_MATH_DEFINES",
		"/W0",
	],
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


load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "9f9fb8b2f0213989247c9d5c0e814a8451d18d7f",
    remote = "https://github.com/nelhage/rules_boost",
    shallow_since = "1570056263 -0700",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()
