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
    strip_prefix = "armadillo-code-9.200.x",
    urls = ["https://gitlab.com/conradsnicta/armadillo-code/-/archive/9.200.x/armadillo-code-9.200.x.zip"],
    sha256 = "df078186a5bcc66e8621139451c751c8b54d4a899ae38470fcf0f37bfd6bb5f0",
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
