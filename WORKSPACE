########################
# Platform Independent #
########################
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

# The order of importing the archives is very important here.  In particular, tensorflow's
# workspace import function imports a version of pybind11 and protobuf that are older and
# not compatible with current pybind11_protobuf/abseil.  Since the first import is the only
# one bazel uses, import the ones we need before the tensorflow workspace functions.
# Unfortunately, this means we will have a warning about a double import.  We can't simply
# use the tf workspace functions because many of them are private (start with underscore),
# so this is a workaround until we submit a PR to make them public or modifiable (e.g. a
# flag for TFLite-only deps, suchas workspace1(only_tflite_deps=True))

## `pybind11_bazel`
# See https://github.com/pybind/pybind11_bazel
http_archive(
    name = "pybind11_bazel",
    patch_args = ["-p1"],
    # TODO(b/228236234): Remove patch after merging https://github.com/pybind/pybind11_bazel/pull/38
    patches = ["//:pybind11_fixdistutils.patch"],
    strip_prefix = "pybind11_bazel-72cbbf1fbc830e487e3012862b7b720001b70672",
    sha256 = "516c1b3a10d87740d2b7de6f121f8e19dde2c372ecbfe59aef44cd1872c10395",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.tar.gz"],
)

# We still require the pybind library.
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-2.9.2",
    sha256 = "6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1",
    urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz"],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

git_repository(
    name = "pybind11_abseil",
    remote = "https://github.com/mchinen/pybind11_abseil.git",
    commit = "a0c36ca08d894b5a138dff31a9057a7dcacfb8fc",
    shallow_since = "1649119381 -0700",
)

git_repository(
    name = "pybind11_protobuf",
    remote = "https://github.com/pybind/pybind11_protobuf.git",
    commit = "83f055cc82d983b7d5c3ce3f59ec034ba546d094",
    shallow_since = "1647996190 -0700",
)

# For protobuf rule:
http_archive(
    name = "six",
    build_file = "@com_google_protobuf//:third_party/six.BUILD",
    sha256 = "d16a0141ec1a18405cd4ce8b4613101da75da0e9a7aec5bdd4fa804d0e0eba73",
    urls = ["https://pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz"],
)

# proto_library, cc_proto_library, and java_proto_library rules implicitly
# depend on @com_google_protobuf for protoc and proto runtimes.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    sha256 = "87407cd28e7a9c95d9f61a098a53cf031109d451a7763e7dd1253abf8b4df422",
    strip_prefix = "protobuf-3.19.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.19.1.tar.gz"],
)

git_repository(
    name = "org_tensorflow",
    remote = "https://github.com/tensorflow/tensorflow.git",
    # Below is reproducible and equivalent to `tag = "v2.8.0"`
    commit = "3f878cff5b698b82eea85db2b60d65a2e320850e",
    shallow_since = "1643656653 -0800" ,
)

# Import all of TensorFlow Serving's external dependencies.
# Downstream projects (projects importing TensorFlow Serving) need to
# duplicate all code below in their WORKSPACE file in order to also initialize
# those external dependencies.
http_archive(
    name = "rules_pkg",
    sha256 = "352c090cc3d3f9a6b4e676cf42a6047c16824959b438895a76c2989c6d7c246a",
    url = "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz",
)

# Check bazel version requirement, which is stricter than TensorFlow's.
load(
    "@org_tensorflow//tensorflow:version_check.bzl",
    "check_bazel_version_at_least",
)

check_bazel_version_at_least("3.7.2")

# Begin TF WORKSPACE Loading
# This section uses a subset of the tensorflow WORKSPACE loading by reusing its contents.
# TF's loading is very complicated, and we only need a subset for TFLite.
# If we use the full TF loading sequence, we also run into conflicts and errors on some platforms.

load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")
workspace()

load("@org_tensorflow//tensorflow:workspace2.bzl", workspace2 = "workspace")
workspace2()

# Custom import/workarounds for workspace1.bzl.
# Load the grpc deps which resolve the protoc error: "every rule of type _generate_pb2_src implicitly depends upon the target '//external:protocol_compiler', but this target could not be found because of: no such target"
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

# There are cc_libary rule errors without @rules_cc.
git_repository(
    name = "rules_cc",
    commit = "40548a2974f1aea06215272d9c2b47a14a24e556",
    remote = "https://github.com/bazelbuild/rules_cc.git",
    shallow_since = "1612528209 -0800"
)

# End TF WORKSPACE Loading

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# GoogleTest/GoogleMock framework.
git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.10.0",
)

# Google Abseil Libs
git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp.git",
    tag = "20211102",
)

# LIBSVM
http_archive(
    name = "svm_lib",
    build_file_content = """
cc_library(
    name = "libsvm",
    srcs = glob(["svm.cpp"]),
    hdrs = glob(["svm.h"]),
    visibility = ["//visibility:public"],
)
""",
    sha256 = "401a60bd828bce8870b9eebf5023602028c7751b0db928a6a3bc351560b8b618",
    strip_prefix = "libsvm-324",
    # Depending on a recent stable version.
    urls = ["https://github.com/cjlin1/libsvm/archive/v324.zip"],
)

# Armadillo Headers
http_archive(
    name = "armadillo_headers",
    build_file_content = """
cc_library(
    name = "armadillo_header",
    hdrs = glob(["include/armadillo", "include/armadillo_bits/*.hpp"]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "d856ea58c18998997bcae6689784d2d3eeb5daf1379d569fddc277fe046a996b",
    strip_prefix = "armadillo-9.860.2",
    urls = ["http://sourceforge.net/projects/arma/files/armadillo-9.860.2.tar.xz"],
)

# PFFFT
new_git_repository(
    name = "pffft_lib",
    commit = "7c3b5a7dc510a0f513b9c5b6dc5b56f7aeeda422",
    shallow_since = "1644946905 +0000",
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
    remote = "https://bitbucket.org/jpommier/pffft.git",
)

bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)
