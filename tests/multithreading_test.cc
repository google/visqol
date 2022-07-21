// Copyright 2019 Google LLC, Andrew Hines
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <thread>  // NOLINT(build/c++11)

#include "absl/flags/flag.h"
#include "conformance.h"
#include "gtest/gtest.h"
#include "similarity_result.h"
#include "test_utility.h"
#include "visqol_manager.h"

namespace Visqol {
namespace {

const double kTolerance = .00001;

const FilePath kDefaultModel =
    FilePath(FilePath::currentWorkingDir() + "/model/libsvm_nu_svr_model.txt");
const FilePath kTestModel = FilePath(FilePath::currentWorkingDir() +
                                     "/testdata/test_model/cpp_model.txt");

const double kGuitarMoslqoNewModel = 4.7776205494442028;

void thread_glock_test() {
  // Build command line args.
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "glock48_stereo.wav",
      "testdata/conformance_testdata_subset/"
      "glock48_stereo_48kbps_aac.wav");
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  // Init ViSQOL.
  Visqol::VisqolManager visqol;
  auto status = visqol.Init(cmd_args.similarity_to_quality_mapper_model, false,
                            false, 60);
  ASSERT_TRUE(status.ok());

  // Run ViSQOL.
  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  ASSERT_NEAR(kConformanceGlock48aac, status_or.value().moslqo(), kTolerance);
}

void thread_guitar_test(const FilePath model, const double moslqo) {
  // Build command line args.
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav",
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo_64kbps_aac.wav");
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  // Init ViSQOL.
  Visqol::VisqolManager visqol;
  auto status = visqol.Init(model, false, false, 60);
  ASSERT_TRUE(status.ok());

  // Run ViSQOL.
  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  ASSERT_NEAR(moslqo, status_or.value().moslqo(), kTolerance);
}

// Run two visqol tests simultaneously with the same input files for each run
// and with the same model file.
TEST(MultithreadingTest, SameInputSameModel) {
  std::thread thread_1(thread_glock_test);
  std::thread thread_2(thread_glock_test);
  thread_1.join();
  thread_2.join();
}

// Run two visqol tests simultaneously with the same input files for each run
// but with different model files.
TEST(MultithreadingTest, SameInputDiffModel) {
  std::thread thread_1(thread_guitar_test, kDefaultModel,
                       kConformanceGuitar64aac);
  std::thread thread_2(thread_guitar_test, kTestModel, kGuitarMoslqoNewModel);
  thread_1.join();
  thread_2.join();
}

// Run two visqol tests simultaneously with different input files for each run.
TEST(MultithreadingTest, DifferentInput) {
  std::thread thread_1(thread_glock_test);
  std::thread thread_2(thread_guitar_test, kDefaultModel,
                       kConformanceGuitar64aac);
  thread_1.join();
  thread_2.join();
}

}  // namespace
}  // namespace Visqol
