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

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "similarity_result.h"
#include "test_utility.h"
#include "visqol_manager.h"

namespace Visqol {
namespace {

const double kIdenticalMinMoslqo = 4.5;

// Struct for holding ShortDurationTest test data.
struct ShortDurationTestData {
  const double expected_result;
  const Visqol::CommandLineArgs test_inputs;

  ShortDurationTestData(absl::string_view reference_file,
                        absl::string_view degraded_file, double expected_result)
      : expected_result{expected_result},
        test_inputs{CommandLineArgsHelper(reference_file, degraded_file)} {}
};

// Class definition necessary for Value-Parameterized Tests.
class ShortDurationTest
    : public ::testing::TestWithParam<ShortDurationTestData> {};

// Initialise the input paramaters.
INSTANTIATE_TEST_CASE_P(
    TestParams, ShortDurationTest,
    testing::Values(
        ShortDurationTestData("testdata/short_duration/"
                              "1_sample/guitar48_stereo_1_sample.wav",
                              "testdata/short_duration/"
                              "1_sample/guitar48_stereo_1_sample.wav",
                              0),
        ShortDurationTestData("testdata/short_duration/"
                              "10_sample/guitar48_stereo_10_sample.wav",
                              "testdata/short_duration/"
                              "10_sample/guitar48_stereo_10_sample.wav",
                              0),
        ShortDurationTestData("testdata/short_duration/"
                              "100_sample/guitar48_stereo_100_sample.wav",
                              "testdata/short_duration/"
                              "100_sample/guitar48_stereo_100_sample.wav",
                              0),
        ShortDurationTestData("testdata/short_duration/"
                              "1000_sample/guitar48_stereo_1000_sample.wav",
                              "testdata/short_duration/"
                              "1000_sample/guitar48_stereo_1000_sample.wav",
                              0),
        ShortDurationTestData("testdata/short_duration/"
                              "10000_sample/guitar48_stereo_10000_sample.wav",
                              "testdata/short_duration/"
                              "10000_sample/guitar48_stereo_10000_sample.wav",
                              0)));

// Assert the an error code of INVALID_ARGUMENT is returned for a range of
// inputs that are too short in duration.
TEST_P(ShortDurationTest, InvalidArgsTest) {
  // Build command line args.
  Visqol::VisqolManager visqol;
  auto files_to_compare =
      VisqolCommandLineParser::BuildFilePairPaths(GetParam().test_inputs);

  // Init ViSQOL.
  auto status =
      visqol.Init(GetParam().test_inputs.similarity_to_quality_mapper_model,
                  GetParam().test_inputs.use_speech_mode,
                  GetParam().test_inputs.use_unscaled_speech_mos_mapping,
                  GetParam().test_inputs.search_window_radius);
  ASSERT_TRUE(status.ok());

  // Run ViSQOL and assert failure.
  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_FALSE(status_or.ok());
  ASSERT_EQ(absl::StatusCode::kInvalidArgument, status_or.status().code());
}

// Test visqol with input that is 1 second long (which at a sample rate of
// 48kHz equals 48,000 samples). Expect to pass.
TEST(ShortDuration, 1_second) {
  // Build command line args.
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/short_duration/1_second/"
      "guitar48_stereo_1_sec.wav",
      "testdata/short_duration/1_second/"
      "guitar48_stereo_1_sec.wav");
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  // Init ViSQOL.
  Visqol::VisqolManager visqol;
  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius);
  ASSERT_TRUE(status.ok());

  // Run ViSQOL.
  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  ASSERT_TRUE(status_or.value().moslqo() > kIdenticalMinMoslqo);
}

// Test visqol with input that is 5 second long (which at a sample rate of
// 48kHz equals 240,000 samples). Expect to pass.
TEST(ShortDuration, 5_second) {
  // Build command line args.
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/short_duration/5_second/"
      "guitar48_stereo_5_sec.wav",
      "testdata/short_duration/5_second/"
      "guitar48_stereo_5_sec.wav");
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  // Init ViSQOL.
  Visqol::VisqolManager visqol;
  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius);
  ASSERT_TRUE(status.ok());

  // Run ViSQOL.
  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  ASSERT_TRUE(status_or.value().moslqo() > kIdenticalMinMoslqo);
}

}  // namespace
}  // namespace Visqol
