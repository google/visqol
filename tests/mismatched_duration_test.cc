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

#include "visqol_manager.h"
#include <iostream>
#include "gtest/gtest.h"

#include "similarity_result.h"
#include "test_utility.h"

namespace Visqol {
namespace {

const double kMinMoslqo = 1.0;
const double kConformanceGuitarX2MisMatch = 4.7321012530423481;
const double kConformanceGuitar2secMisMatch = 4.7262192026045007;
const double kConformanceGuitar50msMisMatch = 4.6672912779523177;
const double kTolerance = 0.0001;

// Test that when the degraded file is shorter than the reference (by more than
// 1 second) visqol will still run to completion without failure.
TEST(MismatchedLengths, deg_too_short) {
  // Build command line args.
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper
      ("testdata/mismatched_duration/guitar48_stereo_x2.wav",
       "testdata/conformance_testdata_subset/guitar48_stereo.wav");
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  // Init ViSQOL.
  Visqol::VisqolManager visqol;
  auto status = visqol.Init(cmd_args.sim_to_quality_mapper_model,
      cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping,
      cmd_args.search_window_radius);
  ASSERT_TRUE(status.ok());

  // Run ViSQOL.
  auto status_or = visqol.Run(files_to_compare[0].reference,
                              files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  ASSERT_TRUE(status_or.value().moslqo() > kMinMoslqo);
  EXPECT_NEAR(kConformanceGuitarX2MisMatch, status_or.value().moslqo(),
              kTolerance);
}

// Test that when the degraded file is longer than the reference (by more than
// 1 second) visqol will still run to completion without failure.
TEST(MismatchedLengths, deg_too_long) {
  // Build command line args.
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper
      ("testdata/mismatched_duration/guitar48_stereo_middle_2sec_cut.wav",
       "testdata/conformance_testdata_subset/guitar48_stereo.wav");
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  // Init ViSQOL.
  Visqol::VisqolManager visqol;
  auto status = visqol.Init(cmd_args.sim_to_quality_mapper_model,
      cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping,
      cmd_args.search_window_radius);
  ASSERT_TRUE(status.ok());

  // Run ViSQOL.
  auto status_or = visqol.Run(files_to_compare[0].reference,
                              files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  ASSERT_TRUE(status_or.value().moslqo() > kMinMoslqo);
  EXPECT_NEAR(kConformanceGuitar2secMisMatch, status_or.value().moslqo(),
              kTolerance);
}

// Test that when the degraded file is longer than the reference (by about 50 ms
// ) visqol will still run to completion without failure.
TEST(MismatchedLengths, deg_long) {
  // Build command line args.
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav",
      "testdata/mismatched_duration/"
      "guitar48_stereo_middle_50ms_cut.wav");
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  // Init ViSQOL.
  Visqol::VisqolManager visqol;
  auto status = visqol.Init(cmd_args.sim_to_quality_mapper_model,
                            cmd_args.use_speech_mode,
                            cmd_args.use_unscaled_speech_mos_mapping,
                            cmd_args.search_window_radius);
  ASSERT_TRUE(status.ok());

  // Run ViSQOL.
  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  ASSERT_TRUE(status_or.value().moslqo() > kMinMoslqo);
  EXPECT_NEAR(kConformanceGuitar50msMisMatch, status_or.value().moslqo(),
              kTolerance);
}

} // namespace
} // namespace Visqol
