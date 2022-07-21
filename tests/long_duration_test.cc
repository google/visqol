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

#include <iostream>

#include "conformance.h"
#include "gtest/gtest.h"
#include "similarity_result.h"
#include "test_utility.h"
#include "visqol_manager.h"

namespace Visqol {
namespace {

const double kMOSGuitarLongDuration = 4.0;

const double kMinMoslqo = 1.0;
// This tolerance is set very wide because this is not a score conformance test.
const double kTolerance = 1.0;

// Confirm that a run can succesfully complete with a long file. In this test,
// file duration is 1 min.
TEST(LongFiles, 1_min) {
  // Build command line args.
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/long_duration/1_min/"
      "guitar48_stereo_ref_25s.wav",
      "testdata/long_duration/1_min/"
      "guitar48_stereo_deg_25s.wav",
      "", false);
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
  ASSERT_TRUE(status_or.value().moslqo() > kMinMoslqo);
  EXPECT_NEAR(kMOSGuitarLongDuration, status_or.value().moslqo(), kTolerance);
}

}  // namespace
}  // namespace Visqol
