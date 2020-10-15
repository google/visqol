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

#include <vector>

#include "gtest/gtest.h"

#include "conformance.h"
#include "similarity_result.h"
#include "test_utility.h"

namespace Visqol {
namespace {

const double kTolerance = 0.0001;

// Struct for holding input args and expected output MOSLQO.
struct ConformanceTestData{
  const double expected_result;
  const Visqol::CommandLineArgs test_inputs;

  ConformanceTestData(const std::string &ref_file, const std::string &deg_file,
                      const bool speech_mode, const double exp_result)
      : expected_result{exp_result},
        test_inputs{
            CommandLineArgsHelper(ref_file, deg_file, "", speech_mode)} {}
};

// Class definition necessary for Value-Parameterized Tests.
class ConformanceTest : public ::testing::TestWithParam<ConformanceTestData> {};

// Assert that the MOSLQO returned matches the last known version.
TEST_P(ConformanceTest, ConformanceWithKnownScores) {
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(
      GetParam().test_inputs);

  auto status = visqol.Init(GetParam().test_inputs.sim_to_quality_mapper_model,
      GetParam().test_inputs.use_speech_mode,
      GetParam().test_inputs.use_unscaled_speech_mos_mapping,
      GetParam().test_inputs.search_window_radius);
  ASSERT_TRUE(status.ok());

  auto status_or = visqol.Run(files_to_compare[0].reference,
                              files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  EXPECT_NEAR(GetParam().expected_result, status_or.value().moslqo(),
              kTolerance);
}

// Initialise the input paramaters.
INSTANTIATE_TEST_CASE_P(
    TestParams, ConformanceTest,
    testing::Values(
        ConformanceTestData(
            "testdata/clean_speech/CA01_01.wav",
            "testdata/clean_speech/transcoded_CA01_01.wav",
            true, kConformanceSpeechCA01Transcoded),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "strauss48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "strauss48_stereo_lp35.wav",
            false, kConformanceStraussLp35),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "steely48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "steely48_stereo_lp7.wav",
            false, kConformanceSteelyLp7),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "sopr48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "sopr48_stereo_256kbps_aac.wav",
            false, kConformanceSopr256aac),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "ravel48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "ravel48_stereo_128kbps_opus.wav",
            false, kConformanceRavel128opus),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "moonlight48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "moonlight48_stereo_128kbps_aac.wav",
            false, kConformanceMoonlight128aac),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "harpsichord48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "harpsichord48_stereo_96kbps_mp3.wav",
            false, kConformanceHarpsichord96mp3),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "guitar48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "guitar48_stereo_64kbps_aac.wav",
            false, kConformanceGuitar64aac),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "glock48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "glock48_stereo_48kbps_aac.wav",
            false, kConformanceGlock48aac),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "contrabassoon48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "contrabassoon48_stereo_24kbps_aac.wav",
            false, kConformanceContrabassoon24aac),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "castanets48_stereo.wav",
            "testdata/conformance_testdata_subset/"
            "castanets48_stereo.wav",
            false, kConformanceCastanetsIdentity),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "guitar48_stereo.wav",
            "testdata/short_duration/5_second/"
            "guitar48_stereo_5_sec.wav",
            false, kConformanceGuitarShortDegradedPatch),
        ConformanceTestData(
            "testdata/short_duration/5_second/"
            "guitar48_stereo_5_sec.wav",
            "testdata/conformance_testdata_subset/"
            "guitar48_stereo.wav",
            false, kConformanceGuitarShortReferencePatch),
        ConformanceTestData(
            "testdata/conformance_testdata_subset/"
            "guitar48_stereo.wav",
            "testdata/clean_speech/CA01_01.wav", true,
            kConformanceDifferentAudios),
        ConformanceTestData(
            "testdata/alignment/reference.wav",
            "testdata/alignment/degraded.wav", true,
            kConformanceBadDegraded)));
} // namespace
} // namespace Visqol
