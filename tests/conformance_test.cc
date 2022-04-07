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

#include "conformance.h"

#include <vector>

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "similarity_result.h"
#include "test_utility.h"
#include "visqol_manager.h"

namespace Visqol {
namespace {

const double kTolerance = 0.0001;

// Struct for holding input args and expected output MOSLQO.
struct ConformanceTestData {
  const double expected_result;
  const Visqol::CommandLineArgs test_inputs;

  ConformanceTestData(absl::string_view reference_file,
                      absl::string_view degraded_file, bool speech_mode,
                      double exp_result, bool use_lattice,
                      bool unscaled_speech = false)
      : expected_result{exp_result},
        test_inputs{CommandLineArgsHelper(reference_file, degraded_file, "",
                                          speech_mode, unscaled_speech, 60,
                                          use_lattice)} {}
};

// Class definition necessary for Value-Parameterized Tests.
class ConformanceTest : public ::testing::TestWithParam<ConformanceTestData> {};

// Assert that the MOSLQO returned matches the last known version.
TEST_P(ConformanceTest, ConformanceWithKnownScores) {
  Visqol::VisqolManager visqol;
  auto files_to_compare =
      VisqolCommandLineParser::BuildFilePairPaths(GetParam().test_inputs);

  auto status =
      visqol.Init(GetParam().test_inputs.similarity_to_quality_mapper_model,
                  GetParam().test_inputs.use_speech_mode,
                  GetParam().test_inputs.use_unscaled_speech_mos_mapping,
                  GetParam().test_inputs.search_window_radius,
                  GetParam().test_inputs.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  EXPECT_NEAR(GetParam().expected_result, status_or.value().moslqo(),
              kTolerance);
}

// Initialise the input paramaters.
INSTANTIATE_TEST_CASE_P(
    TestParams, ConformanceTest,
    testing::Values(
        ConformanceTestData("testdata/clean_speech/CA01_01.wav",
                            "testdata/clean_speech/transcoded_CA01_01.wav",
                            true, kConformanceSpeechCA01TranscodedLattice,
                            true),
        ConformanceTestData("testdata/clean_speech/CA01_01.wav",
                            "testdata/clean_speech/transcoded_CA01_01.wav",
                            true, kConformanceSpeechCA01TranscodedExponential,
                            false),
        ConformanceTestData("testdata/clean_speech/CA01_01.wav",
                            "testdata/clean_speech/CA01_01.wav", true,
                            kConformanceCA01PerfectScoreLattice, true),
        ConformanceTestData("testdata/clean_speech/CA01_01.wav",
                            "testdata/clean_speech/CA01_01.wav", true,
                            kConformanceUnscaledPerfectScoreExponential, false,
                            true),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "strauss48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "strauss48_stereo_lp35.wav",
                            false, kConformanceStraussLp35, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "steely48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "steely48_stereo_lp7.wav",
                            false, kConformanceSteelyLp7, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "sopr48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "sopr48_stereo_256kbps_aac.wav",
                            false, kConformanceSopr256aac, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "ravel48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "ravel48_stereo_128kbps_opus.wav",
                            false, kConformanceRavel128opus, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "moonlight48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "moonlight48_stereo_128kbps_aac.wav",
                            false, kConformanceMoonlight128aac, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "harpsichord48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "harpsichord48_stereo_96kbps_mp3.wav",
                            false, kConformanceHarpsichord96mp3, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "guitar48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "guitar48_stereo_64kbps_aac.wav",
                            false, kConformanceGuitar64aac, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "glock48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "glock48_stereo_48kbps_aac.wav",
                            false, kConformanceGlock48aac, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "contrabassoon48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "contrabassoon48_stereo_24kbps_aac.wav",
                            false, kConformanceContrabassoon24aac, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "castanets48_stereo.wav",
                            "testdata/conformance_testdata_subset/"
                            "castanets48_stereo.wav",
                            false, kConformanceCastanetsIdentity, false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "guitar48_stereo.wav",
                            "testdata/short_duration/5_second/"
                            "guitar48_stereo_5_sec.wav",
                            false, kConformanceGuitarShortDegradedPatch, false),
        ConformanceTestData("testdata/short_duration/5_second/"
                            "guitar48_stereo_5_sec.wav",
                            "testdata/conformance_testdata_subset/"
                            "guitar48_stereo.wav",
                            false, kConformanceGuitarShortReferencePatch,
                            false),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "guitar48_stereo.wav",
                            "testdata/clean_speech/CA01_01.wav", true,
                            kConformanceDifferentAudiosLattice, true),
        ConformanceTestData("testdata/conformance_testdata_subset/"
                            "guitar48_stereo.wav",
                            "testdata/clean_speech/CA01_01.wav", true,
                            kConformanceDifferentAudiosExponential, false),
        ConformanceTestData("testdata/alignment/reference.wav",
                            "testdata/alignment/degraded.wav", true,
                            kConformanceBadDegradedLattice, true),
        ConformanceTestData("testdata/alignment/reference.wav",
                            "testdata/alignment/degraded.wav", true,
                            kConformanceBadDegradedExponential, false)));
}  // namespace
}  // namespace Visqol
