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

#include "absl/flags/flag.h"
#include "commandline_parser.h"
#include "conformance.h"
#include "gtest/gtest.h"
#include "similarity_result.h"
#include "test_utility.h"

namespace Visqol {
namespace {

const double kTolerance = 0.0001;
const double kLagTolerance = 0.1;
const int kGuitarNumPatches = 20;
const double kFirstGuitarTimestamp = 0.28000;
const size_t kGuitarStartLagIndex = 12;
const double kLag = 0.05;  // 50ms
const double k10kCenterFreqBand = 10261.08660;
const size_t k10kCenterFreqBandIndex = 26;
const double kPerfectScore = 5.0;

/**
 *  Compare against the ground truth obtained from the KNOWN version
 */
TEST(RegressionTest, Mono) {
  const Visqol::CommandLineArgs cmd_args =
      CommandLineArgsHelper("testdata/clean_speech/CA01_01.wav",
                            "testdata/clean_speech/transcoded_CA01_01.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  EXPECT_NEAR(kCA01_01AsAudio, status_or.value().moslqo(), kTolerance);
}

/**
 *  Compare against the ground truth obtained from the KNOWN version
 */
TEST(RegressionTest, Stereo) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav",
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo_64kbps_aac.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  EXPECT_NEAR(kConformanceGuitar64aac, status_or.value().moslqo(), kTolerance);
}

/**
 * Pass an invalid model to VisqolManager and ensure an INVALID_ARGUMENT
 * status is returned.
 */
TEST(VisqolCommandLineTest, FailedInit) {
  Visqol::VisqolManager visqol;
  auto status =
      visqol.Init(FilePath("non/existent/file.txt"), false, false, 60, false);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(absl::StatusCode::kInvalidArgument, status.code());
}

/**
 * Run VisqolManager and without initialization and ensure an ABORTED
 * status is returned.
 */
TEST(VisqolCommandLineTest, MissingInit) {
  const Visqol::CommandLineArgs cmd_args =
      CommandLineArgsHelper("testdata/clean_speech/CA01_01.wav",
                            "testdata/clean_speech/transcoded_CA01_01.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  // Run withouth calling Init()
  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_FALSE(status_or.ok());
  ASSERT_EQ(absl::StatusCode::kAborted, status_or.status().code());
}

/**
 * Ensure that the ordering of the FVNSIM and center freq bands is correct.
 * Also ensure that the value of the lowest FVNSIM result is correct. Finally,
 * ensure that the ordering of the per-patch FVNSIM values match the ordering
 * of the overall FVNSIM results.
 *
 * For this test, frequencies around the 10.26kHz center frequency (which is
 * one of the center frequencies seen in a 48k sample rate file) were filtered
 * using the following command:
 *
 * sox guitar48_stereo.wav guitar48_stereo_10k_filtered_freqs.wav sinc 10.4k-10k
 *
 * This test asserts that the FVNSIM value at this center freq is the lowest.
 */
TEST(VisqolCommandLineTest, FilteredFreqs) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav",
      "testdata/filtered_freqs/"
      "guitar48_stereo_10k_filtered_freqs.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  SimilarityResultMsg sim_result_msg = status_or.value();
  auto fvnsim = sim_result_msg.fvnsim();
  auto fstdnsim = sim_result_msg.fstdnsim();
  auto cfb = sim_result_msg.center_freq_bands();
  ASSERT_EQ(fvnsim.size(), cfb.size());

  // Assert that the 10k freq band is at the right index.
  ASSERT_NEAR(k10kCenterFreqBand, cfb[k10kCenterFreqBandIndex], kTolerance);

  // Assert that the 10k freq band FVNSIM is the lowest value of the FVNSIMs.
  const double fvnsim_10k = fvnsim[k10kCenterFreqBandIndex];
  double lowest_fvnsim = 1.0;
  for (double each_fvnsim : fvnsim) {
    if (each_fvnsim < lowest_fvnsim) {
      lowest_fvnsim = each_fvnsim;
    }
  }
  ASSERT_NEAR(fvnsim_10k, lowest_fvnsim, kTolerance);

  ASSERT_GT(fstdnsim[k10kCenterFreqBandIndex], 0.0);

  // Assert that the ordering of the per-patch FVNSIMs matches the overall
  // FVNSIM ordering by testing the 10k band for both.
  auto per_patch_dbg = sim_result_msg.patch_sims();
  double fbm_10k = 0.0;
  for (const auto& per_patch_fbm : per_patch_dbg) {
    fbm_10k += per_patch_fbm.freq_band_means()[k10kCenterFreqBandIndex];
  }
  fbm_10k = fbm_10k / per_patch_dbg.size();
  ASSERT_NEAR(fvnsim_10k, fbm_10k, kTolerance);
}

/**
 * Test idential files and check that stddev is 0 and nsim is 1.
 */
TEST(VisqolCommandLineTest, IdenticalStddevNsim) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav",
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  SimilarityResultMsg sim_result_msg = status_or.value();
  auto fvnsim = sim_result_msg.fvnsim();
  auto fstdnsim = sim_result_msg.fstdnsim();
  auto fvdegenergy = sim_result_msg.fvdegenergy();

  for (double each_fvnsim : fvnsim) {
    ASSERT_EQ(each_fvnsim, 1.0);
  }
  for (double each_fstdnsim : fstdnsim) {
    ASSERT_EQ(each_fstdnsim, 0.0);
  }
  for (double each_fvdegenergy : fvdegenergy) {
    ASSERT_GT(each_fvdegenergy, 0.0);
  }
}

/**
 * Ensure that ViSQOL will complete successfully for input audio signals with
 * non-48k sample rates.
 */
TEST(VisqolCommandLineTest, Non48kSampleRate) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/non_48k_sample_rate/"
      "guitar48_stereo_44100Hz.wav",
      "testdata/non_48k_sample_rate/"
      "guitar48_stereo_44100Hz.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  // We're not interested in the MOS-LQO for this test, just that it completed
  // successfully with non-48k input.
  ASSERT_TRUE(status_or.ok());
}

/**
 * Ensure that ViSQOL will return an error for input audio signals with
 * sample rates that do not match each other.
 */
TEST(VisqolCommandLineTest, DifferentSampleRate) {
  // Ref 48k, Deg 44.1k
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav",
      "testdata/non_48k_sample_rate/"
      "guitar48_stereo_44100Hz.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_FALSE(status_or.ok());
}

/**
 * Test the debug output patch timestamps. Test with two identical files.
 */
TEST(VisqolCommandLineTest, PatchTimestampsIdenticalFiles) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav",
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  auto patch_sims = status_or.value().patch_sims();
  ASSERT_EQ(kGuitarNumPatches, patch_sims.size());
  for (size_t i = 0; i < kGuitarNumPatches; i++) {
    if (i == 0) {
      ASSERT_NEAR(kFirstGuitarTimestamp, patch_sims[i].ref_patch_start_time(),
                  kTolerance);
    } else {
      ASSERT_NEAR(patch_sims[i - 1].ref_patch_end_time(),
                  patch_sims[i].ref_patch_start_time(), kTolerance);
    }

    ASSERT_NEAR(patch_sims[i].ref_patch_start_time(),
                patch_sims[i].deg_patch_start_time(), kTolerance);
    ASSERT_NEAR(patch_sims[i].ref_patch_end_time(),
                patch_sims[i].deg_patch_end_time(), kTolerance);
  }
}

/**
 * Test the debug output patch timestamps. Test with two files that are
 * identical, except for a 50ms cut taken out of the middle of the degraded
 * signal. This test shows that the degraded patch timestamps will correctly
 * show this 50ms lag.
 */
TEST(VisqolCommandLineTest, PatchTimestampsMissing50ms) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav",
      "testdata/mismatched_duration/"
      "guitar48_stereo_middle_50ms_cut.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  auto patch_sims = status_or.value().patch_sims();
  ASSERT_EQ(kGuitarNumPatches, patch_sims.size());

  // Test the non-lagged patches
  for (size_t i = 0; i < kGuitarStartLagIndex; i++) {
    if (i == 0) {
      ASSERT_NEAR(kFirstGuitarTimestamp, patch_sims[i].ref_patch_start_time(),
                  kTolerance);
    } else {
      ASSERT_NEAR(patch_sims[i - 1].ref_patch_end_time(),
                  patch_sims[i].ref_patch_start_time(), kLagTolerance);
    }

    ASSERT_NEAR(patch_sims[i].ref_patch_start_time(),
                patch_sims[i].deg_patch_start_time(), kLagTolerance);
    ASSERT_NEAR(patch_sims[i].ref_patch_end_time(),
                patch_sims[i].deg_patch_end_time(), kLagTolerance);
  }

  // Test the lagged patches
  for (size_t i = kGuitarStartLagIndex; i < kGuitarNumPatches; i++) {
    ASSERT_NEAR(patch_sims[i - 1].ref_patch_end_time(),
                patch_sims[i].ref_patch_start_time(), kLagTolerance);
    ASSERT_NEAR(patch_sims[i].ref_patch_start_time(),
                patch_sims[i].deg_patch_start_time() + kLag, kLagTolerance);
    ASSERT_NEAR(patch_sims[i].ref_patch_end_time(),
                patch_sims[i].deg_patch_end_time() + kLag, kLagTolerance);
  }
}

/**
 * Test that running ViSQOL with speech mode disabled (even with the 'use
 * unscaled mapping' bool set to true), the input files will be compared as
 * audio.
 */
TEST(VisqolCommandLineTest, SpeechModeDisabled) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/clean_speech/CA01_01.wav",
      "testdata/clean_speech/transcoded_CA01_01.wav", "", false, true);
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  EXPECT_NEAR(kCA01_01AsAudio, status_or.value().moslqo(), kTolerance);
}

/**
 * Test ViSQOL running in speech mode. Use the same file for both the reference
 * and degraded signals and run in scaled mode. A perfect score of 5.0 is
 * expected.
 */
TEST(VisqolCommandLineTest, ScaledSpeechMode) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/clean_speech/CA01_01.wav", "testdata/clean_speech/CA01_01.wav",
      "", true, false, 60, false);
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  EXPECT_NEAR(kPerfectScore, status_or.value().moslqo(), kTolerance);
}

/**
 * Test ViSQOL running in speech mode. Use the same file for both the reference
 * and degraded signals and run in unscaled mode. A score of 4.x is expected.
 */
TEST(VisqolCommandLineTest, UnscaledSpeechMode) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/clean_speech/CA01_01.wav", "testdata/clean_speech/CA01_01.wav",
      "", true, true, 60, false);
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  EXPECT_NEAR(kConformanceUnscaledPerfectScoreExponential,
              status_or.value().moslqo(), kTolerance);
}

/**
 * Test idential files has lag of 0.
 */
TEST(VisqolCommandLineTest, ZeroLagOnidenticalFiles) {
  const Visqol::CommandLineArgs cmd_args = CommandLineArgsHelper(
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav",
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav");
  Visqol::VisqolManager visqol;
  auto files_to_compare = VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  auto status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  ASSERT_TRUE(status.ok());

  auto status_or =
      visqol.Run(files_to_compare[0].reference, files_to_compare[0].degraded);
  ASSERT_TRUE(status_or.ok());
  SimilarityResultMsg sim_result_msg = status_or.value();
  EXPECT_NEAR(0.0, status_or.value().alignment_lag_s(), kLagTolerance);
}

}  // namespace
}  // namespace Visqol
