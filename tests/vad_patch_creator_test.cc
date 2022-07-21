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

#include "vad_patch_creator.h"

#include <memory>
#include <random>

#include "absl/memory/memory.h"
#include "analysis_window.h"
#include "file_path.h"
#include "gammatone_spectrogram_builder.h"
#include "gtest/gtest.h"
#include "misc_audio.h"

namespace Visqol {

const std::vector<size_t> kCA01_01Patches{9, 29, 49, 69, 89};
const double kMinimumFreq = 50;
const size_t kPatchSize = 20;
const size_t kNumBands = 21;
const size_t kStartSample = 14;
const size_t kTotalSample = 115200;
const size_t kFrameLen = 480;
const size_t kCA01_01VadResCount = 240;

/**
 * Test that the correct count of VAD results are returned for the CA01_01.wav
 * file.
 */
TEST(VadPatchCreatorTest, CleanSpeechVAD) {
  // Test that the VAD accepts clean speech.
  // File containing speech ("The birch canoe slid on the smooth planks")
  const char* ref_file = "testdata/clean_speech/CA01_01.wav";

  AudioSignal ref_signal = MiscAudio::LoadAsMono(FilePath(ref_file));
  VadPatchCreator vad(kPatchSize);
  ASSERT_EQ(kCA01_01VadResCount, vad.GetVoiceActivity(ref_signal, kStartSample,
                                                      kTotalSample, kFrameLen)
                                     .size());
}

/**
 * Test that the VAD Patch Creator will return the correct patch indices.
 */
TEST(VadPatchCreatorTest, PatchIndices) {
  // Load the signal.
  const char* ref_file = "testdata/clean_speech/CA01_01.wav";
  AudioSignal ref_signal = MiscAudio::LoadAsMono(FilePath(ref_file));

  // Create the analysis window.
  const AnalysisWindow window{ref_signal.sample_rate, .25, .08};

  // Create the spectrogram.
  std::unique_ptr<SpectrogramBuilder> spectro_builder =
      absl::make_unique<GammatoneSpectrogramBuilder>(
          GammatoneFilterBank{kNumBands, kMinimumFreq}, true);
  const auto spectro_result = spectro_builder->Build(ref_signal, window);
  ASSERT_TRUE(spectro_result.ok());
  const auto spectro = spectro_result.value();

  // Create the reference signal patch indices.
  VadPatchCreator vad(kPatchSize);
  const auto patches_result =
      vad.CreateRefPatchIndices(spectro.Data(), ref_signal, window);
  ASSERT_TRUE(patches_result.ok());
  const auto patches = patches_result.value();
  ASSERT_TRUE(kCA01_01Patches == patches);
}

}  // namespace Visqol
