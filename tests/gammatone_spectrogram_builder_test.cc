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

#include "gammatone_spectrogram_builder.h"

#include "absl/memory/memory.h"
#include "amatrix.h"
#include "analysis_window.h"
#include "equivalent_rectangular_bandwidth.h"
#include "file_path.h"
#include "gammatone_filterbank.h"
#include "gtest/gtest.h"
#include "misc_audio.h"
#include "spectrogram.h"
#include "spectrogram_builder.h"

namespace Visqol {
namespace {

const double kMinimumFreq = 50;
const size_t kNumBands = 32;
const double kOverlap = 0.25;

// These col counts are calculated by dividing the total number of samples in
// signal, by the frame size (which is based on the window size, overlap and
// sample rate).
const size_t kRefSpectroNumCols = 802;
const size_t kDegSpectroNumCols = 807;

// Ensure that the spectrograms produced have the correct dimensions. Test with
// two signals to ensure that consecutive spectrograms can be produced without
// issue (the filter bank is shared between them).
TEST(BuildSpectrogramTest, basic_positive_flow) {
  FilePath stereo_file_ref{
      "testdata/conformance_testdata_subset/"
      "contrabassoon48_stereo.wav"};
  FilePath stereo_file_deg{
      "testdata/conformance_testdata_subset/"
      "contrabassoon48_stereo_24kbps_aac."
      "wav"};

  // Load the audio signals
  const AudioSignal signal_ref = MiscAudio::LoadAsMono(stereo_file_ref);
  const AudioSignal signal_deg = MiscAudio::LoadAsMono(stereo_file_deg);

  // Build the gammatone filter and window
  auto filter_bank = GammatoneFilterBank{kNumBands, kMinimumFreq};
  const AnalysisWindow window{signal_ref.sample_rate, kOverlap};

  // Create the spectrograms
  GammatoneSpectrogramBuilder spectroBuilder(filter_bank, false);
  Spectrogram spectrogram_ref =
      spectroBuilder.Build(signal_ref, window).value();
  Spectrogram spectrogram_deg =
      spectroBuilder.Build(signal_deg, window).value();

  ASSERT_EQ(kRefSpectroNumCols, spectrogram_ref.Data().NumCols());
  ASSERT_EQ(kNumBands, spectrogram_ref.Data().NumRows());
  ASSERT_EQ(kDegSpectroNumCols, spectrogram_deg.Data().NumCols());
  ASSERT_EQ(kNumBands, spectrogram_deg.Data().NumRows());
}

}  // namespace
}  // namespace Visqol
