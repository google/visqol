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

#include "analysis_window.h"

#include <math.h>

#include "gtest/gtest.h"

/**
 * Calculate the value of the temporal window for the given sample rate and
 * window size.
 *
 * @param sample_rate The sample rate of the input signal.
 * @param win_size The size of the window in samples.
 *
 * @return The duration of time captured in this window, in ms.
 */
inline int CalcTemporalWindow(const int sample_rate, const size_t win_size) {
  return round((1000.0 / sample_rate) * win_size);
}

namespace Visqol {
namespace {

const int kSampleRate8000 = 8000;
const int kSampleRate16000 = 16000;
const int kSampleRate22050 = 22050;
const int kSampleRate44100 = 44100;
const int kSampleRate48000 = 48000;
const int kSampleRate96000 = 96000;
const int kTemporalWindow = 80;  // 80ms
const double kOverlap = 0.25;

/**
 * Ensure that the window size is temporally consistent regardless of the
 * sample rate.
 */
TEST(AnalysisWindow, calc_window_size) {
  // 8khz Sample Rate.
  size_t window_size = AnalysisWindow{kSampleRate8000, kOverlap}.size;
  ASSERT_EQ(CalcTemporalWindow(kSampleRate8000, window_size), kTemporalWindow);
  // 16khz Sample Rate.
  window_size = AnalysisWindow{kSampleRate16000, kOverlap}.size;
  ASSERT_EQ(CalcTemporalWindow(kSampleRate16000, window_size), kTemporalWindow);
  // 22.05khz Sample Rate.
  window_size = AnalysisWindow{kSampleRate22050, kOverlap}.size;
  ASSERT_EQ(CalcTemporalWindow(kSampleRate22050, window_size), kTemporalWindow);
  // 44.1khz Sample Rate.
  window_size = AnalysisWindow{kSampleRate44100, kOverlap}.size;
  ASSERT_EQ(CalcTemporalWindow(kSampleRate44100, window_size), kTemporalWindow);
  // 48khz Sample Rate.
  window_size = AnalysisWindow{kSampleRate48000, kOverlap}.size;
  ASSERT_EQ(CalcTemporalWindow(kSampleRate48000, window_size), kTemporalWindow);
  // 96khz Sample Rate.
  window_size = AnalysisWindow{kSampleRate96000, kOverlap}.size;
  ASSERT_EQ(CalcTemporalWindow(kSampleRate96000, window_size), kTemporalWindow);
}
}  // namespace
}  // namespace Visqol
