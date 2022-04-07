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

#include "alignment.h"

#include "audio_signal.h"
#include "gtest/gtest.h"
#include "xcorr.h"

namespace Visqol {
namespace {

// The reference signal
const AMatrix<double> kReferenceSignal{std::valarray<double>{
    2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0}};

// A degraded signal with a 2 sample lag.
const AMatrix<double> kDegradedSignalLag2{std::valarray<double>{
    1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2, 2.1}};

// A degraded signal with a negative 2 sample lag.
const AMatrix<double> kDegradedSignalNegativeLag2{std::valarray<double>{
    2.0, 2.0, 2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0}};

// These lag values were calculated manually from the simple signals above.
constexpr int kBestLagPositive2 = 2;
constexpr int kBestLagNegative2 = -2;
constexpr int kZeroLag = 0;

// Test the alignment of a degraded signal with a given reference signal.
// Test case where the lag between the two signals has a positive value.
TEST(Alignment, AlignSignalWithPositiveLag) {
  AudioSignal reference_signal{kReferenceSignal, 1};
  AudioSignal degraded_signal{kDegradedSignalLag2, 1};

  const int64_t initial_lag = XCorr::FindLowestLagIndex(
      reference_signal.data_matrix, degraded_signal.data_matrix);
  // Ensure there is an positive initial lag.
  EXPECT_EQ(kBestLagPositive2, initial_lag);

  const std::tuple<AudioSignal, double> alignment_result =
      Alignment::GloballyAlign(reference_signal, degraded_signal);
  degraded_signal = std::get<0>(alignment_result);

  const int64_t final_lag = XCorr::FindLowestLagIndex(
      reference_signal.data_matrix, degraded_signal.data_matrix);
  // Confirm the lag is gone.
  EXPECT_EQ(kZeroLag, final_lag);

  // Confirm that the new degraded signal has been padded by the lag amount.
  EXPECT_EQ(reference_signal.data_matrix.NumElements() + kBestLagPositive2,
            degraded_signal.data_matrix.NumElements());
}

// Test the alignment of a degraded signal with a given reference signal.
// Test case where the lag between the two signals has a negative value.
TEST(Alignment, AlignSignalWithNegativeLag) {
  AudioSignal reference_signal{kReferenceSignal, 1};
  AudioSignal degraded_signal{kDegradedSignalNegativeLag2, 1};

  const int64_t initial_lag = XCorr::FindLowestLagIndex(
      reference_signal.data_matrix, degraded_signal.data_matrix);
  // Ensure there is an initial negative lag.
  EXPECT_EQ(kBestLagNegative2, initial_lag);

  const std::tuple<AudioSignal, double> alignment_result =
      Alignment::GloballyAlign(reference_signal, degraded_signal);
  degraded_signal = std::get<0>(alignment_result);

  const int64_t final_lag = XCorr::FindLowestLagIndex(
      reference_signal.data_matrix, degraded_signal.data_matrix);
  // Confirm the lag is gone.
  EXPECT_EQ(kZeroLag, final_lag);

  // Confirm that the new degraded signal has been cut by the lag amount.
  EXPECT_EQ(reference_signal.data_matrix.NumElements(),
            degraded_signal.data_matrix.NumElements() - kBestLagNegative2);
}

// Test the alignment of a degraded signal with a given reference signal.
// Test case where there is no lag between the two signals.
TEST(Alignment, AlignSignalWithNoLag) {
  AudioSignal reference_signal{kReferenceSignal, 1};
  AudioSignal degraded_signal{kReferenceSignal, 1};
  const int degraded_signal_init_size =
      degraded_signal.data_matrix.NumElements();

  const int64_t initial_lag = XCorr::FindLowestLagIndex(
      reference_signal.data_matrix, degraded_signal.data_matrix);
  // Ensure there is no initial lag.
  EXPECT_EQ(kZeroLag, initial_lag);

  const std::tuple<AudioSignal, double> alignment_result =
      Alignment::GloballyAlign(reference_signal, degraded_signal);
  degraded_signal = std::get<0>(alignment_result);

  const int64_t final_lag = XCorr::FindLowestLagIndex(
      reference_signal.data_matrix, degraded_signal.data_matrix);
  // Confirm there is still no lag.
  EXPECT_EQ(kZeroLag, final_lag);

  // Confirm that the degraded signal is the same size.
  EXPECT_EQ(degraded_signal_init_size,
            degraded_signal.data_matrix.NumElements());
}

// Test the alignment of a degraded signal with a given reference signal.
// Test case where the lag between the two signals has a negative value.
TEST(Alignment, AlignAndTruncateSignalWithNegativeLag) {
  AudioSignal reference_signal{kReferenceSignal, 1};
  AudioSignal degraded_signal{kDegradedSignalNegativeLag2, 1};

  const double original_reference_duration = reference_signal.GetDuration();
  const std::tuple<AudioSignal, AudioSignal, double> alignment_result =
      Alignment::AlignAndTruncate(reference_signal, degraded_signal);
  reference_signal = std::get<0>(alignment_result);
  degraded_signal = std::get<1>(alignment_result);
  const double lag = std::get<2>(alignment_result);

  // Ensure there is an initial negative lag.
  EXPECT_EQ(lag, kBestLagNegative2);

  // Confirm that the new degraded signal has been truncated to a common
  // duration.
  EXPECT_EQ(original_reference_duration + kBestLagNegative2,
            degraded_signal.GetDuration());
  EXPECT_EQ(original_reference_duration + kBestLagNegative2,
            reference_signal.GetDuration());
}

// Test the alignment of a degraded signal with a given reference signal.
// Test case where the lag between the two signals has a positive value.
TEST(Alignment, AlignAndTruncateSignalWithPositiveLag) {
  AudioSignal reference_signal{kReferenceSignal, 1};
  AudioSignal degraded_signal{kDegradedSignalLag2, 1};

  const double original_reference_duration = reference_signal.GetDuration();
  const std::tuple<AudioSignal, AudioSignal, double> alignment_result =
      Alignment::AlignAndTruncate(reference_signal, degraded_signal);
  reference_signal = std::get<0>(alignment_result);
  degraded_signal = std::get<1>(alignment_result);
  const double lag = std::get<2>(alignment_result);

  // Ensure there is an initial postive lag.
  EXPECT_EQ(lag, kBestLagPositive2);

  // Confirm that the new degraded signal has been truncated to only
  // the aligned overlap, without the zeros in front.
  EXPECT_EQ(original_reference_duration - kBestLagPositive2,
            degraded_signal.GetDuration());
  EXPECT_EQ(original_reference_duration - kBestLagPositive2,
            reference_signal.GetDuration());
}

}  // namespace
}  // namespace Visqol
