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

#include "gtest/gtest.h"

#include "audio_signal.h"
#include "xcorr.h"

namespace Visqol {
namespace {

// The reference signal
const AMatrix<double> kRefSignal{std::valarray<double>{
  2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0
}};

// A degraded signal with a 2 sample lag.
const AMatrix<double> kDegSignalLag2{std::valarray<double>{
  1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2, 2.1
}};

// A degraded signal with a negative 2 sample lag.
const AMatrix<double> kDegSignalNegativeLag2{std::valarray<double>{
  2.0, 2.0, 2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0
}};

// These lag values were calculated manually from the simple signals above.
const long kBestLagPositive2 = 2;
const long kBestLagNegative2 = -2;
const long kZeroLag = 0;

// Test the alignment of a degraded signal with a given reference signal.
// Test case where the lag between the two signals has a positive value.
TEST(Alignment, AlignSignalWithPositiveLag) {
  AudioSignal ref_signal{kRefSignal, 1};
  AudioSignal deg_signal{kDegSignalLag2, 1};

  auto initial_lag = XCorr::CalcBestLag(ref_signal.data_matrix,
                                        deg_signal.data_matrix);
  // Ensure there is an positive initial lag.
  ASSERT_EQ(kBestLagPositive2, initial_lag);

  auto alignment_result = Alignment::GloballyAlign(ref_signal, deg_signal);
  deg_signal = std::get<0>(alignment_result);

  auto final_lag = XCorr::CalcBestLag(ref_signal.data_matrix,
                                      deg_signal.data_matrix);
  // Confirm the lag is gone.
  ASSERT_EQ(kZeroLag, final_lag);

  // Confirm that the new degraded signal has been padded by the lag amount.
  ASSERT_EQ(ref_signal.data_matrix.NumElements()+kBestLagPositive2,
            deg_signal.data_matrix.NumElements());
}

// Test the alignment of a degraded signal with a given reference signal.
// Test case where the lag between the two signals has a negative value.
TEST(Alignment, AlignSignalWithNegativeLag) {
  AudioSignal ref_signal{kRefSignal, 1};
  AudioSignal deg_signal{kDegSignalNegativeLag2, 1};

  auto initial_lag = XCorr::CalcBestLag(ref_signal.data_matrix,
                                        deg_signal.data_matrix);
  // Ensure there is an initial negative lag.
  ASSERT_EQ(kBestLagNegative2, initial_lag);

  auto alignment_result = Alignment::GloballyAlign(ref_signal, deg_signal);
  deg_signal = std::get<0>(alignment_result);

  auto final_lag = XCorr::CalcBestLag(ref_signal.data_matrix,                                        deg_signal.data_matrix);
  // Confirm the lag is gone.
  ASSERT_EQ(kZeroLag, final_lag);

  // Confirm that the new degraded signal has been cut by the lag amount.
  ASSERT_EQ(ref_signal.data_matrix.NumElements(),
            deg_signal.data_matrix.NumElements()-kBestLagNegative2);
}

// Test the alignment of a degraded signal with a given reference signal.
// Test case where there is no lag between the two signals.
TEST(Alignment, AlignSignalWithNoLag) {
  AudioSignal ref_signal{kRefSignal, 1};
  AudioSignal deg_signal{kRefSignal, 1};
  const auto deg_signal_init_size = deg_signal.data_matrix.NumElements();

  auto initial_lag = XCorr::CalcBestLag(ref_signal.data_matrix,
                                        deg_signal.data_matrix);
  // Ensure there is no initial lag.
  ASSERT_EQ(kZeroLag, initial_lag);

  auto alignment_result = Alignment::GloballyAlign(ref_signal, deg_signal);
  deg_signal = std::get<0>(alignment_result);

  auto final_lag = XCorr::CalcBestLag(ref_signal.data_matrix,
                                      deg_signal.data_matrix);
  // Confirm there is still no lag.
  ASSERT_EQ(kZeroLag, final_lag);

  // Confirm that the degraded signal is the same size.
  ASSERT_EQ(deg_signal_init_size, deg_signal.data_matrix.NumElements());
}

// Test the alignment of a degraded signal with a given reference signal.
// Test case where the lag between the two signals has a negative value.
TEST(Alignment, AlignAndTruncateSignalWithNegativeLag) {
  AudioSignal ref_signal{kRefSignal, 1};
  AudioSignal deg_signal{kDegSignalNegativeLag2, 1};

  double original_ref_duration = ref_signal.GetDuration();
  auto alignment_result = Alignment::AlignAndTruncate(ref_signal, deg_signal);
  ref_signal = std::get<0>(alignment_result);
  deg_signal = std::get<1>(alignment_result);
  double lag = std::get<2>(alignment_result);

  // Ensure there is an initial negative lag.
  ASSERT_EQ(lag, kBestLagNegative2);

  // Confirm that the new degraded signal has been truncated to a common
  // duration.
  ASSERT_EQ(original_ref_duration + kBestLagNegative2,
            deg_signal.GetDuration());
  ASSERT_EQ(original_ref_duration + kBestLagNegative2,
            ref_signal.GetDuration());
}

// Test the alignment of a degraded signal with a given reference signal.
// Test case where the lag between the two signals has a positive value.
TEST(Alignment, AlignAndTruncateSignalWithPositiveLag) {
  AudioSignal ref_signal{kRefSignal, 1};
  AudioSignal deg_signal{kDegSignalLag2, 1};

  double original_ref_duration = ref_signal.GetDuration();
  auto alignment_result = Alignment::AlignAndTruncate(ref_signal, deg_signal);
  ref_signal = std::get<0>(alignment_result);
  deg_signal = std::get<1>(alignment_result);
  double lag = std::get<2>(alignment_result);

  // Ensure there is an initial postive lag.
  ASSERT_EQ(lag, kBestLagPositive2);

  // Confirm that the new degraded signal has been truncated to only
  // the aligned overlap, without the zeros in front.
  ASSERT_EQ(original_ref_duration - kBestLagPositive2,
            deg_signal.GetDuration());
  ASSERT_EQ(original_ref_duration - kBestLagPositive2,
            ref_signal.GetDuration());
}

}  // namespace
}  // namespace Visqol
