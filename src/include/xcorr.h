/*
 * Copyright 2019 Google LLC, Andrew Hines
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef VISQOL_INCLUDE_XCORR_H
#define VISQOL_INCLUDE_XCORR_H

#include <complex>
#include <memory>
#include <vector>

#include "amatrix.h"
#include "fast_fourier_transform.h"

namespace Visqol {

/**
 * This class is used to perform a cross correlation between two signals.
 */
class XCorr {
 public:
  /**
   * Using cross correlation, calculate the best lag value between the two
   * signals. The lag describes how many samples one signal lags behind the
   * other. If signal_1 is ahead of signal_2 the lag value is positive. If
   * signal_1 is behind signal_2, the lag value is negative. If the signals are
   * already aligned the lag will be 0.
   *
   * @param signal_1 The first signal in the pair of signals to be correlated.
   * @param signal_2 The second signal in the pair of signals to be correlated.
   *
   * @return The best lag value resulting from the cross correlation.
   */
  static int64_t FindLowestLagIndex(const AMatrix<double>& signal_1,
                                    const AMatrix<double>& signal_2);

 private:
  /**
   * Helper function used to calculate the inverse fft of the result of the
   * pointwise product of the two signal's forward fft.
   *
   * These two fft operations are split over these functions to shorten the
   * lifespan of these variables to reduce peak memory consumption.
   *
   * @param signal_1 The first signal to be processed.
   * @param signal_1 The second signal to be processed.
   *
   * @return The inverse fft of the pointwise product of the two signal's
   *    forward fft.
   */
  static std::vector<double> InverseFFTPointwiseProduct(
      const AMatrix<double>& signal_1, const AMatrix<double>& signal_2);

  /**
   * Helper function used to the pointwise product of the two signal's forward
   * fft.
   *
   * These two fft operations are split over these functions to shorten the
   * lifespan of these variables to reduce peak memory consumption.
   *
   * @param signal_1 The first signal to be processed.
   * @param signal_1 The second signal to be processed.
   *
   * @return The pointwise product of the two signal's forward fft.
   */
  static AMatrix<std::complex<double>> FFTPointwiseProduct(
      const std::vector<double>& signal_1, const std::vector<double>& signal_2,
      const std::unique_ptr<FftManager>& fft_manager, const size_t fft_points);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_XCORR_H
