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

#include "xcorr.h"

#include <math.h>

#include <algorithm>
#include <complex>
#include <memory>
#include <vector>

#include "amatrix.h"
#include "fast_fourier_transform.h"

namespace Visqol {

// Assumes inputs are column vectors
int64_t XCorr::FindLowestLagIndex(const AMatrix<double>& signal_1,
                                  const AMatrix<double>& signal_2) {
  int64_t max_lag = std::max(static_cast<int64_t>(signal_1.NumRows()),
                             static_cast<int64_t>(signal_2.NumRows())) -
                    1;

  const std::vector<double> pointwise_fft_vec =
      InverseFFTPointwiseProduct(signal_1, signal_2);
  // Scan the negative and positive lag ranges directly. Strict comparison
  // preserves the first maximum, including ties across the two ranges.
  double best =
      pointwise_fft_vec[max_lag == 0 ? 0 : pointwise_fft_vec.size() - max_lag];
  int64_t best_lag = -max_lag;
  for (int64_t lag = -max_lag + 1; lag <= max_lag; ++lag) {
    const size_t index = lag < 0 ? pointwise_fft_vec.size() + lag : lag;
    if (pointwise_fft_vec[index] > best) {
      best = pointwise_fft_vec[index];
      best_lag = lag;
    }
  }
  return best_lag;
}

std::vector<double> XCorr::InverseFFTPointwiseProduct(
    const AMatrix<double>& signal_1, const AMatrix<double>& signal_2) {
  const size_t samples = std::max(signal_1.NumRows(), signal_2.NumRows());
  int exponent;
  frexp(std::abs(static_cast<int64_t>(samples) * 2 - 1), &exponent);
  const size_t points = pow(2, exponent);
  thread_local std::unique_ptr<FftManager> fft_manager;
  if (!fft_manager || fft_manager->GetSamplesPerChannel() != points) {
    fft_manager = std::make_unique<FftManager>(points);
  }
  auto& time = fft_manager->GetTimeChannel();
  auto& freq = fft_manager->GetFreqChannel();
  time.Clear();
  for (size_t i = 0; i < signal_2.NumRows(); ++i) time[i] = signal_2(i);
  fft_manager->FreqFromTimeDomain(time, &freq);
  AudioChannel second;
  second.Init(fft_manager->GetFftSize());
  std::copy(freq.begin(), freq.end(), second.begin());
  time.Clear();
  for (size_t i = 0; i < signal_1.NumRows(); ++i) time[i] = signal_1(i);
  fft_manager->FreqFromTimeDomain(time, &freq);
  // Multiply only the nonredundant real FFT bins. Compute in double and round
  // back to float at the same point as the original complex-matrix path.
  freq[0] = static_cast<double>(freq[0]) * second[0];
  freq[1] = static_cast<double>(freq[1]) * second[1];
  for (size_t i = 2; i < freq.size(); i += 2) {
    const std::complex<double> x(freq[i], freq[i + 1]);
    const std::complex<double> y(second[i], second[i + 1]);
    const auto product = x * std::conj(y);
    freq[i] = product.real();
    freq[i + 1] = product.imag();
  }
  fft_manager->GetPffftFormatFreqBuffer(freq, &second);
  fft_manager->TimeFromFreqDomain(second, &time);
  fft_manager->ApplyReverseFftScaling(&time);
  return std::vector<double>(time.begin(), time.end());
}
}  // namespace Visqol
