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

#include "envelope.h"

#include <algorithm>
#include <complex>
#include <cstdio>
#include <vector>

#include "absl/memory/memory.h"
#include "fast_fourier_transform.h"
#include "misc_vector.h"

namespace Visqol {
AMatrix<double> Envelope::CalcUpperEnv(const AMatrix<double>& signal) {
  double mean = MiscVector::Mean(signal);
  const auto signal_centered = signal - mean;
  AMatrix<std::complex<double>> hilbert = Hilbert(signal_centered);
  AMatrix<double> hilbert_amp(hilbert.NumRows(), hilbert.NumCols());
  // to amplitude
  for (size_t i = 0; i < hilbert.NumRows(); i++) {
    hilbert_amp(i) = std::abs(hilbert(i));
  }

  return hilbert_amp + mean;
}

AMatrix<std::complex<double>> Envelope::Hilbert(const AMatrix<double>& signal) {
  auto fft_manager = absl::make_unique<FftManager>(signal.NumElements());
  AMatrix<std::complex<double>> freq_domain_signal =
      FastFourierTransform::Forward1d(fft_manager, signal);

  const bool is_odd = signal.NumRows() % 2 == 1;
  const bool is_non_empty = signal.NumRows() > 0;
  const double kInitVal = 0.0;
  std::vector<double> hilbert_scaling(freq_domain_signal.NumRows(), kInitVal);
  hilbert_scaling[0] = 1;

  // even and nonempty, used for scaling
  if (!is_odd && is_non_empty) {
    hilbert_scaling[signal.NumRows() / 2] = 1.0;
  } else if (is_odd && is_non_empty) {
    hilbert_scaling[signal.NumRows() / 2] = 2.0;
  }
  const size_t n = (is_odd) ? (freq_domain_signal.NumRows() + 1) / 2
                            : ((freq_domain_signal.NumRows()) / 2);
  for (size_t row_index = 1; row_index < n; row_index++) {
    hilbert_scaling[row_index] = 2.0;
  }

  AMatrix<std::complex<double>> element_wise_prod(
      freq_domain_signal.NumElements(), 1);
  for (size_t i = 0; i < freq_domain_signal.NumRows(); i++) {
    element_wise_prod(i) = freq_domain_signal(i) * hilbert_scaling[i];
  }
  auto hilbert =
      FastFourierTransform::Inverse1d(fft_manager, element_wise_prod);
  return hilbert;
}
}  // namespace Visqol
