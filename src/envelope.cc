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

#include <cmath>
#include <memory>

#include "fast_fourier_transform.h"
#include "misc_vector.h"

namespace Visqol {
AMatrix<double> Envelope::CalcUpperEnv(const AMatrix<double>& signal) {
  const double mean = MiscVector::Mean(signal);
  // One bounded workspace per thread; replace it when the input size changes.
  // Keep PFFFT's float arithmetic and original Hilbert scaling unchanged.
  thread_local std::unique_ptr<FftManager> fft_manager;
  if (!fft_manager ||
      fft_manager->GetSamplesPerChannel() != signal.NumElements()) {
    fft_manager = std::make_unique<FftManager>(signal.NumElements());
  }
  auto& time = fft_manager->GetTimeChannel();
  auto& freq = fft_manager->GetFreqChannel();
  for (size_t i = 0; i < signal.NumElements(); ++i) time[i] = signal(i) - mean;
  fft_manager->FreqFromTimeDomain(time, &freq);
  // Canonical real FFT storage packs DC and Nyquist into the first pair.
  // The padded Hilbert spectrum has zero weight at Nyquist.
  if (signal.NumRows() != fft_manager->GetFftSize()) {
    freq[1] = static_cast<double>(freq[1]) * 0.0;
  }
  for (size_t i = 2; i < freq.size(); ++i) {
    freq[i] = static_cast<double>(freq[i]) * 2.0;
  }
  AudioChannel reordered;
  reordered.Init(fft_manager->GetFftSize());
  fft_manager->GetPffftFormatFreqBuffer(freq, &reordered);
  fft_manager->TimeFromFreqDomain(reordered, &time);
  fft_manager->ApplyReverseFftScaling(&time);
  AMatrix<double> amplitude(signal.NumRows(), signal.NumCols());
  for (size_t i = 0; i < signal.NumRows(); ++i) {
    amplitude(i) = std::abs(static_cast<double>(time[i]));
  }
  return amplitude + mean;
}
}  // namespace Visqol
