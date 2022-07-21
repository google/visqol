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

#include "fast_fourier_transform.h"

#include <complex>
#include <iterator>
#include <memory>
#include <vector>

#include "amatrix.h"

namespace Visqol {

AMatrix<std::complex<double>> FastFourierTransform::Forward1d(
    const std::unique_ptr<FftManager>& fft_manager,
    const AMatrix<double>& in_matrix) {
  // Populate the time channel.
  auto input_itr = in_matrix.cbegin();
  fft_manager->GetTimeChannel().Clear();
  for (size_t i = 0; i < fft_manager->GetSamplesPerChannel(); i++) {
    fft_manager->GetTimeChannel()[i] = *input_itr;
    input_itr++;
  }

  // Convert time to freq domain, ordered canonically.
  fft_manager->GetFreqChannel().Clear();
  fft_manager->FreqFromTimeDomain(fft_manager->GetTimeChannel(),
                                  &fft_manager->GetFreqChannel());

  // Convert freq domain to complex vector.
  std::vector<std::complex<double>> freq_cplx_vector;
  freq_cplx_vector.reserve(fft_manager->GetFftSize());
  for (size_t i = 0; i < fft_manager->GetFftSize(); i += 2) {
    const auto real_num = static_cast<double>(fft_manager->GetFreqChannel()[i]);
    const auto imag_num =
        static_cast<double>(fft_manager->GetFreqChannel()[i + 1]);
    freq_cplx_vector.push_back(std::complex<double>{real_num, imag_num});
  }

  // Pull out the Nyquist bin from the 0Hz bin and push onto end.
  std::complex<double> nyquist_bin{freq_cplx_vector[0].imag(), 0.0};
  std::complex<double> zero_hz_bin{freq_cplx_vector[0].real(), 0.0};
  freq_cplx_vector[0] = zero_hz_bin;
  freq_cplx_vector.push_back(nyquist_bin);

  // Mirror the freq domain complex vector. Do not add the 0hz or
  // Nyquist bins to the mirror.
  for (int i = freq_cplx_vector.size() - 2; i > 0; i--) {
    freq_cplx_vector.push_back(std::complex<double>{
        freq_cplx_vector[i].real(), freq_cplx_vector[i].imag() * -1});
  }

  // Create a Matrix to return with the mirrored freq domain.
  AMatrix<std::complex<double>> out_matrix(
      freq_cplx_vector.size(), in_matrix.NumCols(), freq_cplx_vector);
  return out_matrix;
}

AMatrix<std::complex<double>> FastFourierTransform::Forward1d(
    const std::unique_ptr<FftManager>& fft_manager,
    const AMatrix<double>& in_matrix, const size_t points) {
  // append zeros to matrix until it's the same size as points
  AMatrix<double> signal = in_matrix;
  signal.Resize(points, signal.NumCols());
  for (size_t i = in_matrix.NumRows(); i < points; i++) {
    signal(i, 0) = 0.0;
  }

  return Forward1d(fft_manager, signal);
}

AMatrix<std::complex<double>> FastFourierTransform::Inverse1d(
    const std::unique_ptr<FftManager>& fft_manager,
    const AMatrix<std::complex<double>>& in_matrix) {
  // Populate the freq channel. Increment i twice per iteration to remove mirror
  auto input_itr = in_matrix.cbegin();
  fft_manager->GetFreqChannel().Clear();
  for (size_t i = 0; i < fft_manager->GetFftSize(); i += 2) {
    fft_manager->GetFreqChannel()[i] = input_itr->real();
    fft_manager->GetFreqChannel()[i + 1] = input_itr->imag();
    input_itr++;
  }

  // Our iterator is now at the Nyquist bin i.e. the start of the mirror.
  // Pull out the Nyquist bin and re-insert into 0Hz bin imaginary part.
  fft_manager->GetFreqChannel()[1] = input_itr->real();

  // Convert the freq domain ordering from canonical to pffft style.
  AudioChannel pffft_channel;
  pffft_channel.Init(fft_manager->GetFftSize());
  pffft_channel.Clear();
  fft_manager->GetPffftFormatFreqBuffer(fft_manager->GetFreqChannel(),
                                        &pffft_channel);

  // Convert the pffft ordered freq domain to time domain.
  fft_manager->GetTimeChannel().Clear();
  fft_manager->TimeFromFreqDomain(pffft_channel,
                                  &fft_manager->GetTimeChannel());
  fft_manager->ApplyReverseFftScaling(&fft_manager->GetTimeChannel());

  // Populate a vector of complex doubles for populating output matrix.
  std::vector<std::complex<double>> out_cmplx_vector;
  out_cmplx_vector.reserve(fft_manager->GetSamplesPerChannel());
  for (size_t i = 0; i < fft_manager->GetSamplesPerChannel(); i++) {
    auto cplx_num = std::complex<double>{
        static_cast<double>(fft_manager->GetTimeChannel()[i]), 0.0};
    out_cmplx_vector.push_back(cplx_num);
  }

  AMatrix<std::complex<double>> out_matrix(
      out_cmplx_vector.size(), in_matrix.NumCols(), out_cmplx_vector);
  return out_matrix;
}

AMatrix<double> FastFourierTransform::Inverse1dConjSym(
    const std::unique_ptr<FftManager>& fft_manager,
    const AMatrix<std::complex<double>>& in_matrix) {
  auto cmplx_inv = Inverse1d(fft_manager, in_matrix);

  // Populate a vector of doubles with the 'real' part of the ifft for return.
  std::vector<double> out_double_vector;
  out_double_vector.reserve(cmplx_inv.NumElements());
  for (auto itr = cmplx_inv.begin(); itr != cmplx_inv.end(); itr++) {
    out_double_vector.push_back(itr->real());
  }

  AMatrix<double> out_matrix(out_double_vector.size(), in_matrix.NumCols(),
                             out_double_vector);
  return out_matrix;
}
}  // namespace Visqol
