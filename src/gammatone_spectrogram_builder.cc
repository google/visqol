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

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "amatrix.h"
#include "analysis_window.h"
#include "audio_signal.h"
#include "equivalent_rectangular_bandwidth.h"
#include "signal_filter.h"
#include "spectrogram.h"

#include "Eigen/Dense"

using namespace Eigen;

namespace Visqol {

const double GammatoneSpectrogramBuilder::kSpeechModeMaxFreq = 8000.0;

GammatoneSpectrogramBuilder::GammatoneSpectrogramBuilder(
    const GammatoneFilterBank& filter_bank, const bool use_speech_mode)
    : filter_bank_(filter_bank), speech_mode_(use_speech_mode) {}

absl::StatusOr<Spectrogram> GammatoneSpectrogramBuilder::Build(
    const AudioSignal& signal, const AnalysisWindow& window) {
  const AMatrix<double>& sig = signal.data_matrix;
  size_t sample_rate = signal.sample_rate;
  double max_freq = speech_mode_ ? kSpeechModeMaxFreq : sample_rate / 2.0;

  // Get gammatone coeffients.
  ErbFiltersResult erb_rslt = EquivalentRectangularBandwidth::MakeFilters(
      sample_rate, filter_bank_.GetNumBands(), filter_bank_.GetMinFreq(),
      max_freq);
  AMatrix<double> filter_coeffs = AMatrix<double>(erb_rslt.filterCoeffs);
  filter_coeffs = filter_coeffs.FlipUpDown();

  // Set the filter coefficients and init the filter conditions to 0.
  filter_bank_.SetFilterCoefficients(filter_coeffs);
  filter_bank_.ResetFilterConditions();

  // Set up the windowing.
  size_t hop_size = window.size * window.overlap;

  // Ensure that the signal is large enough.
  if (sig.NumRows() <= window.size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Too few samples (", sig.NumRows(),
                     ") in signal to  build spectrogram (", window.size,
                     " required minimum)."));
  }
  size_t num_cols = 1 + floor((sig.NumRows() - window.size) / hop_size);

  MatrixXd out_eigen (filter_bank_.GetNumBands(), num_cols);
  MatrixXd frame_eigen (window.size, 1);
  MatrixXd hann_window (window.size, 1);
  for (int i = 0; i < window.size; ++i) {
    hann_window(i, 0) = 0.5 - (0.5 * cos(2.0 * M_PI * i / (window.size - 1)));
  }

  // run the windowing
  for (size_t i = 0; i < out_eigen.cols(); i++) {
    const size_t start_row = i * hop_size;
    // select the next frame from the input signal to filter.
    frame_eigen = sig.GetBackendMat().col(0).middleRows(start_row, window.size);

    // Apply a Hann window to reduce artifacts.
    frame_eigen.array() *= hann_window.array();

    // apply the filter
    filter_bank_.ResetFilterConditions();
    MatrixXd filtered_signal = filter_bank_.ApplyFilter(frame_eigen);

    // calculate the mean of each row
    out_eigen.col(i) = filtered_signal.array().square().rowwise().mean().sqrt();
  }

  AMatrix<double> out_matrix(out_eigen);

  // Order the center freq bands from lowest to highest.
  std::vector<double> ordered_cfb;
  ordered_cfb.reserve(erb_rslt.centerFreqs.size());
  for (auto itr = erb_rslt.centerFreqs.rbegin();
       itr != erb_rslt.centerFreqs.rend(); ++itr) {
    ordered_cfb.push_back(*itr);
  }

  Spectrogram spectro(std::move(out_matrix));
  spectro.SetCenterFreqBands(ordered_cfb);
  return spectro;
}
}  // namespace Visqol
