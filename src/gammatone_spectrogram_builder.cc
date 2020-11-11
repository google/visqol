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

#include "amatrix.h"
#include "analysis_window.h"
#include "audio_signal.h"
#include "equivalent_rectangular_bandwidth.h"
#include "signal_filter.h"
#include "spectrogram.h"
#include "absl/status/statusor.h"

namespace Visqol {

const double GammatoneSpectrogramBuilder::kSpeechModeMaxFreq = 8000.0;

GammatoneSpectrogramBuilder::GammatoneSpectrogramBuilder(
    const GammatoneFilterBank &filter_bank, const bool use_speech_mode) :
    filter_bank_(filter_bank), speech_mode_(use_speech_mode) {}

absl::StatusOr<Spectrogram> GammatoneSpectrogramBuilder::Build(
    const AudioSignal &signal, const AnalysisWindow &window) {
  const auto &sig = signal.data_matrix;
  size_t sample_rate = signal.sample_rate;
  double max_freq = speech_mode_ ? kSpeechModeMaxFreq : sample_rate / 2.0;

  // get gammatone coeffients
  ErbFiltersResult erb_rslt = EquivalentRectangularBandwidth::MakeFilters(
      sample_rate, filter_bank_.GetNumBands(), filter_bank_.GetMinFreq(),
      max_freq);
  AMatrix<double> filter_coeffs = AMatrix<double>(erb_rslt.filterCoeffs);
  filter_coeffs = filter_coeffs.FlipUpDown();

  // set the filter coefficients and init the filter conditions to 0.
  filter_bank_.SetFilterCoefficients(filter_coeffs);
  filter_bank_.ResetFilterConditions();

  // set up the windowing
  size_t hop_size = window.size * window.overlap;

  // ensure that the signal is large enough.
  if (sig.NumRows() <= window.size) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        "Too few samples (" + std::to_string(sig.NumRows()) + ") in signal to "
        "build spectrogram (" + std::to_string(window.size) +
        " required minimum).");
  }
  size_t num_cols = 1 + floor((sig.NumRows() - window.size) / hop_size);
  AMatrix<double> out_matrix(filter_bank_.GetNumBands(), num_cols);

  // run the windowing
  auto sig_val_arr = sig.GetColumn(0).ToValArray();
  for (size_t i = 0; i < out_matrix.NumCols(); i++) {
    const size_t start_col = i * hop_size;
    // select the next frame from the input signal to filter.
    const auto frame = sig_val_arr[std::slice(start_col, window.size, 1)];
    // apply the filter
    filter_bank_.ResetFilterConditions();
    auto filtered_signal = filter_bank_.ApplyFilter(frame);
    // calculate the mean of each row
    std::transform(filtered_signal.begin(), filtered_signal.end(),
                   filtered_signal.begin(),
                   [](decltype(*filtered_signal.begin()) &d) { return d * d; });
    AMatrix<double> row_means = filtered_signal.Mean(kDimension::ROW);
    std::transform(row_means.begin(), row_means.end(), row_means.begin(),
                   [](decltype(*row_means.begin()) &d) { return sqrt(d); });
    // set this filtered frame as a column in the spectrogram
    out_matrix.SetColumn(i, std::move(row_means));
  }

  // Order the center freq bands from lowest to highest.
  std::vector<double> ordered_cfb;
  ordered_cfb.reserve(erb_rslt.centerFreqs.size());
  for (auto itr = erb_rslt.centerFreqs.rbegin();
       itr != erb_rslt.centerFreqs.rend();
       ++itr) {
    ordered_cfb.push_back(*itr);
  }

  Spectrogram spectro(std::move(out_matrix));
  spectro.SetCenterFreqBands(ordered_cfb);
  return spectro;
}
}  // namespace Visqol
