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

#include "gammatone_filterbank.h"

#include <utility>
#include <valarray>

#include "amatrix.h"
#include "signal_filter.h"

namespace Visqol {
GammatoneFilterBank::GammatoneFilterBank(const size_t num_bands,
                                         const double min_freq)
    : num_bands_(num_bands),
      min_freq_(min_freq),
      fltr_cond_1_({0.0, 0.0}, num_bands),
      fltr_cond_2_({0.0, 0.0}, num_bands),
      fltr_cond_3_({0.0, 0.0}, num_bands),
      fltr_cond_4_({0.0, 0.0}, num_bands) {}

size_t GammatoneFilterBank::GetNumBands() const { return num_bands_; }

double GammatoneFilterBank::GetMinFreq() const { return min_freq_; }

void GammatoneFilterBank::ResetFilterConditions() {
  fltr_cond_1_ = {{0.0, 0.0}, num_bands_};
  fltr_cond_2_ = {{0.0, 0.0}, num_bands_};
  fltr_cond_3_ = {{0.0, 0.0}, num_bands_};
  fltr_cond_4_ = {{0.0, 0.0}, num_bands_};
}

void GammatoneFilterBank::SetFilterCoefficients(
    const AMatrix<double>& filter_coeffs) {
  fltr_coeff_A0_ = filter_coeffs.GetColumn(0).ToValArray();
  fltr_coeff_A11_ = filter_coeffs.GetColumn(1).ToValArray();
  fltr_coeff_A12_ = filter_coeffs.GetColumn(2).ToValArray();
  fltr_coeff_A13_ = filter_coeffs.GetColumn(3).ToValArray();
  fltr_coeff_A14_ = filter_coeffs.GetColumn(4).ToValArray();
  fltr_coeff_A2_ = filter_coeffs.GetColumn(5).ToValArray();
  fltr_coeff_B0_ = filter_coeffs.GetColumn(6).ToValArray();
  fltr_coeff_B1_ = filter_coeffs.GetColumn(7).ToValArray();
  fltr_coeff_B2_ = filter_coeffs.GetColumn(8).ToValArray();
  fltr_coeff_gain_ = filter_coeffs.GetColumn(9).ToValArray();
}

AMatrix<double> GammatoneFilterBank::ApplyFilter(
    const std::valarray<double>& signal) {
  AMatrix<double> output(num_bands_, signal.size());
  std::valarray<double> a1, a2, a3, a4, b;

  // Loop over each filter coefficient now to produce a filtered column.
  for (size_t chan = 0; chan < num_bands_; chan++) {
    a1 = {fltr_coeff_A0_[chan] / fltr_coeff_gain_[chan],
          fltr_coeff_A11_[chan] / fltr_coeff_gain_[chan],
          fltr_coeff_A2_[chan] / fltr_coeff_gain_[chan]};
    a2 = {fltr_coeff_A0_[chan], fltr_coeff_A12_[chan], fltr_coeff_A2_[chan]};
    a3 = {fltr_coeff_A0_[chan], fltr_coeff_A13_[chan], fltr_coeff_A2_[chan]};
    a4 = {fltr_coeff_A0_[chan], fltr_coeff_A14_[chan], fltr_coeff_A2_[chan]};
    b = {fltr_coeff_B0_[chan], fltr_coeff_B1_[chan], fltr_coeff_B2_[chan]};

    // First filter
    auto fltr_rslt = SignalFilter::Filter(a1, b, signal, fltr_cond_1_[chan]);
    fltr_cond_1_[chan] = fltr_rslt.finalConditions;

    // Second filter
    fltr_rslt = SignalFilter::Filter(a2, b, std::move(fltr_rslt.filteredSignal),
                                     fltr_cond_2_[chan]);
    fltr_cond_2_[chan] = fltr_rslt.finalConditions;

    // Third filter
    fltr_rslt = SignalFilter::Filter(a3, b, std::move(fltr_rslt.filteredSignal),
                                     fltr_cond_3_[chan]);
    fltr_cond_3_[chan] = fltr_rslt.finalConditions;

    // Fourth filter
    fltr_rslt = SignalFilter::Filter(a4, b, std::move(fltr_rslt.filteredSignal),
                                     fltr_cond_4_[chan]);
    fltr_cond_4_[chan] = fltr_rslt.finalConditions;

    output.SetRow(chan, std::move(fltr_rslt.filteredSignal));
  }
  return output;
}
}  // namespace Visqol
