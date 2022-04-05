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

#include "signal_filter.h"

#include <algorithm>
#include <utility>

namespace Visqol {

FilterResults SignalFilter::Filter(
    const std::valarray<double>& numer_coeffs,
    const std::valarray<double>& denom_coeffs,
    const std::valarray<double>& signal,
    const std::valarray<double>& init_conditions) {
  std::valarray<double> fltred_signal(signal.size());
  // Pad final_conditions
  std::valarray<double> final_conditions(denom_coeffs.size());
  std::copy(std::begin(init_conditions), std::end(init_conditions),
            std::begin(final_conditions));

  size_t i, n = denom_coeffs.size();
  for (size_t m = 0; m < fltred_signal.size(); m++) {
    fltred_signal[m] = numer_coeffs[0] * signal[m] + final_conditions[0];
    for (i = 1; i < n; i++) {
      final_conditions[i - 1] = numer_coeffs[i] * signal[m] +
                                final_conditions[i] -
                                denom_coeffs[i] * fltred_signal[m];
    }
  }
  std::valarray<double> final_conditions_out =
      final_conditions[std::slice(0, final_conditions.size() - 1, 1)];
  FilterResults r;
  r.filteredSignal = std::move(fltred_signal);
  r.finalConditions = std::move(final_conditions_out);
  return r;
}
}  // namespace Visqol
