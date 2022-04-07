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

#ifndef VISQOL_INCLUDE_SIGNALFILTER_H
#define VISQOL_INCLUDE_SIGNALFILTER_H

#include <valarray>

namespace Visqol {

/**
 * Struct containing the results of a filter.
 */
struct FilterResults {
  /**
   * The output filtered signal.
   */
  std::valarray<double> filteredSignal;

  /**
   * The output filter conditions after applying the filter.
   */
  std::valarray<double> finalConditions;
};

/**
 * Class used for applying a filter to a signal.
 */
class SignalFilter {
 public:
  /**
   * Takes an input singal and applies a filter using the input
   * coefficients and conditions.
   *
   * @param numer_coeffs The numerator coefficients to filter with.
   * @param denom_coeffs The denominator coefficients to filter with.
   * @param signal The signal to be filtered.
   * @param init_conditions The initial filter conditions.
   *
   * @return The resulting filtered signal and conditions.
   */
  static FilterResults Filter(const std::valarray<double>& numer_coeffs,
                              const std::valarray<double>& denom_coeffs,
                              const std::valarray<double>& signal,
                              const std::valarray<double>& init_conditions);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_SIGNALFILTER_H
