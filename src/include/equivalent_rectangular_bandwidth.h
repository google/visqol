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

#ifndef VISQOL_INCLUDE_EQUIVALENTRECTANGULARBANDWIDTH_H
#define VISQOL_INCLUDE_EQUIVALENTRECTANGULARBANDWIDTH_H

#include <cstdint>
#include <vector>

namespace Visqol {

/**
 * The result of the equivalent rectangular bandwidth (ERB) filter set
 * creation.
 */
struct ErbFiltersResult {
  /**
   * The filter coefficients for the ERB filter.
   */
  std::vector<std::vector<double>> filterCoeffs;

  /**
   * The vector of center frequencies to be used in the ERB filter.
   */
  std::vector<double> centerFreqs;
};

/**
 * Implementation of an equivalent rectangular bandwidth (ERB) class based on
 * Slaney's Auditory Toolbox functions. Calculates the frequency bands to use
 * and filter coefficients.
 */
class EquivalentRectangularBandwidth {
 public:
  /**
   * An implementation of Slaney's Matlab MakeERBFilters.
   *
   * @param sample_rate The sample rate of the input signals.
   * @param num_channels The number of frequency bands desired in the filter
   *    set.
   * @param low_freq The value of the lowest center frequency to use in the
   *    filter set.
   * @param high_freq The value of the highest center frequency to use in the
   *    filter set. If this frequency is greater than half the sampling rate,
   *    then the value of half the sampling rate will be used instead.
   *
   * @return The resulting ERB center frequencies and filter coefficients.
   */
  static ErbFiltersResult MakeFilters(std::size_t sample_rate,
                                      std::size_t num_channels, double low_freq,
                                      double high_freq);

 private:
  /**
   * Compute N center frequencies that are uniformly spaced between the given
   * highest frequency and the given lowest frequency on an ERB scale.
   * Equivalent to Slaney's ERBSpace function.
   *
   * @param low_freq The value of the lowest center frequency to use in the
   *    filter set.
   * @param high_freq The value of the highest possible frequency for the sample
   *    rate of the input signal.
   * @param num_channels The number of frequency bands desired in the filter
   *    set.
   *
   * @return A vector of center frequencies to be used in the ERB filter.
   */
  static std::vector<double> CalcUniformCenterFreqs(double low_freq,
                                                    double high_freq,
                                                    std::size_t num_channels);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_EQUIVALENTRECTANGULARBANDWIDTH_H
