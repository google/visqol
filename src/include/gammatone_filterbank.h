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

#ifndef VISQOL_INCLUDE_GAMMATONEFILTERBANK_H
#define VISQOL_INCLUDE_GAMMATONEFILTERBANK_H

#include <cstddef>
#include <valarray>

#include "amatrix.h"

#include "Eigen/Dense"

namespace Visqol {

/**
 * A bank of gammatone filters that are applied to each frame in the signal
 * during the production of a spectrogram representation of the signal.
 */
class GammatoneFilterBank {
 public:
  /**
   * Constructs the GammatoneFilterBank with the specified number of bands and
   * the minimum frequency to utilise.
   *
   * @param num_bands The number of frequency bands to filter with. There will
   *    be this number of bands in the spectrogram that is produced.
   * @param min_freq The lowest frequency value to include in the filter.
   */
  GammatoneFilterBank(const size_t num_bands, const double min_freq);

  /**
   * Get the number of bands in this filter bank.
   *
   * @return The number of bands in this filter bank.
   */
  size_t GetNumBands() const;

  /**
   * Get the lowest frequency that is used in this filter bank.
   *
   * @return The lowest frequency that is used in this filter bank.
   */
  double GetMinFreq() const;

  /**
   * Apply the filter bank to the signal in a column wise manner. This greatly
   * reduces the memory consumption when compared to row rise filtering.
   *
   * The matrix that is returned will have dimensions equaling: the number
   * of bands in the filter bank * the number of samples in the input frame.
   * These values correspond to rows and cols in the matrix respectively.
   *
   * @param signal The signal to be filtered.
   *
   * @return The filtered output.
   */
  Eigen::MatrixXd ApplyFilter(const Eigen::MatrixXd &signal);

  /**
   * Set the equivalent rectangular bandwidth filter coefficients that are to
   * be used.
   *
   * @param filter_coeffs The input filter coefficients.
   */
  void SetFilterCoefficients(const AMatrix<double>& filter_coeffs);

  /**
   * Reset the filter conditions to zero before filtering a signal. If the
   * filter bank is to be used for multiple signals, this must be called before
   * starting to filter each signal.
   */
  void ResetFilterConditions();

 private:
  /**
   * The number of frequency bands to filter with. There will be this number of
   * bands in the spectrogram that is produced.
   */
  size_t num_bands_;

  /**
   * The lowest frequency value to include in the filter.
   */
  double min_freq_;

  Eigen::MatrixXd fltr_cond_1_;
  Eigen::MatrixXd fltr_cond_2_;
  Eigen::MatrixXd fltr_cond_3_;
  Eigen::MatrixXd fltr_cond_4_;
  Eigen::MatrixXd a1;
  Eigen::MatrixXd a2;
  Eigen::MatrixXd a3;
  Eigen::MatrixXd a4;
  Eigen::MatrixXd b;

  /**
   * This is dependent to the filters order. For a 2nd order filter
   * we need 3 coeffs for numerator and denominator.
   */
   const int kFilterLength = 3;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_GAMMATONEFILTERBANK_H
