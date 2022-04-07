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

#ifndef VISQOL_INCLUDE_ENVELOPE_H
#define VISQOL_INCLUDE_ENVELOPE_H

#include <complex>

#include "amatrix.h"

namespace Visqol {

/**
 * This class is used to calculate the envelope for a given signal.
 */
class Envelope {
 public:
  /**
   * For a given signal, calculate the upper envelope.
   * Assumes a single dimensional input matrix.
   *
   * @param signal The input single dimensional matrix representing the signal.
   * @return The upper envelope for the input signal.
   */
  static AMatrix<double> CalcUpperEnv(const AMatrix<double>& signal);

 private:
  /**
   * Perform a Hilbert Transform on a given single dimensional input signal.
   * Based on the Matlab implementation for Hilbert.
   *
   * @param signal The input single dimensional signal to perform the Hilbert
   *               transform on.
   * @return The matrix of complex doubles containing the result of the Hilbert
   *         Transform.
   */
  static AMatrix<std::complex<double>> Hilbert(const AMatrix<double>& signal);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_ENVELOPE_H
