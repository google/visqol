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

#ifndef VISQOL_INCLUDE_FAST_FOURIER_TRANSFORM_H
#define VISQOL_INCLUDE_FAST_FOURIER_TRANSFORM_H

#include <complex>
#include <memory>

#include "amatrix.h"
#include "fft_manager.h"

namespace Visqol {
/**
 * This class provides fast Fourier Transform functionality.
 */
class FastFourierTransform {
 public:
  /**
   * For a given input matrix of doubles, perform a Fast Fourier Transform on
   * it. Audio files are converted to mono channel for processing, so this
   * 'matrix' will be 1D.
   *
   * @param fft_manager The manager required for performing the FFT.
   * @param in_matrix The input matrix.
   *
   * @return The resulting 1D matrix of complex doubles.
   */
  static AMatrix<std::complex<double>> Forward1d(
      const std::unique_ptr<FftManager>& fft_manager,
      const AMatrix<double>& in_matrix);

  /**
   * For a given input matrix of doubles, perform a Fast Fourier Transform on
   * it. Audio files are converted to mono channel for processing, so this
   * 'matrix' will be 1D. Allows the FFT size to be specified - zeros will be
   * added to the end of the input 'matrix' to pad to the desired size.
   *
   * @param fft_manager The manager required for performing the FFT.
   * @param in_matrix The input matrix.
   * @param points The FFT size.
   *
   * @return The resulting 1D matrix of complex doubles.
   */
  static AMatrix<std::complex<double>> Forward1d(
      const std::unique_ptr<FftManager>& fft_manager,
      const AMatrix<double>& in_matrix, const size_t points);

  /**
   * For a given input 1D matrix of complex doubles, perform the Inverse Fast
   * Fourier Transform on it. A 1D matrix of complex doubles will be returned.
   *
   * @param fft_manager The manager required for performing the FFT.
   * @param in_matrix the input matrix.
   *
   * @return the resulting 1D matrix of complex doubles.
   */
  static AMatrix<std::complex<double>> Inverse1d(
      const std::unique_ptr<FftManager>& fft_manager,
      const AMatrix<std::complex<double>>& in_matrix);

  /**
   * For a given input 1D matrix of complex doubles, perform the Inverse Fast
   * Fourier Transform on it and return a 1D matrix of real values.
   *
   * @param fft_manager The manager required for performing the FFT.
   * @param in_matrix the input matrix.
   *
   * @return the resulting 1D matrix of doubles.
   */
  static AMatrix<double> Inverse1dConjSym(
      const std::unique_ptr<FftManager>& fft_manager,
      const AMatrix<std::complex<double>>& in_matrix);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_FAST_FOURIER_TRANSFORM_H
