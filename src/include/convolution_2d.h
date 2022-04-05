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

#ifndef VISQOL_INCLUDE_CONVOLUTION_2D_H
#define VISQOL_INCLUDE_CONVOLUTION_2D_H

#include "amatrix.h"

namespace Visqol {

template <class T>
class Convolution2D {
 public:
  /**
   * Perform a 2D convolution on an input matrix with a padded boundary added
   * to it. The convolution result will be a 'valid' shape.
   *
   * @param fir_filter The first matrix with which to perform the convolution.
   *    It functions as a finite impulse response filter.
   * @param input_matrix The second matrix with which to perform the
   *    convolution.
   *
   * @return The resulting 'valid' 2d convolution.
   */
  static AMatrix<T> Valid2DConvWithBoundary(const AMatrix<T>& fir_filter,
                                            AMatrix<T> input_matrix);

 private:
  /**
   * Add a padded boundary to a matrix.
   *
   * @param input_matrix The matrix to pad.
   *
   * @return The padded matrix.
   */
  static AMatrix<T> AddMatrixBoundary(AMatrix<T>&& input_matrix);

  /**
   * Copies a matrix into another matrix, adding the specified amounts of empty
   * row and cols required for padding.
   *
   * @param input_matrix The matrix to copy data from.
   * @param row_prepad_amt The amount of padded rows required at the start of
   *    the output matrix.
   * @param row_postpad_amt The amount of padded rows required at the end of
   *    the output matrix.
   * @param col_prepad_amt The amount of padded cols required at the start of
   *    the output matrix.
   * @param col_postpad_amt The amount of padded cols required at the end of
   *    the output matrix.
   *
   * @return A matrix with the data copied into it from the input matrix, with
   *    the specified amounts of padding around that input data.
   */
  static AMatrix<T> CopyMatrixWithinPadding(const AMatrix<T>& input_matrix,
                                            const size_t row_prepad_amt,
                                            const size_t row_postpad_amt,
                                            const size_t col_prepad_amt,
                                            const size_t col_postpad_amt);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_CONVOLUTION_2D_H
