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

#include "convolution_2d.h"

#include <utility>

#include "amatrix.h"

namespace Visqol {

template <class T>
AMatrix<T> Convolution2D<T>::Valid2DConvWithBoundary(
    const AMatrix<T>& fir_filter, AMatrix<T> input_matrix) {
  input_matrix = AddMatrixBoundary(std::move(input_matrix));

  int i_r_c = input_matrix.NumRows();  // input row count
  int i_c_c = input_matrix.NumCols();  // input col count
  int f_r_c = fir_filter.NumRows();    // filter row count
  int f_c_c = fir_filter.NumCols();    // filter col count
  int o_r_c = i_r_c - f_r_c + 1;       // output row count
  int o_c_c = i_c_c - f_c_c + 1;       // output col count
  if (o_r_c < 0) o_r_c = 0;
  if (o_c_c < 0) o_c_c = 0;

  int filter_size = f_r_c * f_c_c;

  AMatrix<T> out_matrix(o_r_c, o_c_c);

  for (int o_row = 0; o_row < o_r_c; o_row++) {    // output rows
    for (int o_col = 0; o_col < o_c_c; o_col++) {  // output cols
      T sum = 0;
      size_t filter_index = filter_size - 1;
      for (int f_col = 0; f_col < f_c_c; f_col++) {    // filter cols
        for (int f_row = 0; f_row < f_r_c; f_row++) {  // filter rows
          const size_t idx = ((f_col + o_col) * i_r_c) + f_row + o_row;
          sum += input_matrix(idx) * fir_filter(filter_index--);
        }
      }
      out_matrix(o_row, o_col) = sum;
    }
  }

  return out_matrix;
}

template <class T>
AMatrix<T> Convolution2D<T>::AddMatrixBoundary(AMatrix<T>&& input_matrix) {
  // Pad the matrix by 1 on either side of both dimensions.
  AMatrix<T> output_matrix = CopyMatrixWithinPadding(input_matrix, 1, 1, 1, 1);
  output_matrix.SetRow(0, output_matrix.GetRow(1));
  output_matrix.SetRow(output_matrix.NumRows() - 1,
                       output_matrix.GetRow(output_matrix.NumRows() - 2));
  output_matrix.SetColumn(0, output_matrix.GetColumn(1));
  output_matrix.SetColumn(output_matrix.NumCols() - 1,
                          output_matrix.GetColumn(output_matrix.NumCols() - 2));
  return output_matrix;
}

template <class T>
AMatrix<T> Convolution2D<T>::CopyMatrixWithinPadding(
    const AMatrix<T>& input_matrix, const size_t row_prepad_amt,
    const size_t row_postpad_amt, const size_t col_prepad_amt,
    const size_t col_postpad_amt) {
  AMatrix<T> output_matrix(
      input_matrix.NumRows() + row_prepad_amt + row_postpad_amt,
      input_matrix.NumCols() + col_prepad_amt + col_postpad_amt);
  for (size_t row_i = 0; row_i < input_matrix.NumRows(); row_i++) {
    for (size_t col_i = 0; col_i < input_matrix.NumCols(); col_i++) {
      output_matrix(row_i + row_prepad_amt, col_i + col_prepad_amt) =
          input_matrix(row_i, col_i);
    }
  }
  return output_matrix;
}

template class Convolution2D<double>;
}  // namespace Visqol
