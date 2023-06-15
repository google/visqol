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

#include "amatrix.h"

#include <complex>
#include <iostream>
#include <utility>
#include <valarray>
#include <vector>

#include "absl/types/span.h"

namespace Visqol {
template <typename T>
inline AMatrix<T>::AMatrix(const AMatrix<T>& other) {
  matrix_ = BackendMatrix<T>(other.matrix_);
}

template <typename T>
inline AMatrix<T>::AMatrix(const BackendMatrix<T>& mat) {
  matrix_ = mat;
}

template <typename T>
inline AMatrix<T>::AMatrix(const std::vector<T>& col) {
  matrix_ = BackendMatrix<T>::Map(col.data(), col.size(), 1);
}

template <typename T>
inline AMatrix<T>::AMatrix(const absl::Span<T>& col) {
  matrix_ = BackendMatrix<T>::Map(col.data(), col.size(), 1);
}

template <typename T>
inline AMatrix<T>::AMatrix(const std::valarray<T>& va) {
  std::vector<T> v;
  v.resize(va.size());
  v.assign(std::begin(va), std::end(va));
  matrix_ = BackendMatrix<T>::Map(v.data(), v.size(), 1);
}

template <typename T>
inline AMatrix<T>::AMatrix(const std::vector<std::vector<T>>& vecOfCols) {
  // assumes all vectors all of equal length
  matrix_ = BackendMatrix<T>(vecOfCols[0].size(), vecOfCols.size());
  for (size_t i = 0; i < vecOfCols.size(); i++) {
    matrix_.col(i) = BackendMatrix<T>::Map(vecOfCols[i].data(), vecOfCols[i].size(), 1);
  }
}

template <typename T>
inline AMatrix<T>::AMatrix(size_t rows, size_t cols, std::vector<T>&& data) {
  // TODO: We should avoid copying here
  matrix_ = BackendMatrix<T>::Map(data.data(), rows, cols);
}

template <typename T>
inline AMatrix<T>::AMatrix(size_t rows, size_t cols,
                           const std::vector<T>& data) {
  matrix_ = BackendMatrix<T>::Map(data.data(), rows, cols);
}

template <typename T>
inline AMatrix<T>::AMatrix(size_t rows, size_t cols) {
  matrix_ = BackendMatrix<T>(rows, cols);
}

template <typename T>
inline AMatrix<T>::AMatrix(BackendMatrix<T>&& matrix) {
  matrix_ = std::move(matrix);
}

template <typename T>
AMatrix<T> AMatrix<T>::GetSpan(size_t rowStart, size_t rowEnd, size_t colStart,
                               size_t colEnd) const {
  size_t num_rows = rowEnd - rowStart + 1;
  size_t num_cols = colEnd - colStart + 1;
  BackendMatrix<T> m = matrix_.block(rowStart, colStart, num_rows, num_cols);
  return AMatrix<T>(m);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::Filled(size_t rows, size_t cols, T initialValue) {
  BackendMatrix<T> m (rows, cols);
  m.fill(initialValue);
  return AMatrix<T>(std::move(m));
}

template <typename T>
inline T& AMatrix<T>::operator()(size_t row, size_t column) {
  return matrix_(row, column);
}

template <typename T>
inline T AMatrix<T>::operator()(size_t row, size_t column) const {
  return matrix_(row, column);
}

template <typename T>
inline T& AMatrix<T>::operator()(size_t elementIndex) {
  return *(matrix_.data() + elementIndex);
}

template <typename T>
inline T AMatrix<T>::operator()(size_t elementIndex) const {
  return *(matrix_.data() + elementIndex);
}

template <typename T>
inline AMatrix<T>& AMatrix<T>::operator=(const AMatrix<T>& other) {
  matrix_ = other.matrix_;
  return *this;
}

template <typename T>
inline bool AMatrix<T>::operator==(const AMatrix<T>& other) const {
  if (matrix_.rows() != other.matrix_.rows()) return false;
  if (matrix_.cols() != other.matrix_.cols()) return false;
  return matrix_ == other.matrix_;
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator+(const AMatrix<T>& other) const {
  return AMatrix<T>((matrix_.array() + other.matrix_.array()).matrix());
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator+(T v) const {
  return AMatrix<T>(matrix_.array() + v);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator*(T v) const {
  return AMatrix<T>(matrix_.array() * v);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator-(T v) const {
  return AMatrix<T>(matrix_.array() - v);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator/(T v) const {
  return AMatrix<T>(matrix_.array() / v);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator-(const AMatrix<T>& m) const {
  return AMatrix<T>((matrix_.array() - m.matrix_.array()).matrix());
}

template <typename T>
inline AMatrix<T> AMatrix<T>::PointWiseProduct(const AMatrix<T>& m) const {
  return AMatrix<T>(std::move(matrix_.cwiseProduct(m.matrix_)));
}

template <typename T>
inline AMatrix<T> AMatrix<T>::PointWiseDivide(const AMatrix<T>& m) const {
  return AMatrix<T>(std::move(matrix_.array() / m.matrix_.array()));
}

template <typename T>
inline AMatrix<T> AMatrix<T>::Transpose() const {
  return AMatrix<T>{std::move(matrix_.transpose())};
}

template <typename T>
inline std::vector<T> AMatrix<T>::RowSubset(size_t rowIndex,
                                            size_t startColumnIndex,
                                            size_t endColumnIndex) const {
  std::vector<T> vec;
  vec.resize(endColumnIndex - startColumnIndex + 1);
  for (size_t colIndex = startColumnIndex; colIndex <= endColumnIndex; ++colIndex) {
    vec[colIndex - startColumnIndex] = matrix_(rowIndex, colIndex);
  }
  return vec;
}

template <typename T>
inline size_t AMatrix<T>::LongestDimensionLength() const {
  return (matrix_.rows() > matrix_.cols()) ? matrix_.rows() : matrix_.cols();
}

template <>
inline AMatrix<double> AMatrix<double>::Abs() const {
  return AMatrix<double>(matrix_.cwiseAbs());
}

template <>
inline AMatrix<double> AMatrix<std::complex<double>>::Abs() const {
  BackendMatrix<double> absMatrix (matrix_.rows(), matrix_.cols());
  for (size_t rowIndex = 0; rowIndex < matrix_.rows(); rowIndex++) {
    for (size_t colIndex = 0; colIndex < matrix_.rows(); colIndex++) {
      absMatrix(rowIndex, colIndex) = std::abs(matrix_(rowIndex, colIndex));
    }
  }
  return AMatrix<double>(std::move(absMatrix));
}

template <typename T>
inline std::vector<T> AMatrix<T>::GetRow(size_t row) const {
  std::vector<T> vec;
  vec.resize(matrix_.cols());
  for (size_t i = 0; i < matrix_.cols(); ++i)
  {
    vec[i] = matrix_(row, i);
  }
  return vec;
}

template <typename T>
inline AMatrix<T> AMatrix<T>::GetRows(size_t rowStart, size_t rowEnd) const {
  return AMatrix<T>(std::move(matrix_.middleRows(rowStart, rowEnd - rowStart + 1)));
}

template <typename T>
inline AMatrix<T> AMatrix<T>::GetColumn(size_t column) const {
  return AMatrix<T>(matrix_.col(column));
}

template <typename T>
inline AMatrix<T> AMatrix<T>::GetColumns(size_t colStart, size_t colEnd) const {
  return AMatrix<T>(matrix_.middleCols(colStart, colEnd - colStart + 1));
}

template <typename T>
inline void AMatrix<T>::SetRow(size_t rowIndex, const std::vector<T>& row) {
  matrix_.row(rowIndex) = BackendMatrix<T>::Map(row.data(), 1, row.size());
}

template <typename T>
inline void AMatrix<T>::SetColumn(size_t colIndex, AMatrix<T>&& col) {
  matrix_.col(colIndex) = std::move(col.matrix_);
}

template <typename T>
inline void AMatrix<T>::SetColumn(size_t colIndex, std::vector<T>&& col) {
  matrix_.col(colIndex) = std::move(BackendMatrix<T>::Map(col.data(), col.size(), 1));
}

template <typename T>
inline void AMatrix<T>::SetColumn(size_t colIndex, std::valarray<T>&& col) {
  for (size_t i = 0; i < col.size(); ++i) {
    matrix_(i, colIndex) = col[i];
  }
}

template <typename T>
inline void AMatrix<T>::SetRow(size_t rowIndex, std::valarray<T>&& row) {
  for (size_t colIndex = 0; colIndex < matrix_.cols(); ++colIndex) {
    matrix_(rowIndex, colIndex) = row[colIndex];
  }
}

template <typename T>
void AMatrix<T>::Print() const {
  printf("\n");
  std::cout << matrix_ << std::endl;
}

template <>
void AMatrix<std::complex<double>>::PrintSummary(const char* s) const {
  printf("%s\n", s);
  size_t outMatSize = matrix_.rows();
  printf("first five\n");
  for (size_t i = 0; i < 5; i++) {
    printf("complex[%2zu] = %9.20f , %9.20f\n", i, (matrix_.data() + i)->real(),
           (matrix_.data() + i)->imag());
  }
  printf("middle \n");
  for (size_t i = (outMatSize / 2) - 4; i < (outMatSize / 2) + 6; i++) {
    printf("complex[%2zu] = %9.20f , %9.20f\n", i, (matrix_.data() + i)->real(),
           (matrix_.data() + i)->imag());
  }
  printf("last five\n");
  for (size_t i = outMatSize - 6; i < outMatSize; i++) {
    printf("complex[%2zu] = %9.20f , %9.20f\n", i, (matrix_.data() + i)->real(),
           (matrix_.data() + i)->imag());
  }
}

template <>
void AMatrix<double>::PrintSummary(const char* s) const {
  printf("%s\n", s);
  size_t outMatSize = matrix_.rows();
  printf("first five\n");
  for (size_t i = 0; i < 5; i++) {
    printf("double[%2zu] = %9.20f\n", i, *(matrix_.data() + i));
  }
  printf("middle \n");
  for (size_t i = (outMatSize / 2) - 4; i < (outMatSize / 2) + 6; i++) {
    printf("double[%2zu] = %9.20f\n", i, *(matrix_.data() + i));
  }
  printf("last five\n");
  for (size_t i = outMatSize - 6; i < outMatSize; i++) {
    printf("double[%2zu] = %9.20f\n", i, *(matrix_.data() + i));
  }
}

template <typename T>
inline const T* AMatrix<T>::MemPtr() const {
  return matrix_.data();
}

template <typename T>
inline void AMatrix<T>::Resize(const size_t rows, const size_t cols) {
  // We need to try to keep the same data
  matrix_.conservativeResize(rows, cols);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::JoinVertically(const AMatrix<T>& other) const {
  // Asserting other matrix has the same number of columns
  if (matrix_.cols() != other.matrix_.cols()) return AMatrix<T> ();
  size_t numCols = matrix_.cols();
  size_t numRows = matrix_.rows() + other.matrix_.rows();
  BackendMatrix<T> m (numRows, numCols);
  m.topRows(matrix_.rows()) = matrix_;
  m.bottomRows(other.matrix_.rows()) = other.matrix_;
  return m;
}

template <typename T>
inline AMatrix<T> AMatrix<T>::FlipUpDown() const {
  return AMatrix<T>(matrix_.colwise().reverse());
}

template <typename T>
inline AMatrix<T> AMatrix<T>::Mean(kDimension dim) const {
  if (dim == kDimension::COLUMN) {
    return AMatrix<T>(matrix_.colwise().mean());
  }
  else { // if (dim == kDimension::ROW)
    return AMatrix<T>(matrix_.rowwise().mean());
  }
}

template <typename T>
inline AMatrix<double> AMatrix<T>::StdDev(kDimension dim) const {
  // The second parameter of arma::stddev is norm_type.
  // norm_type=0 means that stddev should use the unbiased estimate.
  if (dim == kDimension::COLUMN) {
    if (matrix_.rows() == 1) {
      return AMatrix<double>(BackendMatrix<double>::Zero(1, matrix_.cols()));
    }
    else {
      return AMatrix<double>((matrix_.rowwise() - matrix_.colwise().mean()).colwise().norm()/std::sqrt((double)matrix_.rows()-1.0));
    }
  }
  else { // if (dim == kDimension::ROW)
    if (matrix_.cols() == 1) {
      return AMatrix<double>(BackendMatrix<double>::Zero(matrix_.rows(), 1));
    }
    else {
      return AMatrix<double>((matrix_.colwise() - matrix_.rowwise().mean()).rowwise().norm()/std::sqrt((double)matrix_.cols()-1.0));
    }
  }
}

template <typename T>
inline const BackendMatrix<T>& AMatrix<T>::GetBackendMat() const {
  return matrix_;
}

template <typename T>
inline std::vector<T> AMatrix<T>::ToVector() const {
  std::vector<T> v (static_cast<size_t>(matrix_.rows()));
  for (size_t rowIndex = 0; rowIndex < matrix_.rows(); ++rowIndex) {
    v[rowIndex] = matrix_(rowIndex, 0);
  }
  return std::move(v);
}

template <typename T>
inline std::valarray<T> AMatrix<T>::ToValArray() const {
  std::valarray<T> v (static_cast<size_t>(matrix_.rows()));
  for (size_t rowIndex = 0; rowIndex < matrix_.rows(); ++rowIndex) {
    v[rowIndex] = matrix_(rowIndex, 0);
  }
  return std::move(v);
}

template <typename T>
inline size_t AMatrix<T>::NumRows() const {
  return matrix_.rows();
}
template <typename T>
inline size_t AMatrix<T>::NumCols() const {
  return matrix_.cols();
}
template <typename T>
inline size_t AMatrix<T>::NumElements() const {
  return matrix_.size();
}

template <typename T>
inline typename AMatrix<T>::iterator AMatrix<T>::begin() {
  return matrix_.data();
}
template <typename T>
inline typename AMatrix<T>::iterator AMatrix<T>::end() {
  return matrix_.data() + matrix_.size();
}
template <typename T>
inline typename AMatrix<T>::const_iterator AMatrix<T>::cbegin() const {
  return matrix_.data();
}
template <typename T>
inline typename AMatrix<T>::const_iterator AMatrix<T>::cend() const {
  return matrix_.data() + matrix_.size();
}

template <typename T>
inline const T* AMatrix<T>::data() const {
  return matrix_.data();
}
template <typename T>
inline T* AMatrix<T>::mutData() const {
  return const_cast<T*>(matrix_.data());
}

template class AMatrix<double>;
template class AMatrix<std::complex<double>>;

}  // namespace Visqol
