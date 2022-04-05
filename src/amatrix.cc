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

#include <armadillo>
#include <complex>
#include <utility>
#include <valarray>
#include <vector>

#include "absl/types/span.h"

namespace Visqol {
template <typename T>
inline AMatrix<T>::AMatrix(const AMatrix<T>& other) {
  matrix_ = arma::Mat<T>(other.matrix_);
}

template <typename T>
inline AMatrix<T>::AMatrix(const arma::Mat<T>& mat) {
  matrix_ = mat;
}

template <typename T>
inline AMatrix<T>::AMatrix(const std::vector<T>& col) {
  matrix_ = arma::Mat<T>(col);
}

template <typename T>
inline AMatrix<T>::AMatrix(const absl::Span<T>& col) {
  matrix_ = arma::Mat<T>(col.data(), col.size(), 1);
}

template <typename T>
inline AMatrix<T>::AMatrix(const std::valarray<T>& va) {
  std::vector<T> v;
  v.reserve(va.size());
  v.assign(std::begin(va), std::end(va));
  matrix_ = arma::Mat<T>(v);
}

template <typename T>
inline AMatrix<T>::AMatrix(const std::vector<std::vector<T>>& vecOfCols) {
  // assumes all vectors all of equal length
  matrix_ = arma::Mat<T>(vecOfCols[0].size(), vecOfCols.size());
  for (size_t i = 0; i < vecOfCols.size(); i++) {
    matrix_.col(i) = arma::Col<T>(vecOfCols[i]);
  }
}

template <typename T>
inline AMatrix<T>::AMatrix(size_t rows, size_t cols, std::vector<T>&& data) {
  matrix_ = arma::Mat<T>(&data[0], (arma::uword)rows, (arma::uword)cols, false);
}

template <typename T>
inline AMatrix<T>::AMatrix(size_t rows, size_t cols,
                           const std::vector<T>& data) {
  matrix_ = arma::Mat<T>(&data[0], (arma::uword)rows, (arma::uword)cols);
}

template <typename T>
inline AMatrix<T>::AMatrix(size_t rows, size_t cols) {
  matrix_ = arma::Mat<T>((arma::uword)rows, (arma::uword)cols);
}

template <typename T>
inline AMatrix<T>::AMatrix(arma::Mat<T>&& matrix) {
  matrix_ = std::move(matrix);
}

template <typename T>
AMatrix<T> AMatrix<T>::GetSpan(size_t rowStart, size_t rowEnd, size_t colStart,
                               size_t colEnd) const {
  arma::Mat<T> m = matrix_.submat(rowStart, colStart, rowEnd, colEnd);
  return AMatrix<T>(m);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::Filled(size_t rows, size_t cols, T initialValue) {
  arma::Mat<T> m((arma::uword)rows, (arma::uword)cols);
  m.fill(initialValue);
  AMatrix<T> am(std::move(m));
  return am;
}

template <typename T>
inline T& AMatrix<T>::operator()(size_t row, size_t column) {
  return matrix_((arma::uword)row, (arma::uword)column);
}

template <typename T>
inline T AMatrix<T>::operator()(size_t row, size_t column) const {
  return matrix_((arma::uword)row, (arma::uword)column);
}

template <typename T>
inline T& AMatrix<T>::operator()(size_t elementIndex) {
  return matrix_.at((arma::uword)elementIndex);
}

template <typename T>
inline T AMatrix<T>::operator()(size_t elementIndex) const {
  return matrix_((arma::uword)elementIndex);
}

template <typename T>
inline AMatrix<T>& AMatrix<T>::operator=(const AMatrix<T>& other) {
  matrix_ = other.matrix_;
  return *this;
}

template <typename T>
inline bool AMatrix<T>::operator==(const AMatrix<T>& other) const {
  if (matrix_.n_rows != other.matrix_.n_rows) return false;
  if (matrix_.n_cols != other.matrix_.n_cols) return true;
  return std::equal(matrix_.cbegin(), matrix_.cend(), other.matrix_.cbegin());
  // Why doesn't == work when documentation says it does? matrix_ ==
  // other.matrix_;
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator+(const AMatrix<T>& other) const {
  return AMatrix<T>(matrix_ + other.matrix_);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator+(T v) const {
  return AMatrix<T>(matrix_ + v);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator*(T v) const {
  return AMatrix<T>(matrix_ * v);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator-(T v) const {
  return AMatrix<T>(matrix_ - v);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator/(T v) const {
  return AMatrix<T>(matrix_ / v);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::operator-(const AMatrix<T>& m) const {
  return AMatrix<T>(matrix_ - m.matrix_);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::PointWiseProduct(const AMatrix<T>& m) const {
  return AMatrix<T>(std::move(matrix_ % m.matrix_));
}

template <typename T>
inline AMatrix<T> AMatrix<T>::PointWiseDivide(const AMatrix<T>& m) const {
  return AMatrix<T>(std::move(matrix_ / m.matrix_));
}

template <typename T>
inline AMatrix<T> AMatrix<T>::Transpose() const {
  return AMatrix<T>{std::move(trans(matrix_))};
}

template <typename T>
inline std::vector<T> AMatrix<T>::RowSubset(size_t rowIndex,
                                            size_t startColumnIndex,
                                            size_t endColumnIndex) const {
  std::vector<T> vec(arma::conv_to<std::vector<T>>::from(
      matrix_.row(rowIndex).subvec(startColumnIndex, endColumnIndex)));
  return vec;
}

template <typename T>
inline size_t AMatrix<T>::LongestDimensionLength() const {
  return (matrix_.n_rows > matrix_.n_cols) ? matrix_.n_rows : matrix_.n_cols;
}

template <>
inline AMatrix<double> AMatrix<double>::Abs() const {
  AMatrix<double> absMatrix(matrix_.n_rows, matrix_.n_cols);
  for (size_t i = 0; i < absMatrix.NumElements(); i++) {
    absMatrix.matrix_(i) = std::abs(matrix_(i));
  }
  return absMatrix;
}

template <>
inline AMatrix<double> AMatrix<std::complex<double>>::Abs() const {
  arma::Mat<double> absMatrix;
  absMatrix.set_size(matrix_.n_rows, matrix_.n_cols);
  for (size_t i = 0; i < absMatrix.n_elem; i++) {
    absMatrix(i) = std::abs(matrix_(i));
  }
  return AMatrix<double>(std::move(absMatrix));
}

template <typename T>
inline std::vector<T> AMatrix<T>::GetRow(size_t row) const {
  std::vector<T> vec(arma::conv_to<std::vector<T>>::from(matrix_.row(row)));
  return vec;
}

template <typename T>
inline AMatrix<T> AMatrix<T>::GetRows(size_t rowStart, size_t rowEnd) const {
  return AMatrix<T>(std::move(matrix_.rows(rowStart, rowEnd)));
}

template <typename T>
inline AMatrix<T> AMatrix<T>::GetColumn(size_t column) const {
  return AMatrix<T>(matrix_.cols(column, column));
}

template <typename T>
inline AMatrix<T> AMatrix<T>::GetColumns(size_t colStart, size_t colEnd) const {
  return AMatrix<T>(matrix_.cols(colStart, colEnd));
}

template <typename T>
inline void AMatrix<T>::SetRow(size_t rowIndex, const std::vector<T>& row) {
  matrix_.row(rowIndex) = std::move(arma::Row<T>(row));
}

template <typename T>
inline void AMatrix<T>::SetColumn(size_t colIndex, AMatrix<T>&& col) {
  matrix_.col(colIndex) = std::move(col.GetArmaMat());
}

template <typename T>
inline void AMatrix<T>::SetColumn(size_t colIndex, std::vector<T>&& col) {
  matrix_.col(colIndex) = std::move(arma::Col<T>(col));
}

template <typename T>
inline void AMatrix<T>::SetColumn(size_t colIndex, std::valarray<T>&& col) {
  std::vector<T> v;
  v.reserve(col.size());
  v.assign(std::begin(col), std::end(col));
  matrix_.col(colIndex) = std::move(arma::Col<T>(v));
}

template <typename T>
inline void AMatrix<T>::SetRow(size_t rowIndex, std::valarray<T>&& row) {
  std::vector<T> v;
  v.reserve(row.size());
  v.assign(std::begin(row), std::end(row));
  matrix_.row(rowIndex) = std::move(arma::Row<T>(v));
}

template <typename T>
void AMatrix<T>::Print() const {
  printf("\n");
  matrix_.print();
}

template <>
void AMatrix<std::complex<double>>::PrintSummary(const char* s) const {
  printf("%s\n", s);
  size_t outMatSize = matrix_.n_rows;
  printf("first five\n");
  for (size_t i = 0; i < 5; i++) {
    printf("complex[%2zu] = %9.20f , %9.20f\n", i, matrix_(i).real(),
           matrix_(i).imag());
  }
  printf("middle \n");
  for (size_t i = (outMatSize / 2) - 4; i < (outMatSize / 2) + 6; i++) {
    printf("complex[%2zu] = %9.20f , %9.20f\n", i, matrix_(i).real(),
           matrix_(i).imag());
  }
  printf("last five\n");
  for (size_t i = outMatSize - 6; i < outMatSize; i++) {
    printf("complex[%2zu] = %9.20f , %9.20f\n", i, matrix_(i).real(),
           matrix_(i).imag());
  }
}

template <>
void AMatrix<double>::PrintSummary(const char* s) const {
  printf("%s\n", s);
  size_t outMatSize = matrix_.n_rows;
  printf("first five\n");
  for (size_t i = 0; i < 5; i++) {
    printf("double[%2zu] = %9.20f\n", i, matrix_(i));
  }
  printf("middle \n");
  for (size_t i = (outMatSize / 2) - 4; i < (outMatSize / 2) + 6; i++) {
    printf("double[%2zu] = %9.20f\n", i, matrix_(i));
  }
  printf("last five\n");
  for (size_t i = outMatSize - 6; i < outMatSize; i++) {
    printf("double[%2zu] = %9.20f\n", i, matrix_(i));
  }
}

template <typename T>
inline const T* AMatrix<T>::MemPtr() const {
  return matrix_.memptr();
}

template <typename T>
inline void AMatrix<T>::Resize(const size_t rows, const size_t cols) {
  matrix_.resize(rows, cols);
}

template <typename T>
inline AMatrix<T> AMatrix<T>::JoinVertically(const AMatrix<T>& other) const {
  // Having a separate line for return is needed to cast to Mat
  arma::Mat<T> m = arma::join_vert(matrix_, other.GetArmaMat());
  return m;
}

template <typename T>
inline AMatrix<T> AMatrix<T>::FlipUpDown() const {
  return AMatrix<T>(arma::flipud(matrix_));
}

template <typename T>
inline AMatrix<T> AMatrix<T>::Mean(kDimension dim) const {
  return AMatrix<T>(arma::mean(matrix_, static_cast<int>(dim)));
}

template <typename T>
inline AMatrix<double> AMatrix<T>::StdDev(kDimension dim) const {
  // The second parameter of arma::stddev is norm_type.
  // norm_type=0 means that stddev should use the unbiased estimate.
  return AMatrix<double>(arma::stddev(matrix_, 0, static_cast<int>(dim)));
}

template <typename T>
inline const arma::Mat<T>& AMatrix<T>::GetArmaMat() const {
  return matrix_;
}

template <typename T>
inline std::vector<T> AMatrix<T>::ToVector() const {
  return arma::conv_to<std::vector<T>>::from(matrix_.col(0));
}

template <typename T>
inline std::valarray<T> AMatrix<T>::ToValArray() const {
  std::vector<T> v = arma::conv_to<std::vector<T>>::from(matrix_.col(0));
  return std::valarray<T>(v.data(), v.size());
}

template <>
inline std::vector<std::complex<double>>
AMatrix<std::complex<double>>::ToVector() const {
  return arma::conv_to<std::vector<std::complex<double>>>::from(matrix_.col(0));
}

template <typename T>
inline size_t AMatrix<T>::NumRows() const {
  return matrix_.n_rows;
}
template <typename T>
inline size_t AMatrix<T>::NumCols() const {
  return matrix_.n_cols;
}
template <typename T>
inline size_t AMatrix<T>::NumElements() const {
  return matrix_.n_elem;
}

template <typename T>
inline typename AMatrix<T>::iterator AMatrix<T>::begin() {
  return matrix_.begin();
}
template <typename T>
inline typename AMatrix<T>::iterator AMatrix<T>::end() {
  return matrix_.end();
}
template <typename T>
inline typename AMatrix<T>::const_iterator AMatrix<T>::cbegin() const {
  return matrix_.cbegin();
}
template <typename T>
inline typename AMatrix<T>::const_iterator AMatrix<T>::cend() const {
  return matrix_.cend();
}

template <typename T>
inline const T* AMatrix<T>::data() const {
  return matrix_.mem;
}
template <typename T>
inline T* AMatrix<T>::mutData() const {
  return const_cast<T*>(matrix_.mem);
}

template class AMatrix<double>;
template class AMatrix<std::complex<double>>;

}  // namespace Visqol
