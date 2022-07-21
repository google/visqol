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

#ifndef VISQOL_INCLUDE_AMATRIX_H
#define VISQOL_INCLUDE_AMATRIX_H

#include <armadillo>
#include <memory>
#include <valarray>
#include <vector>

#include "absl/types/span.h"
#define ARMA_64BIT_WORD

#if defined(_MSC_VER) && _MSC_VER >= 1400
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

namespace Visqol {
enum class kDimension { COLUMN = 0, ROW = 1 };

template <typename T>
class AMatrix;
template <class T>
using AMatrixUniq = std::unique_ptr<AMatrix<T>>;

// Entries are stored column-wise to be like Matlab
template <typename T>
class AMatrix {
 public:
  AMatrix<T>() {}
  AMatrix<T>(const arma::Mat<T>& mat);
  AMatrix<T>(const AMatrix<T>& other);
  AMatrix<T>(const std::vector<T>& col);
  AMatrix<T>(const absl::Span<T>& col);
  AMatrix<T>(const std::valarray<T>& va);
  AMatrix<T>(const std::vector<std::vector<T>>& vecOfCols);
  AMatrix<T>(size_t rows, size_t cols);
  AMatrix<T>(size_t rows, size_t cols, std::vector<T>&& data);
  AMatrix<T>(size_t rows, size_t cols, const std::vector<T>& data);
  AMatrix<T>(arma::Mat<T>&& matrix);  // could this be private?
  T& operator()(size_t row, size_t column);
  T operator()(size_t row, size_t column) const;
  T& operator()(size_t elementIndex);
  T operator()(size_t elementIndex) const;
  bool operator==(const AMatrix<T>& other) const;
  AMatrix<T>& operator=(const AMatrix<T>& other);
  AMatrix<T> operator+(const AMatrix<T>& other) const;
  AMatrix<T> operator+(T v) const;
  AMatrix<T> operator*(T v) const;
  AMatrix<T> operator-(T v) const;
  AMatrix<T> operator/(T v) const;
  AMatrix<T> operator-(const AMatrix<T>& m) const;
  static AMatrix<T> Filled(size_t rows, size_t cols, T initialValue);

  AMatrix<T> PointWiseProduct(const AMatrix<T>& m) const;
  AMatrix<T> PointWiseDivide(const AMatrix<T>& m) const;
  AMatrix<T> Transpose() const;
  AMatrix<double> Abs() const;
  std::vector<T> RowSubset(size_t row, size_t startColumnIndex,
                           size_t endColumnIndex) const;
  size_t LongestDimensionLength() const;

  std::vector<T> GetRow(size_t row) const;
  AMatrix<T> GetRows(size_t rowStart, size_t rowEnd) const;
  void SetRow(size_t rowIndex, const std::vector<T>& row);
  void SetRow(size_t rowIndex, std::valarray<T>&& row);
  AMatrix<T> GetColumn(size_t column) const;
  AMatrix<T> GetColumns(size_t colStart, size_t colEnd) const;
  void SetColumn(size_t colIndex, AMatrix<T>&& col);
  void SetColumn(size_t colIndex, std::vector<T>&& col);
  void SetColumn(size_t colIndex, std::valarray<T>&& col);
  AMatrix<T> GetSpan(size_t rowStart, size_t rowEnd, size_t colStart,
                     size_t colEnd) const;

  void Print() const;
  void PrintSummary(const char* s) const;
  const T* MemPtr() const;
  void Resize(const size_t rows, const size_t cols);
  AMatrix<T> JoinVertically(const AMatrix<T>& other) const;
  AMatrix<T> FlipUpDown() const;
  AMatrix<T> Mean(kDimension dim) const;
  AMatrix<double> StdDev(kDimension dim) const;

  size_t NumRows() const;
  size_t NumCols() const;
  size_t NumElements() const;

  std::vector<T> ToVector() const;
  std::valarray<T> ToValArray() const;

  // hack to access private member
  const arma::Mat<T>& GetArmaMat() const;

 public:
  typedef typename arma::Mat<T>::iterator iterator;
  typedef typename arma::Mat<T>::const_iterator const_iterator;
  iterator begin();
  iterator end();
  const_iterator cbegin() const;
  const_iterator cend() const;

  const T* data() const;
  T* mutData() const;  // bit of a hack

 private:
  arma::Mat<T> matrix_;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_AMATRIX_H
