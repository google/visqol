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
#include <vector>

#include "absl/memory/memory.h"
#include "gtest/gtest.h"

#include "test_utility.h"

namespace Visqol {
namespace {

const double kTolerance = 0.001;

TEST(AMatrix, OperatorPlusTRef) {
  const double kVal = 2.0;  
  const size_t kRows = 2;
  const size_t kCols = 2;
  const std::vector<double> inputs = {kVal, kVal, kVal, kVal};

  // Create the original matrix.
  AMatrix<double> mat{kRows, kCols, inputs};

  // Plus T
  auto plus_t = mat + kVal;

  // Ensure that every element in the new matrix is the old value plus the kVal.
  EXPECT_EQ(inputs.size(), plus_t.NumElements());
  EXPECT_EQ(kRows, plus_t.NumRows());
  EXPECT_EQ(kCols, plus_t.NumCols());
  for (auto it = plus_t.cbegin(); it != plus_t.cend(); it++)
  {
    EXPECT_DOUBLE_EQ(kVal + kVal, *it);
  }
}

TEST(AMatrix, OperatorPlusTPtr) {
  const double kVal = 2.0;  
  const size_t kRows = 2;
  const size_t kCols = 2;
  const std::vector<double> inputs = {kVal, kVal, kVal, kVal};

  // Create the original matrix.
  AMatrix<double> mat{kRows, kCols, inputs};

  // Plus T
  auto plus_t = mat + &kVal;

  // Ensure that every element in the new matrix is the old value plus the kVal.
  EXPECT_EQ(inputs.size(), plus_t->NumElements());
  EXPECT_EQ(kRows, plus_t->NumRows());
  EXPECT_EQ(kCols, plus_t->NumCols());
  for (auto it = plus_t->cbegin(); it != plus_t->cend(); it++)
  {
    EXPECT_DOUBLE_EQ(kVal + kVal, *it);
  }
}

TEST(AMatrix, PointWiseProductRef) {
  const double kVal = 2.0;  
  const size_t kRows = 2;
  const size_t kCols = 2;
  const std::vector<double> inputs = {kVal, kVal, kVal, kVal};

  // Create the original matrix.
  AMatrix<double> mat{kRows, kCols, inputs};

  // Multiply the original matrix by itself.
  auto ppw = mat.PointWiseProduct(mat);

  // Ensure the point wise product matrix is correct.
  EXPECT_EQ(inputs.size(), ppw.NumElements());
  EXPECT_EQ(kRows, ppw.NumRows());
  EXPECT_EQ(kCols, ppw.NumCols());
  for (auto it = ppw.cbegin(); it != ppw.cend(); it++)
  {
    EXPECT_DOUBLE_EQ(kVal * kVal, *it);
  }
}

TEST(AMatrix, PointWiseProductPtrSameSize) {
  const double kVal = 2.0;  
  const size_t kRows = 2;
  const size_t kCols = 2;
  const std::vector<double> inputs = {kVal, kVal, kVal, kVal};

  // Create the original matrix.
  auto mat = absl::make_unique<AMatrix<double>>(kRows, kCols, inputs);

  // Multiply the original matrix by itself.
  auto ppw = mat->PointWiseProduct(mat);

  // Ensure the point wise product matrix is correct.
  EXPECT_EQ(inputs.size(), ppw->NumElements());
  EXPECT_EQ(kRows, ppw->NumRows());
  EXPECT_EQ(kCols, ppw->NumCols());
  for (auto it = ppw->cbegin(); it != ppw->cend(); it++)
  {
    EXPECT_DOUBLE_EQ(kVal * kVal, *it);
  }
}

TEST(AMatrix, PointWiseProductPtrLargerInput) {
  const double kVal = 2.0;  
  const size_t kRows = 2;
  const size_t kCols = 2;
  const std::vector<double> inputs = {kVal, kVal, kVal, kVal};
  const size_t kRowsLarger = 3;
  const size_t kColsLarger = 3;
  const std::vector<double> inputsLarger = {kVal, kVal, kVal, kVal, kVal, kVal, kVal, kVal, kVal};

  // Create the original matrix.
  auto mat = absl::make_unique<AMatrix<double>>(kRows, kCols, inputs);

  // Create the larger input matrix
  auto mat_larger = absl::make_unique<AMatrix<double>>(kRowsLarger, kColsLarger, inputsLarger);

  // Multiply the original matrix by the larger one.
  auto ppw = mat->PointWiseProduct(mat_larger);

  // Ensure the point wise product matrix is correct.
  EXPECT_EQ(inputs.size(), ppw->NumElements());
  EXPECT_EQ(kRows, ppw->NumRows());
  EXPECT_EQ(kCols, ppw->NumCols());
  for (auto it = ppw->cbegin(); it != ppw->cend(); it++)
  {
    EXPECT_DOUBLE_EQ(kVal * kVal, *it);
  }
}

TEST(AMatrix, PointWiseProductPtrSmallerInput) {
  const double kVal = 2.0;
  const size_t kRows = 3;
  const size_t kCols = 3;
  const std::vector<double> inputs = {kVal, kVal, kVal, kVal, kVal, kVal, kVal, kVal, kVal};
  const size_t kRowsSmaller = 2;
  const size_t kColsSmaller = 2;
  const std::vector<double> inputsSmaller = {kVal, kVal, kVal, kVal};
  const std::vector<double> expectedPpw = {kVal*kVal, kVal*kVal, kVal, kVal*kVal, kVal*kVal, kVal, kVal, kVal, kVal};

  // Create the original matrix.
  auto mat = absl::make_unique<AMatrix<double>>(kRows, kCols, inputs);

  // Create the smaller input matrix
  auto mat_smaller = absl::make_unique<AMatrix<double>>(kRowsSmaller, kColsSmaller, inputsSmaller);

   // Create the expected ppw matrix
  auto mat_expected = absl::make_unique<AMatrix<double>>(kRows, kCols, expectedPpw);

  // Multiply the original matrix by the smaller one.
  auto ppw = mat->PointWiseProduct(mat_smaller);

  // Ensure the point wise product matrix is correct.
  std::string fail_msg;
  ASSERT_TRUE(CompareDoubleMatrix(*mat_expected, *ppw, kTolerance, &fail_msg));
}

}  // namespace
}  // namespace Visqol
