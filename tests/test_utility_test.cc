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

#include "test_utility.h"

#include "amatrix.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::HasSubstr;

namespace Visqol {
namespace {

constexpr double kTolerance = 0.00000001;

// Test the matrix comparator function
TEST(DoubleMatrixComparison, DifferentColumnCount) {
  std::string fail_msg;
  const AMatrix<double> different_cols_a(2, 2, std::vector<double>{0, 0, 0, 0});
  const AMatrix<double> different_cols_b(4, 1, std::vector<double>{0, 0, 0, 0});
  EXPECT_FALSE(CompareDoubleMatrix(different_cols_a, different_cols_b,
                                   kTolerance, &fail_msg));
  EXPECT_THAT(fail_msg, HasSubstr("num cols"));
}

TEST(DoubleMatrixComparison, DifferentRowCount) {
  std::string fail_msg;
  const AMatrix<double> different_rows_a{std::valarray<double>{0, 0, 0, 0, 0}};
  const AMatrix<double> different_rows_b{std::valarray<double>{0, 0, 0, 0}};
  EXPECT_FALSE(CompareDoubleMatrix(different_rows_a, different_rows_b,
                                   kTolerance, &fail_msg));
  EXPECT_THAT(fail_msg, HasSubstr("num rows"));
}

TEST(DoubleMatrixComparison, DifferentContents) {
  std::string fail_msg;
  const AMatrix<double> different_contents_a{std::valarray<double>{0, 0, 0, 0}};
  const AMatrix<double> different_contents_b{std::valarray<double>{0, 0, 1, 0}};
  EXPECT_FALSE(CompareDoubleMatrix(different_contents_a, different_contents_b,
                                   kTolerance, &fail_msg));
  EXPECT_THAT(fail_msg, HasSubstr("abs val"));
}

TEST(DoubleMatrixComparison, SameContents) {
  std::string fail_msg;
  const AMatrix<double> same_contents_a{
      std::valarray<double>{-10.2, 0, 23.1, 0}};
  const AMatrix<double> same_contents_b{
      std::valarray<double>{-10.2, 0, 23.1, 0}};
  EXPECT_TRUE(CompareDoubleMatrix(same_contents_a, same_contents_b, kTolerance,
                                  &fail_msg));
  EXPECT_EQ(fail_msg.size(), 0);
}

TEST(ComplexMatrixComparison, DifferentContents) {
  std::string fail_msg;
  const AMatrix<std::complex<double>> different_contents_comp_a{
      std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {-2, 0.2}}};
  const AMatrix<std::complex<double>> different_contents_comp_b{
      std::valarray<std::complex<double>>{{2, 4}, {5, -0.1}, {-2, 0.2}}};
  EXPECT_FALSE(CompareComplexMatrix(different_contents_comp_a,
                                    different_contents_comp_b, kTolerance,
                                    &fail_msg));
  EXPECT_THAT(fail_msg, HasSubstr("abs val"));
}

TEST(ComplexMatrixComparison, SameContents) {
  std::string fail_msg;

  const AMatrix<std::complex<double>> same_a{
      std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {-2, 0.2}}};
  const AMatrix<std::complex<double>> same_b{
      std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {-2, 0.2}}};
  EXPECT_TRUE(CompareComplexMatrix(same_a, same_b, kTolerance, &fail_msg));
  EXPECT_EQ(fail_msg.size(), 0);
}

TEST(ComplexMatrixComparison, DifferentRealSign) {
  std::string fail_msg;

  const AMatrix<std::complex<double>> different_sign_real_comp_a{
      std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {-2, 0.2}}};
  const AMatrix<std::complex<double>> different_sign_real_comp_b{
      std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {2, 0.2}}};
  EXPECT_FALSE(CompareComplexMatrix(different_sign_real_comp_a,
                                    different_sign_real_comp_b, kTolerance,
                                    &fail_msg));
  EXPECT_THAT(fail_msg, HasSubstr("abs val"));
}

TEST(ComplexMatrixComparison, DifferentImaginarySign) {
  std::string fail_msg;
  const AMatrix<std::complex<double>> different_sign_imag_comp_a{
      std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {-2, 0.2}}};
  const AMatrix<std::complex<double>> different_sign_imag_comp_b{
      std::valarray<std::complex<double>>{{2, -4}, {3, -0.1}, {-2, 0.2}}};
  EXPECT_FALSE(CompareComplexMatrix(different_sign_imag_comp_a,
                                    different_sign_imag_comp_b, kTolerance,
                                    &fail_msg));
  EXPECT_THAT(fail_msg, HasSubstr("abs val"));
}

}  // namespace
}  // namespace Visqol
