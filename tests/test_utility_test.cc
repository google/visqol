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

#include "gtest/gtest.h"

#include "amatrix.h"

namespace Visqol {
namespace {

// Test the matrix comparator function
TEST(TestUtility, MatrixComparison) {
  const double kTolerance = 0.00000001;
  std::string fail_msg;

  // Test for different col count
  const AMatrix<double> different_cols_a(2, 2, std::vector<double>{0, 0, 0, 0});
  const AMatrix<double> different_cols_b(4, 1, std::vector<double>{0, 0, 0, 0});
  const std::string expected_fail_cols =
      "Matrices do not match!"
      " matrix_a num cols: 2 matrix_b num cols: 1";
  CompareDoubleMatrix(different_cols_a, different_cols_b, kTolerance,
      &fail_msg);
  ASSERT_TRUE(expected_fail_cols == fail_msg);

  // Test for different row count
  const AMatrix<double> different_rows_a{std::valarray<double>{0, 0, 0, 0, 0}};
  const AMatrix<double> different_rows_b{std::valarray<double>{0, 0, 0, 0}};
  const std::string expected_fail_rows =
      "Matrices do not match!"
      " matrix_a num rows: 5 matrix_b num rows: 4";
  CompareDoubleMatrix(different_rows_a, different_rows_b, kTolerance,
      &fail_msg);
  ASSERT_TRUE(expected_fail_rows == fail_msg);

  // Test for different contents
  const AMatrix<double> different_contents_a{std::valarray<double>{0, 0, 0, 0}};
  const AMatrix<double> different_contents_b{std::valarray<double>{0, 0, 1, 0}};
  const std::string expected_fail_contents =
      "Matrices do not match!"
      " At index 2 matrix_a abs val: 0.000000 matrix_b abs val: 1.000000";
  CompareDoubleMatrix(different_contents_a, different_contents_b,
      kTolerance, &fail_msg);
  ASSERT_TRUE(expected_fail_contents == fail_msg);

  // Test for different contents - complex number
  const AMatrix<std::complex<double>> different_contents_comp_a
      {std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {-2, 0.2}}};
  const AMatrix<std::complex<double>> different_contents_comp_b
      {std::valarray<std::complex<double>>{{2, 4}, {5, -0.1}, {-2, 0.2}}};
  ASSERT_FALSE(CompareComplexMatrix(different_contents_comp_a,
      different_contents_comp_b, kTolerance, &fail_msg));

  // Test for different sign on real part
  const AMatrix<std::complex<double>> different_sign_real_comp_a
      {std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {-2, 0.2}}};
  const AMatrix<std::complex<double>> different_sign_real_comp_b
      {std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {2, 0.2}}};
  ASSERT_FALSE(CompareComplexMatrix(different_sign_real_comp_a,
      different_sign_real_comp_b, kTolerance, &fail_msg));

  // Test for different sign on imag part
  const AMatrix<std::complex<double>> different_sign_imag_comp_a
      {std::valarray<std::complex<double>>{{2, 4}, {3, -0.1}, {-2, 0.2}}};
  const AMatrix<std::complex<double>> different_sign_imag_comp_b
      {std::valarray<std::complex<double>>{{2, -4}, {3, -0.1}, {-2, 0.2}}};
  ASSERT_FALSE(CompareComplexMatrix(different_sign_imag_comp_a,
      different_sign_imag_comp_b, kTolerance, &fail_msg));
}

}  // namespace
}  // namespace Visqol
