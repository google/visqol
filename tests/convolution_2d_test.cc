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

#include "gtest/gtest.h"
#include "test_utility.h"

namespace Visqol {
namespace {

const double kTolerance = 0.001;

/**
 * Test the 3x3 valid padded 2D convolution logic.
 */
TEST(Convolution2D, 3x3_2d_conv_test) {
  const std::vector<double> w = {
      0.0113033910173052, 0.0838251475442633, 0.0113033910173052,
      0.0838251475442633, 0.619485845753726,  0.0838251475442633,
      0.0113033910173052, 0.0838251475442633, 0.0113033910173052};
  AMatrix<double> window(3, 3, std::move(w));

  std::vector<double> m{40.0392, 43.3409, 39.5270, 41.1731, 41.3591,
                        42.6852, 45.2083, 45.7769, 39.9689, 43.6190,
                        41.0119, 40.4244, 41.5932, 43.6027, 42.6204,
                        43.0624, 42.2610, 42.4725, 43.4258, 42.9079};
  AMatrix<double> matrix(5, 4, std::move(m));

  std::vector<double> r{40.6634, 42.8407, 40.6395, 41.0129, 41.5407,
                        42.4677, 44.2760, 44.2031, 41.2263, 42.9752,
                        41.3784, 41.2656, 42.1388, 43.0366, 42.8042,
                        42.7613, 42.1817, 42.4590, 43.2709, 42.9377};
  AMatrix<double> expected_result(5, 4, std::move(r));

  auto conv_2d_res =
      Convolution2D<double>::Valid2DConvWithBoundary(window, matrix);
  std::string fail_msg;
  ASSERT_TRUE(
      CompareDoubleMatrix(expected_result, conv_2d_res, kTolerance, &fail_msg));
}

}  // namespace
}  // namespace Visqol
