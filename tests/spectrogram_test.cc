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

#include "spectrogram.h"

#include "gtest/gtest.h"
#include "test_utility.h"

namespace Visqol {
namespace {

const double kTolerance = 0.0001;

// The smallest element to include in the test matrix.
const double kMinElem = -53.2;

// The floor to subtract from each element in the test matrix.
const double kFloor = 0.1;

// The test matrix. Elements randomly choosen, but with at least one zero value.
const AMatrix<double> k10ElemMat{std::valarray<double>{
    10.21, -4.63, 0.54, 87.98, 0.065, 0, kMinElem, 8.7, 0, -2.76}};

// The test matrix converted to DB.
const AMatrix<double> k10ElemMatToDB{
    std::valarray<double>{10.0903, 6.6558, -2.6761, 19.4438, -11.8709,
                          -156.5356, 17.2591, 9.3952, -156.5356, 4.4091}};

// The test matrix with the floor subtracted from each element.
const AMatrix<double> k10ElemMatSubFloor{std::valarray<double>{
    10.21 - kFloor, -4.63 - kFloor, 0.54 - kFloor, 87.98 - kFloor,
    0.065 - kFloor, 0 - kFloor, kMinElem - kFloor, 8.7 - kFloor, 0 - kFloor,
    -2.76 - kFloor}};

// Ensure the a matrix with at least one element of value 0 can be successfully
// converted to DB.
TEST(SpectrogramTest, ConvertToDbTest) {
  AMatrix<double> mat_10_elem{k10ElemMat};
  Spectrogram spectro{std::move(mat_10_elem)};
  spectro.ConvertToDb();

  std::string fail_msg;
  ASSERT_TRUE(CompareDoubleMatrix(k10ElemMatToDB, spectro.Data(), kTolerance,
                                  &fail_msg))
      << fail_msg;
}

// Ensure that the Spectrogram can successfully return the lowest value element.
TEST(SpectrogramTest, MinimumTest) {
  AMatrix<double> mat_10_elem{k10ElemMat};
  Spectrogram spectro{std::move(mat_10_elem)};
  ASSERT_NEAR(kMinElem, spectro.Minimum(), kTolerance);
}

// Ensure that a floor value can successfully be subtracted from the
// spectrogram.
TEST(SpectrogramTest, SubtractFloorTest) {
  AMatrix<double> mat_10_elem{k10ElemMat};
  Spectrogram spectro{std::move(mat_10_elem)};
  spectro.SubtractFloor(kFloor);

  std::string fail_msg;
  ASSERT_TRUE(CompareDoubleMatrix(k10ElemMatSubFloor, spectro.Data(),
                                  kTolerance, &fail_msg))
      << fail_msg;
}

}  // namespace
}  // namespace Visqol
