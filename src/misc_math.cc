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

#include "misc_math.h"

#include <algorithm>
#include <vector>

namespace Visqol {
AMatrix<double> MiscMath::Normalize(const AMatrix<double>& m) {
  double maxValue = *std::max_element(m.cbegin(), m.cend());
  AMatrix<double> n(m.NumRows(), m.NumCols());
  for (size_t i = 0; i < m.NumElements(); i++) {
    n(i) = m(i) / maxValue;
  }
  return n;
}

size_t MiscMath::NextPowTwo(const uint32_t input) {
  uint32_t number = input - 1;
  number |= number >> 1;   // Take care of 2 bit numbers.
  number |= number >> 2;   // Take care of 4 bit numbers.
  number |= number >> 4;   // Take care of 8 bit numbers.
  number |= number >> 8;   // Take care of 16 bit numbers.
  number |= number >> 16;  // Take care of 32 bit numbers.
  number++;
  return static_cast<size_t>(number);
}

const float kScalar16bit = 32768.0;
static double abs_scale(int16_t x) { return (x / kScalar16bit); }
std::vector<double> MiscMath::NormalizeInt16ToDouble(
    std::vector<int16_t>& input_vec) {
  std::vector<double> output_vector(input_vec.size());
  std::transform(input_vec.begin(), input_vec.end(), output_vector.begin(),
                 abs_scale);
  return output_vector;
}

double MiscMath::ExponentialFromFit(double x, double a, double b, double x0) {
  return a + exp(b * (x - x0));
}

}  // namespace Visqol
