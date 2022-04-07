/*
 * Copyright 2021 Google LLC, Andrew Hines
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

#include "complex_valarray.h"

namespace Visqol {

ComplexValArray operator*(double x, const ComplexValArray& v) { return v * x; }

ComplexValArray operator+(double x, const ComplexValArray& v) { return v + x; }

ComplexValArray operator*(std::complex<double> x, const ComplexValArray& v) {
  return v * x;
}

ComplexValArray operator/(double d, const ComplexValArray& v) {
  ComplexValArray out(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    out[i] = d / v.at(i);
  }
  return out;
}
}  // namespace Visqol
