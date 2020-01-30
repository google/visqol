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

#ifndef VISQOL_INCLUDE_MISCMATH_H
#define VISQOL_INCLUDE_MISCMATH_H

#include <vector>

#include "amatrix.h"

namespace Visqol {
class MiscMath {
 public:
  /**
   * Finds the next power of two from an integer. If the input is already a
   * power of two, it is returned. This method works with values representable
   * by unsigned 32 bit integers.
   *
   * This method was adopted from the ResonanceAudio project:
   * https://github.com/resonance-audio/resonance-audio
   *
   * @param input Integer value.
   *
   * @return The next power of two, or original input if already power of two.
   */
  static size_t NextPowTwo(const uint32_t input);

  static AMatrix<double> Normalize(const AMatrix<double>& mat);
  static AMatrix<double> Sum(const AMatrix<double>& mat);
  static AMatrix<double> Mean(const AMatrix<double>& mat);
  static std::vector<double> NormalizeInt16ToDouble(std::vector<int16_t>
    &input_vec);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_MISCMATH_H
