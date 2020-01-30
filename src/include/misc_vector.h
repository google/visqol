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

#ifndef VISQOL_INCLUDE_MISCVECTOR_H
#define VISQOL_INCLUDE_MISCVECTOR_H

#include <fstream>
#include <string>
#include <vector>

#include "amatrix.h"

namespace Visqol {

class MiscVector {
 public:
  static double Sum(const AMatrix<double>& mat);
  static double Mean(const AMatrix<double>& mat);

  static std::vector<double> ConvertVecOfVecToVec(
      const std::vector<std::vector<double>>& mat) {
    std::vector<double> targets_vec;
    targets_vec.reserve(mat.size());
    for (size_t i = 0; i < mat.size(); i++) {
      targets_vec.push_back(mat[i][0]);
    }
    return targets_vec;
  }

  static std::vector<double> ReadVectorFromTxtFile(const std::string& path,
                                                   size_t num_samples) {
    std::fstream vec_file(path, std::ios_base::in);
    std::vector<double> v;
    v.reserve(num_samples);
    double d;
    while (vec_file >> d) {
      v.push_back(d);
    }
    vec_file.close();
    return v;
  }
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_MISCVECTOR_H
