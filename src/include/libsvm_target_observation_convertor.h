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

#ifndef VISQOL_INCLUDE_LIBSVMTARGETOBSERVATIONCONVERTOR_H
#define VISQOL_INCLUDE_LIBSVMTARGETOBSERVATIONCONVERTOR_H

#include <cstddef>
#include <vector>

#include "machine_learning.h"
#include "svm.h"

namespace Visqol {
/**
 * This helper class is used for converting observations into LIBSVM format.
 */
class LibSvmTargetObservationConvertor {
 public:
  /**
   * For a given vector of observations, convert them into an array of LIBSVM
   * nodes.
   *
   * @param observations The vector of observations.
   * @param num_features The number of features in each observation. Assumes
   *    all observations have the same number of features.
   *
   * @return An array of pointers to LIBSVM nodes containing the observation
   * data.
   */
  svm_node** ConvertObservations(const std::vector<MlObservation>& observations,
                                 size_t num_features) const;

  /**
   * For a given observation, convert it into a LIBSVM node.
   *
   * @param observation The observation to convert.
   *
   * @return A pointer to a LIBSVM node containing the observation data.
   */
  svm_node* ConvertObservation(const MlObservation& observation) const;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_LIBSVMTARGETOBSERVATIONCONVERTOR_H
