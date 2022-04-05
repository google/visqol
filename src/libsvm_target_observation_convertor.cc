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

#include "libsvm_target_observation_convertor.h"

#include <stdlib.h>

#include <vector>

#include "svm.h"

namespace Visqol {
svm_node** LibSvmTargetObservationConvertor::ConvertObservations(
    const std::vector<MlObservation>& observations, size_t num_features) const {
  svm_node** svm_observations = reinterpret_cast<svm_node**>(
      malloc(sizeof(svm_node*) * observations.size()));
  for (size_t row_i = 0; row_i < observations.size(); row_i++) {
    svm_node* row = reinterpret_cast<svm_node*>(
        malloc(sizeof(svm_node) * (num_features + 1)));
    for (size_t col_i = 0; col_i < num_features; col_i++) {
      row[col_i].index = col_i + 1;
      row[col_i].value = observations[row_i][col_i];
    }
    row[num_features].index = -1;
    svm_observations[row_i] = row;
  }
  return svm_observations;
}

svm_node* LibSvmTargetObservationConvertor::ConvertObservation(
    const MlObservation& observation) const {
  size_t num_features = observation.size();
  // +1 to avoid accessing out-of-bounds index below
  svm_node* obs_node = reinterpret_cast<svm_node*>(
      malloc(sizeof(svm_node) * (num_features + 1)));
  for (size_t row_i = 0; row_i < num_features; row_i++) {
    // Follow the format described in the libsvm FAQ and README:
    // The indices are 1-indexed.
    // The model file must match this.  This allows the svm-predict
    // binary to work with our model.
    obs_node[row_i].index = row_i + 1;
    obs_node[row_i].value = observation[row_i];
  }
  obs_node[num_features].index = -1;
  return obs_node;
}
}  // namespace Visqol
