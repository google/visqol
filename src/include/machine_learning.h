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

#ifndef VISQOL_INCLUDE_MACHINELEARNING_H
#define VISQOL_INCLUDE_MACHINELEARNING_H

#include <vector>

namespace Visqol {
/**
 * Type alias for SVM Observations.
 */
using MlObservation = std::vector<double>;

/**
 * Type alias for an SVM Target.
 */
using MlTarget = double;
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_MACHINELEARNING_H
