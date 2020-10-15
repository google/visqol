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

#ifndef VISQOL_INCLUDE_SIMILARITYTOQUALITYMAPPER_H
#define VISQOL_INCLUDE_SIMILARITYTOQUALITYMAPPER_H

#include <vector>

#include "absl/status/status.h"

namespace Visqol {

/**
 * This base class declares the functions necessary to produce a quality score
 * from an NSIM similarity score.
 */
class SimilarityToQualityMapper {
 public:
  /**
   * Destructor for the similarity to quality mapper.
   */
  virtual ~SimilarityToQualityMapper() {}

  /**
   * Map a vector of quality measures across frequency bands to a MOSLQO.
   *
   * @param similarity_vector A vector of NSIMS scores, each for a different
   *    frequency band.
   *
   * @return A MOSLQO score (1 to 5).
   */
  virtual double PredictQuality(
      const std::vector<double> &similarity_vector) const = 0;

  /**
   * Initializes the similarity to quality mapper.
   *
   * @return An OK status if the model was successfully initialized with the
   * model file. Else, an error status is returned.
   */
  virtual absl::Status Init() = 0;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_SIMILARITYTOQUALITYMAPPER_H
