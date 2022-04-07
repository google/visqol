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

#ifndef VISQOL_INCLUDE_SPEECHSIMILARITYTOQUALITYMAPPER_H
#define VISQOL_INCLUDE_SPEECHSIMILARITYTOQUALITYMAPPER_H

#include <vector>

#include "absl/status/status.h"
#include "similarity_to_quality_mapper.h"
#include "support_vector_regression_model.h"

namespace Visqol {
/**
 * This class is used to produce a quality score from a similarity score for
 * speech audio signals. This mapping from similarity to quality is performed
 * using a polynomial mapping.
 */
class SpeechSimilarityToQualityMapper : public SimilarityToQualityMapper {
 public:
  /**
   * Constructor that takes a bool that controls whether to use scaled or
   * unscaled mapping.
   *
   * @param scale_to_max_mos If true, perfect NSIM scores of 1.0 will be scaled
   * to a MOS-LQO of 5.0. If false, perfect NSIM scores will instead be scaled
   * to ~4.x.
   */
  explicit SpeechSimilarityToQualityMapper(bool scale_to_max_mos);

  // Docs inherited from parent.
  double PredictQuality(
      const std::vector<double>& fvnsim_vector,
      const std::vector<double>& fvnsim10_vector,
      const std::vector<double>& fstdnsim_vector,
      const std::vector<double>& fvdegenergy_vector) const override;

  // Docs inherited from parent.
  absl::Status Init() override;

 protected:
  /**
   * If true, perfect NSIM scores of 1.0 will be scaled to a MOS-LQO of 5.0. If
   * false, perfect NSIM scores will instead be scaled to ~4.x.
   */
  bool scale_to_max_mos_ = true;
};

}  // namespace Visqol
#endif  // VISQOL_INCLUDE_SPEECHSIMILARITYTOQUALITYMAPPER_H
