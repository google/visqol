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

#ifndef VISQOL_INCLUDE_SVRSIMILARITYTOQUALITYMAPPER_H
#define VISQOL_INCLUDE_SVRSIMILARITYTOQUALITYMAPPER_H

#include <vector>

#include "absl/status/status.h"
#include "file_path.h"
#include "similarity_to_quality_mapper.h"
#include "support_vector_regression_model.h"

namespace Visqol {
/**
 * This class utilises a Support Vector Regression machine learning model to
 * produce a quality score from a similarity score, that was produced through
 * spectrogram comparison.
 */
class SvrSimilarityToQualityMapper : public SimilarityToQualityMapper {
 public:
  /**
   * Construct the mapper utilising the given SVR model file.
   *
   * @param support_vector_model The filepath to the SVR model file.
   */
  explicit SvrSimilarityToQualityMapper(const FilePath& support_vector_model);

  // Docs inherited from parent.
  double PredictQuality(
      const std::vector<double>& fvnsim_vector,
      const std::vector<double>& fvnsim10_vector,
      const std::vector<double>& fstdnsim_vector,
      const std::vector<double>& fvdegenergy_vector) const override;

  // Docs inherited from parent.
  absl::Status Init() override;

 private:
  /**
   * The SVR model used for predictions.
   */
  SupportVectorRegressionModel model_;

  /**
   * The filepath to the SVR model file, used for instantiating the SVR model.
   */
  FilePath model_path_;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_SVRSIMILARITYTOQUALITYMAPPER_H
