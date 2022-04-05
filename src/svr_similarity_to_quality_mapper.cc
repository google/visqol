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

#include "svr_similarity_to_quality_mapper.h"

#include <vector>

#include "absl/status/status.h"
#include "file_path.h"

namespace Visqol {
SvrSimilarityToQualityMapper::SvrSimilarityToQualityMapper(
    const FilePath& support_vector_model)
    : model_path_{support_vector_model} {}

absl::Status SvrSimilarityToQualityMapper::Init() {
  return model_.Init(model_path_);
}

double SvrSimilarityToQualityMapper::PredictQuality(
    const std::vector<double>& fvnsim_vector,
    const std::vector<double>& fvnsim10_vector,
    const std::vector<double>& fstdnsim_vector,
    const std::vector<double>& fvdegenergy_vector) const {
  return std::max(1.0, std::min(5.0, model_.Predict(fvnsim_vector)));
}
}  // namespace Visqol
