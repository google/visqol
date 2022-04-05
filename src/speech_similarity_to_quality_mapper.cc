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

#include "speech_similarity_to_quality_mapper.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "absl/status/status.h"
#include "amatrix.h"
#include "misc_math.h"

namespace Visqol {
SpeechSimilarityToQualityMapper::SpeechSimilarityToQualityMapper(
    bool scale_to_max_mos)
    : scale_to_max_mos_{scale_to_max_mos} {}

absl::Status SpeechSimilarityToQualityMapper::Init() { return absl::Status(); }

double SpeechSimilarityToQualityMapper::PredictQuality(
    const std::vector<double>& fvnsim_vector,
    const std::vector<double>& fvnsim10_vector,
    const std::vector<double>& fstdnsim_vector,
    const std::vector<double>& fvdegenergy_vector) const {
  // The prediction uses the fit of three parameters for the function
  // ExponentialFromFit given an NSIM value fit over the TCD-VOIP Dataset.
  // See scripts/fit_nsim_to_mos_poly.py for recalculation.
  // The new coefficients yet has an upper mapping of NSIM=1.0->MOS=~4.5,
  // which we optionally scale to be 1.0->5.0 to provide a perfect score.

  constexpr double kFitParameterA = -262.847869;
  constexpr double kFitParameterB = 0.0154302525;
  constexpr double kFitParameterX0 = -361.063949;
  constexpr double kFitScale = 1.245063;
  double nsim_mean =
      std::accumulate(fvnsim_vector.begin(), fvnsim_vector.end(), 0.0) /
      static_cast<double>(fvnsim_vector.size());

  double mos = MiscMath::ExponentialFromFit(nsim_mean, kFitParameterA,
                                            kFitParameterB, kFitParameterX0);

  float scale = scale_to_max_mos_ ? kFitScale : 1.0;

  // Clamp to 1-5 range
  return std::min(std::max(mos * scale, 1.), 5.);
}
}  // namespace Visqol
