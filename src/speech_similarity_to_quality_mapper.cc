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

#include "google/protobuf/stubs/status.h"

#include "amatrix.h"

namespace Visqol {
SpeechSimilarityToQualityMapper::SpeechSimilarityToQualityMapper(
    bool scale_to_max_mos) : scale_to_max_mos_{scale_to_max_mos} {}


google::protobuf::util::Status SpeechSimilarityToQualityMapper::Init() {
  return google::protobuf::util::Status();
}

double SpeechSimilarityToQualityMapper::PredictQuality(
    const std::vector<double> &similarity_vector) const {
  // Third order polynomial fitting coefficients on the mean NSIM value over
  // the TCD-VOIP Dataset.
  // See scripts/fit_nsim_to_mos_poly.py for recalculation.
  // The new coefficients yet has an upper mapping of NSIM=1.0->MOS=~4.5,
  // which we optionally scale to be 1.0->5.0 to provide a perfect score.
  constexpr std::array<double, 4> kNsimToMosCoefs = {{41.644078,
                                                      -77.034652,
                                                      47.334363,
                                                      -7.487007}};

  auto nsim_mean = std::accumulate(similarity_vector.begin(),
                                    similarity_vector.end(),
                                    0.0) / static_cast<double>(
                                      similarity_vector.size());

  auto mos = 0.0;
  for (size_t i = 0; i < kNsimToMosCoefs.size(); i++) {
    mos += kNsimToMosCoefs[i] * pow(nsim_mean,
        kNsimToMosCoefs.size() - (i + 1));
  }

  float scale = scale_to_max_mos_ ? 1.121886 : 1.0;

  // Clamp to 1-5 range
  return std::min(std::max(mos * scale, 1.), 5.);
}
}  // namespace Visqol
