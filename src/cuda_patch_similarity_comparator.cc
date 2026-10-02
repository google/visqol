// Copyright 2026 The ViSQOL Authors
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

#include "cuda_patch_similarity_comparator.h"

#include <vector>

#include "status_macros.h"

namespace Visqol {
absl::Status CudaPatchSimilarityComparator::Init() {
  absl::MutexLock lock(&mutex_);
  return nsim_.Init();
}

absl::StatusOr<std::vector<ImagePatch>>
CudaPatchSimilarityComparator::PrepareCandidateSearch(
    const AMatrix<double>& spectrum, size_t patch_cols) const {
  if (spectrum.NumRows() == 0 || spectrum.NumCols() == 0) {
    return absl::InvalidArgumentError("Candidate spectrum must be nonempty.");
  }
  absl::MutexLock lock(&mutex_);
  VISQOL_RETURN_IF_ERROR(nsim_.SetSpectrogram(
      &*spectrum.cbegin(), spectrum.NumRows(), spectrum.NumCols(), patch_cols));
  return std::vector<ImagePatch>{};
}

absl::StatusOr<std::vector<double>>
CudaPatchSimilarityComparator::MeasureCandidateSimilarities(
    const ImagePatch& reference, const std::vector<ImagePatch>&, size_t first,
    size_t count) const {
  if (reference.NumRows() == 0 || reference.NumCols() == 0) {
    return absl::InvalidArgumentError("NSIM patches must be nonempty.");
  }
  absl::MutexLock lock(&mutex_);
  return nsim_.ScoreWindows(&*reference.cbegin(), reference.NumRows(),
                            reference.NumCols(), first, count);
}

}  // namespace Visqol
