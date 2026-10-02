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

#ifndef VISQOL_INCLUDE_CUDA_PATCH_SIMILARITY_COMPARATOR_H_
#define VISQOL_INCLUDE_CUDA_PATCH_SIMILARITY_COMPARATOR_H_

#include <cstddef>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "cuda_nsim.h"
#include "neurogram_similiarity_index_measure.h"

namespace Visqol {
// Only candidate search scores use CUDA. Inherited single-patch comparisons
// keep the existing CPU statistics for the selected and realigned patches.
// A prepared search must be serialized from preparation through its last score.
class CudaPatchSimilarityComparator : public NeurogramSimiliarityIndexMeasure {
 public:
  absl::Status Init();
  absl::StatusOr<std::vector<ImagePatch>> PrepareCandidateSearch(
      const AMatrix<double>& spectrum, size_t patch_cols) const override;
  absl::StatusOr<std::vector<double>> MeasureCandidateSimilarities(
      const ImagePatch& reference, const std::vector<ImagePatch>& candidates,
      size_t first, size_t count) const override;

 private:
  mutable absl::Mutex mutex_;
  mutable CudaNsim nsim_ ABSL_GUARDED_BY(mutex_);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_CUDA_PATCH_SIMILARITY_COMPARATOR_H_
