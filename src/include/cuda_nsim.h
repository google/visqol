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

#ifndef VISQOL_INCLUDE_CUDA_NSIM_H_
#define VISQOL_INCLUDE_CUDA_NSIM_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace Visqol {
// Reusable CUDA workspace. Inputs are column-major, matching AMatrix.
// Calls must be serialized and made with the initializing device current.
class CudaNsim {
 public:
  CudaNsim();
  ~CudaNsim();
  CudaNsim(const CudaNsim&) = delete;
  CudaNsim& operator=(const CudaNsim&) = delete;

  absl::Status Init();
  absl::Status SetSpectrogram(const double* spectrum, size_t rows,
                              size_t frames, size_t patch_cols);
  absl::StatusOr<std::vector<double>> ScoreWindows(const double* reference,
                                                   size_t rows, size_t cols,
                                                   size_t first, size_t count);

 private:
  struct Workspace;
  std::unique_ptr<Workspace> workspace_;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_CUDA_NSIM_H_
