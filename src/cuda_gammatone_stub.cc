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

#include "cuda_gammatone.h"

namespace Visqol {
struct CudaGammatone::Workspace {};
CudaGammatone::CudaGammatone() = default;
CudaGammatone::~CudaGammatone() = default;
absl::Status CudaGammatone::Init() {
  return absl::UnimplementedError(
      "CUDA spectrograms require rebuilding ViSQOL with --config=cuda.");
}
absl::Status CudaGammatone::Build(const double*, size_t, const double*, size_t,
                                  const double*, size_t, size_t, double*) {
  return Init();
}
absl::Status CudaGammatone::BuildBatch(const double*, size_t, const double*,
                                       size_t, const double*, size_t,
                                       const std::vector<size_t>&, double*) {
  return Init();
}
}  // namespace Visqol
