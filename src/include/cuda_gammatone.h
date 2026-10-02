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

#ifndef VISQOL_INCLUDE_CUDA_GAMMATONE_H_
#define VISQOL_INCLUDE_CUDA_GAMMATONE_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/status.h"

namespace Visqol {
// Reusable workspace for independent window/band IIR recurrences. Calls must
// be serialized and made with the initializing device current.
class CudaGammatone {
 public:
  CudaGammatone();
  ~CudaGammatone();
  CudaGammatone(const CudaGammatone&) = delete;
  CudaGammatone& operator=(const CudaGammatone&) = delete;

  absl::Status Init();
  // Coefficients (bands x 10) and output (bands x frames) are column-major.
  // The caller supplies Hann weights computed by AnalysisWindow on the CPU.
  absl::Status Build(const double* signal, size_t samples,
                     const double* coefficients, size_t bands,
                     const double* hann, size_t window, size_t hop,
                     double* output);

  // Frame starts index independent windows in one concatenated signal buffer.
  absl::Status BuildBatch(const double* signal, size_t samples,
                          const double* coefficients, size_t bands,
                          const double* hann, size_t window,
                          const std::vector<size_t>& frame_starts,
                          double* output);

 private:
  absl::Status BuildImpl(const double* signal, size_t samples,
                         const double* coefficients, size_t bands,
                         const double* hann, size_t window, size_t hop,
                         const size_t* frame_starts, size_t frames,
                         double* output);
  struct Workspace;
  std::unique_ptr<Workspace> workspace_;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_CUDA_GAMMATONE_H_
