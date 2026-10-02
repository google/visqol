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

#include "cuda_nsim.h"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <memory>

#include "absl/strings/str_cat.h"
#include "status_macros.h"

namespace Visqol {
namespace {
constexpr int kThreads = 128;
constexpr size_t kMaxPatchElements = 2048;

absl::Status CudaStatus(cudaError_t error) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat("CUDA NSIM: ", cudaGetErrorString(error)));
}

__global__ void ScoreCandidates(const double* reference,
                                const double* candidates, int rows, int cols,
                                double* scores, size_t available_cols) {
  const int elements = rows * cols;
  const double* degraded = candidates + static_cast<size_t>(blockIdx.x) * rows;
  extern __shared__ double shared[];
  double* band_means = shared + elements;
  const double weights[9] = {
      0.0113033910173052, 0.0838251475442633, 0.0113033910173052,
      0.0838251475442633, 0.619485845753726,  0.0838251475442633,
      0.0113033910173052, 0.0838251475442633, 0.0113033910173052};
  const double c1 = 0.01 * 0.01;
  const double c3 = 0.03 * 0.03 / 2.0;
  for (int index = threadIdx.x; index < elements; index += blockDim.x) {
    const int row = index % rows;
    const int col = index / rows;
    double mu_r = 0.0, mu_d = 0.0;
    double r_sq = 0.0, d_sq = 0.0, r_d = 0.0;
    int weight = 8;
    // Replicate padding and column-major convolution order from Convolution2D.
    for (int dc = -1; dc <= 1; ++dc) {
      for (int dr = -1; dr <= 1; ++dr) {
        const int r = min(max(row + dr, 0), rows - 1);
        const int c = min(max(col + dc, 0), cols - 1);
        const double x = reference[c * rows + r];
        const double y = static_cast<size_t>(blockIdx.x) + c < available_cols
                             ? degraded[c * rows + r]
                             : 0.0;
        const double w = weights[weight--];
        // Explicit rounding avoids CUDA's automatic multiply-add contraction.
        mu_r = __dadd_rn(mu_r, __dmul_rn(x, w));
        mu_d = __dadd_rn(mu_d, __dmul_rn(y, w));
        r_sq = __dadd_rn(r_sq, __dmul_rn(__dmul_rn(x, x), w));
        d_sq = __dadd_rn(d_sq, __dmul_rn(__dmul_rn(y, y), w));
        r_d = __dadd_rn(r_d, __dmul_rn(__dmul_rn(x, y), w));
      }
    }
    const double mu_r_sq = __dmul_rn(mu_r, mu_r);
    const double mu_d_sq = __dmul_rn(mu_d, mu_d);
    const double mu_rd = __dmul_rn(mu_r, mu_d);
    const double variance_product = __dmul_rn(r_sq - mu_r_sq, d_sq - mu_d_sq);
    const double structure_denom =
        variance_product < 0.0 ? c3 : sqrt(variance_product) + c3;
    const double intensity =
        __dadd_rn(__dmul_rn(2.0, mu_rd), c1) / (mu_r_sq + mu_d_sq + c1);
    shared[index] = intensity * ((r_d - mu_rd + c3) / structure_denom);
  }
  __syncthreads();
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    double sum = 0.0;
    for (int col = 0; col < cols; ++col) sum += shared[col * rows + row];
    band_means[row] = sum / cols;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double sum = 0.0;
    for (int row = 0; row < rows; ++row) sum += band_means[row];
    scores[blockIdx.x] = sum / rows;
  }
}
}  // namespace

struct CudaNsim::Workspace {
  int device = -1;
  cudaStream_t stream = nullptr;
  double* reference = nullptr;
  double* scores = nullptr;
  double* spectrum = nullptr;
  size_t spectrum_capacity = 0;
  size_t spectrum_rows = 0, spectrum_frames = 0, patch_cols = 0;
  size_t reference_capacity = 0;
  size_t scores_capacity = 0;

  ~Workspace() {
    if (device < 0) return;
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (stream != nullptr) cudaStreamSynchronize(stream);
    cudaFree(reference);
    cudaFree(scores);
    cudaFree(spectrum);
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (previous >= 0) cudaSetDevice(previous);
  }

  static absl::Status Reserve(double** buffer, size_t* capacity, size_t size) {
    if (size <= *capacity) return absl::OkStatus();
    double* replacement = nullptr;
    VISQOL_RETURN_IF_ERROR(
        CudaStatus(cudaMalloc(&replacement, size * sizeof(double))));
    const auto status = CudaStatus(cudaFree(*buffer));
    if (!status.ok()) {
      cudaFree(replacement);
      return status;
    }
    *buffer = replacement;
    *capacity = size;
    return absl::OkStatus();
  }
};

CudaNsim::CudaNsim() = default;
CudaNsim::~CudaNsim() = default;

absl::Status CudaNsim::Init() {
  auto workspace = std::make_unique<Workspace>();
  VISQOL_RETURN_IF_ERROR(CudaStatus(cudaGetDevice(&workspace->device)));
  VISQOL_RETURN_IF_ERROR(CudaStatus(
      cudaStreamCreateWithFlags(&workspace->stream, cudaStreamNonBlocking)));
  workspace_ = std::move(workspace);
  return absl::OkStatus();
}

absl::Status CudaNsim::SetSpectrogram(const double* spectrum, size_t rows,
                                      size_t frames, size_t patch_cols) {
  if (!workspace_)
    return absl::FailedPreconditionError("CUDA NSIM is not initialized.");
  auto& w = *workspace_;
  w.spectrum_frames = 0;
  if (!spectrum || rows == 0 || frames == 0 || patch_cols == 0 ||
      rows > kMaxPatchElements / patch_cols ||
      frames > std::numeric_limits<size_t>::max() / sizeof(double) / rows) {
    return absl::InvalidArgumentError("Invalid CUDA NSIM spectrum dimensions.");
  }
  int device;
  VISQOL_RETURN_IF_ERROR(CudaStatus(cudaGetDevice(&device)));
  if (device != w.device)
    return absl::FailedPreconditionError(
        "CUDA device changed after initialization.");
  VISQOL_RETURN_IF_ERROR(
      Workspace::Reserve(&w.spectrum, &w.spectrum_capacity, rows * frames));
  // Pageable cudaMemcpy may return before device DMA finishes. Use the scoring
  // stream and drain it before releasing the borrowed host buffer, even on error.
  const auto upload = CudaStatus(cudaMemcpyAsync(
      w.spectrum, spectrum, rows * frames * sizeof(double),
      cudaMemcpyHostToDevice, w.stream));
  const auto completed = CudaStatus(cudaStreamSynchronize(w.stream));
  VISQOL_RETURN_IF_ERROR(upload);
  VISQOL_RETURN_IF_ERROR(completed);
  w.spectrum_rows = rows;
  w.spectrum_frames = frames;
  w.patch_cols = patch_cols;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<double>> CudaNsim::ScoreWindows(
    const double* reference, size_t rows, size_t cols, size_t first,
    size_t count) {
  if (!workspace_ || workspace_->spectrum_frames == 0) {
    return absl::FailedPreconditionError("CUDA NSIM spectrum is not prepared.");
  }
  auto& w = *workspace_;
  if (rows != w.spectrum_rows || cols != w.patch_cols ||
      first > w.spectrum_frames || count > w.spectrum_frames - first ||
      count > std::numeric_limits<unsigned int>::max() || !reference) {
    return absl::InvalidArgumentError("Invalid CUDA NSIM window range.");
  }
  if (count == 0) return std::vector<double>{};
  int device;
  VISQOL_RETURN_IF_ERROR(CudaStatus(cudaGetDevice(&device)));
  if (device != w.device)
    return absl::FailedPreconditionError(
        "CUDA device changed after initialization.");
  const size_t elements = rows * cols;
  VISQOL_RETURN_IF_ERROR(
      Workspace::Reserve(&w.reference, &w.reference_capacity, elements));
  VISQOL_RETURN_IF_ERROR(
      Workspace::Reserve(&w.scores, &w.scores_capacity, count));
  std::vector<double> scores(count);
  struct Drain {
    cudaStream_t stream;
    ~Drain() {
      if (stream) cudaStreamSynchronize(stream);
    }
  } drain{w.stream};
  VISQOL_RETURN_IF_ERROR(CudaStatus(
      cudaMemcpyAsync(w.reference, reference, elements * sizeof(double),
                      cudaMemcpyHostToDevice, w.stream)));
  ScoreCandidates<<<count, kThreads, (elements + rows) * sizeof(double),
                    w.stream>>>(w.reference, w.spectrum + first * rows, rows,
                                cols, w.scores, w.spectrum_frames - first);
  VISQOL_RETURN_IF_ERROR(CudaStatus(cudaGetLastError()));
  VISQOL_RETURN_IF_ERROR(CudaStatus(
      cudaMemcpyAsync(scores.data(), w.scores, count * sizeof(double),
                      cudaMemcpyDeviceToHost, w.stream)));
  const auto status = CudaStatus(cudaStreamSynchronize(w.stream));
  if (!status.ok()) return status;
  drain.stream = nullptr;
  return scores;
}
}  // namespace Visqol
