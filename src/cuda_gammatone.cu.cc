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

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <memory>

#include "absl/strings/str_cat.h"
#include "status_macros.h"

namespace Visqol {
namespace {
constexpr int kThreads = 128;

absl::Status CudaStatus(cudaError_t error) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat("CUDA Gammatone: ", cudaGetErrorString(error)));
}

__device__ double Filter(double x, double a0, double a1, double a2, double b1,
                         double b2, double* z0, double* z1) {
  // Match SignalFilter::Filter, including separate multiply/add rounding.
  const double y = __dadd_rn(__dmul_rn(a0, x), *z0);
  *z0 = __dsub_rn(__dadd_rn(__dmul_rn(a1, x), *z1), __dmul_rn(b1, y));
  *z1 = __dsub_rn(__dadd_rn(__dmul_rn(a2, x), 0.0), __dmul_rn(b2, y));
  return y;
}

__global__ void BuildSpectrogram(const double* signal,
                                 const double* coefficients, const double* hann,
                                 size_t bands, size_t frames, size_t window,
                                 size_t hop, const size_t* frame_starts,
                                 double* output) {
  for (size_t index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < bands * frames;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t band = index % bands;
    const size_t start =
        frame_starts ? frame_starts[index / bands] : (index / bands) * hop;
    double c[10];
    for (int i = 0; i < 10; ++i) c[i] = coefficients[i * bands + band];
    const double a0 = __ddiv_rn(c[0], c[9]);
    const double a1 = __ddiv_rn(c[1], c[9]);
    const double a2 = __ddiv_rn(c[5], c[9]);
    // Every analysis frame starts with zero filter conditions, as on the CPU.
    double z0[4] = {}, z1[4] = {};
    double sum = 0.0;
    for (size_t sample = 0; sample < window; ++sample) {
      double x = __dmul_rn(signal[start + sample], hann[sample]);
      x = Filter(x, a0, a1, a2, c[7], c[8], &z0[0], &z1[0]);
      x = Filter(x, c[0], c[2], c[5], c[7], c[8], &z0[1], &z1[1]);
      x = Filter(x, c[0], c[3], c[5], c[7], c[8], &z0[2], &z1[2]);
      x = Filter(x, c[0], c[4], c[5], c[7], c[8], &z0[3], &z1[3]);
      // Preserve the CPU row-mean reduction order; no parallel sample
      // reduction.
      sum = __dadd_rn(sum, __dmul_rn(x, x));
    }
    output[index] = sqrt(__ddiv_rn(sum, static_cast<double>(window)));
  }
}
}  // namespace

struct CudaGammatone::Workspace {
  int device = -1;
  cudaStream_t stream = nullptr;
  double* signal = nullptr;
  double* coefficients = nullptr;
  double* hann = nullptr;
  double* output = nullptr;
  size_t* frame_starts = nullptr;
  size_t frame_capacity = 0;
  size_t signal_capacity = 0;
  size_t coefficient_capacity = 0;
  size_t hann_capacity = 0;
  size_t output_capacity = 0;

  ~Workspace() {
    if (device < 0) return;
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (stream != nullptr) cudaStreamSynchronize(stream);
    cudaFree(signal);
    cudaFree(coefficients);
    cudaFree(hann);
    cudaFree(output);
    cudaFree(frame_starts);
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (previous >= 0) cudaSetDevice(previous);
  }

  template <typename T>
  static absl::Status Reserve(T** buffer, size_t* capacity, size_t size) {
    if (size <= *capacity) return absl::OkStatus();
    T* replacement = nullptr;
    VISQOL_RETURN_IF_ERROR(
        CudaStatus(cudaMalloc(&replacement, size * sizeof(T))));
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

CudaGammatone::CudaGammatone() = default;
CudaGammatone::~CudaGammatone() = default;

absl::Status CudaGammatone::Init() {
  auto workspace = std::make_unique<Workspace>();
  VISQOL_RETURN_IF_ERROR(CudaStatus(cudaGetDevice(&workspace->device)));
  VISQOL_RETURN_IF_ERROR(CudaStatus(
      cudaStreamCreateWithFlags(&workspace->stream, cudaStreamNonBlocking)));
  workspace_ = std::move(workspace);
  return absl::OkStatus();
}

absl::Status CudaGammatone::Build(const double* signal, size_t samples,
                                  const double* coefficients, size_t bands,
                                  const double* hann, size_t window, size_t hop,
                                  double* output) {
  return BuildImpl(signal, samples, coefficients, bands, hann, window, hop,
                   nullptr, 0, output);
}

absl::Status CudaGammatone::BuildBatch(const double* signal, size_t samples,
                                       const double* coefficients, size_t bands,
                                       const double* hann, size_t window,
                                       const std::vector<size_t>& frame_starts,
                                       double* output) {
  if (frame_starts.empty())
    return absl::InvalidArgumentError("Empty CUDA frame batch.");
  for (size_t start : frame_starts) {
    if (start > samples || window > samples - start) {
      return absl::InvalidArgumentError(
          "CUDA frame extends beyond the signal.");
    }
  }
  return BuildImpl(signal, samples, coefficients, bands, hann, window, 1,
                   frame_starts.data(), frame_starts.size(), output);
}

absl::Status CudaGammatone::BuildImpl(const double* signal, size_t samples,
                                      const double* coefficients, size_t bands,
                                      const double* hann, size_t window,
                                      size_t hop, const size_t* frame_starts,
                                      size_t frames, double* output) {
  if (!workspace_) {
    return absl::FailedPreconditionError("CUDA Gammatone is not initialized.");
  }
  constexpr size_t kMaxElements =
      std::numeric_limits<size_t>::max() / sizeof(double);
  if (signal == nullptr || coefficients == nullptr || hann == nullptr ||
      output == nullptr || bands == 0 || window < 2 || samples <= window ||
      samples > kMaxElements || hop == 0 || bands > kMaxElements / 10) {
    return absl::InvalidArgumentError("Invalid CUDA Gammatone inputs.");
  }
  if (!frame_starts) frames = 1 + (samples - window) / hop;
  if (frames > kMaxElements / bands) {
    return absl::InvalidArgumentError("CUDA Gammatone output is too large.");
  }
  int device;
  VISQOL_RETURN_IF_ERROR(CudaStatus(cudaGetDevice(&device)));
  if (device != workspace_->device) {
    return absl::FailedPreconditionError(
        "CUDA device changed after initialization.");
  }
  auto& w = *workspace_;
  VISQOL_RETURN_IF_ERROR(
      Workspace::Reserve(&w.signal, &w.signal_capacity, samples));
  VISQOL_RETURN_IF_ERROR(
      Workspace::Reserve(&w.coefficients, &w.coefficient_capacity, bands * 10));
  VISQOL_RETURN_IF_ERROR(Workspace::Reserve(&w.hann, &w.hann_capacity, window));
  VISQOL_RETURN_IF_ERROR(
      Workspace::Reserve(&w.output, &w.output_capacity, bands * frames));
  if (frame_starts) {
    VISQOL_RETURN_IF_ERROR(
        Workspace::Reserve(&w.frame_starts, &w.frame_capacity, frames));
  }
  // Drain borrowed host buffers on both success and any asynchronous failure.
  struct SynchronizeOnExit {
    cudaStream_t stream;
    ~SynchronizeOnExit() {
      if (stream != nullptr) cudaStreamSynchronize(stream);
    }
  } synchronize_on_exit{w.stream};
  VISQOL_RETURN_IF_ERROR(
      CudaStatus(cudaMemcpyAsync(w.signal, signal, samples * sizeof(double),
                                 cudaMemcpyHostToDevice, w.stream)));
  VISQOL_RETURN_IF_ERROR(CudaStatus(
      cudaMemcpyAsync(w.coefficients, coefficients, bands * 10 * sizeof(double),
                      cudaMemcpyHostToDevice, w.stream)));
  VISQOL_RETURN_IF_ERROR(
      CudaStatus(cudaMemcpyAsync(w.hann, hann, window * sizeof(double),
                                 cudaMemcpyHostToDevice, w.stream)));
  if (frame_starts) {
    VISQOL_RETURN_IF_ERROR(CudaStatus(
        cudaMemcpyAsync(w.frame_starts, frame_starts, frames * sizeof(size_t),
                        cudaMemcpyHostToDevice, w.stream)));
  }
  const unsigned int blocks = static_cast<unsigned int>(
      std::min<size_t>(65535, (bands * frames + kThreads - 1) / kThreads));
  BuildSpectrogram<<<blocks, kThreads, 0, w.stream>>>(
      w.signal, w.coefficients, w.hann, bands, frames, window, hop,
      frame_starts ? w.frame_starts : nullptr, w.output);
  VISQOL_RETURN_IF_ERROR(CudaStatus(cudaGetLastError()));
  VISQOL_RETURN_IF_ERROR(CudaStatus(
      cudaMemcpyAsync(output, w.output, bands * frames * sizeof(double),
                      cudaMemcpyDeviceToHost, w.stream)));
  const auto status = CudaStatus(cudaStreamSynchronize(w.stream));
  if (status.ok()) synchronize_on_exit.stream = nullptr;
  return status;
}
}  // namespace Visqol
