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

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "gammatone_spectrogram_builder.h"
#include "gtest/gtest.h"

namespace Visqol {
namespace {
#ifndef VISQOL_TEST_CUDA
TEST(CudaGammatoneTest, CpuBuildReportsUnsupported) {
  CudaGammatone gpu;
  EXPECT_EQ(gpu.Init().code(), absl::StatusCode::kUnimplemented);
  GammatoneSpectrogramBuilder builder(GammatoneFilterBank(32, 50), false);
  EXPECT_EQ(builder.InitCuda().code(), absl::StatusCode::kUnimplemented);
}
#else
// The kernel avoids FMA contraction, but the CPU build may not (e.g. aarch64 or
// -march=native), so compare with a tight relative tolerance.
void ExpectNear(const std::vector<double>& actual,
                const std::vector<double>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-9 * std::abs(expected[i]) + 1e-15)
        << "element " << i;
  }
}

TEST(CudaGammatoneTest, MatchesCpuFramesBandsAndWindowBoundaries) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> random(-0.5, 0.5);
  for (size_t rate : {16000, 44100, 48000}) {
    const bool speech = rate == 16000;
    GammatoneFilterBank bank(speech ? 21 : 32, 50);
    GammatoneSpectrogramBuilder cpu(bank, speech), gpu(bank, speech);
    ASSERT_TRUE(gpu.InitCuda().ok());
    for (double overlap : {0.25, 0.5}) {
      AnalysisWindow window(rate, overlap);
      const size_t hop = window.size * overlap;
      for (size_t extra : {size_t{1}, hop - 1, hop, rate}) {
        AudioSignal signal{AMatrix<double>(window.size + extra, 1), rate};
        for (auto& value : signal.data_matrix) value = random(rng);
        // Include discontinuities and impulses at overlap boundaries.
        signal.data_matrix(0) = 1.0;
        signal.data_matrix(hop) = -1.0;
        auto expected = cpu.Build(signal, window);
        auto actual = gpu.Build(signal, window);
        ASSERT_TRUE(expected.ok()) << expected.status();
        ASSERT_TRUE(actual.ok()) << actual.status();
        EXPECT_EQ(actual->GetCenterFreqBands(), expected->GetCenterFreqBands());
        ASSERT_EQ(actual->Data().NumRows(), expected->Data().NumRows());
        ASSERT_EQ(actual->Data().NumCols(), expected->Data().NumCols());
        SCOPED_TRACE(testing::Message() << "rate " << rate << " overlap "
                                        << overlap << " extra " << extra);
        ExpectNear(actual->Data().ToVector(), expected->Data().ToVector());
      }
    }
  }
}

TEST(CudaGammatoneTest, SilenceAndConstantSignalsResetConditions) {
  GammatoneFilterBank bank(21, 50);
  GammatoneSpectrogramBuilder cpu(bank, true), gpu(bank, true);
  ASSERT_TRUE(gpu.InitCuda().ok());
  AnalysisWindow window(16000, 0.25);
  // Reuse the workspace at different sizes and levels, then return to silence.
  for (double level : {0.0, 0.1, -0.5, 0.0}) {
    for (size_t samples : {size_t{16000}, size_t{1601}, size_t{32000}}) {
      AudioSignal signal{AMatrix<double>::Filled(samples, 1, level), 16000};
      auto expected = cpu.Build(signal, window);
      auto actual = gpu.Build(signal, window);
      ASSERT_TRUE(expected.ok());
      ASSERT_TRUE(actual.ok()) << actual.status();
      ExpectNear(actual->Data().ToVector(), expected->Data().ToVector());
    }
  }
}

TEST(CudaGammatoneTest, BatchMatchesIndependentBuildsAndReusesWorkspace) {
  for (size_t rate : {16000, 44100, 48000}) {
    const bool speech = rate == 16000;
    GammatoneSpectrogramBuilder cpu(GammatoneFilterBank(speech ? 21 : 32, 50),
                                    speech);
    GammatoneSpectrogramBuilder gpu(GammatoneFilterBank(speech ? 21 : 32, 50),
                                    speech);
    ASSERT_TRUE(gpu.InitCuda().ok());
    AnalysisWindow window(rate, 0.25);
    std::vector<AudioSignal> signals;
    for (size_t samples : {window.size + 1, rate / 2, rate}) {
      AudioSignal signal{AMatrix<double>(samples, 1), rate};
      for (size_t i = 0; i < samples; ++i)
        signal.data_matrix(i) = 0.3 * sin(i * 0.17);
      signals.push_back(std::move(signal));
    }
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto batch = gpu.BuildBatch(signals, window);
      ASSERT_TRUE(batch.ok()) << batch.status();
      ASSERT_EQ(batch->size(), signals.size());
      for (size_t i = 0; i < signals.size(); ++i) {
        auto expected = cpu.Build(signals[i], window);
        ASSERT_TRUE(expected.ok());
        ExpectNear((*batch)[i].Data().ToVector(), expected->Data().ToVector());
        EXPECT_EQ((*batch)[i].GetCenterFreqBands(),
                  expected->GetCenterFreqBands());
      }
      signals.pop_back();
    }
    EXPECT_TRUE(gpu.BuildBatch({}, window)->empty());
    signals.front().data_matrix = AMatrix<double>(window.size, 1);
    EXPECT_FALSE(gpu.BuildBatch(signals, window).ok());
  }
}

TEST(CudaGammatoneTest, ValidatesDimensionsAndInitialization) {
  CudaGammatone gpu;
  double data = 0;
  EXPECT_EQ(gpu.Build(&data, 3, &data, 1, &data, 2, 1, &data).code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(gpu.Init().ok());
  EXPECT_EQ(gpu.Build(nullptr, 3, &data, 1, &data, 2, 1, &data).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gpu.Build(&data, 3, &data, 0, &data, 2, 1, &data).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gpu.Build(&data, 3, &data, 1, &data, 2, 0, &data).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gpu.Build(&data, 2, &data, 1, &data, 2, 1, &data).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gpu.Build(&data, 3, &data, std::numeric_limits<size_t>::max(),
                      &data, 2, 1, &data)
                .code(),
            absl::StatusCode::kInvalidArgument);
  GammatoneSpectrogramBuilder builder(GammatoneFilterBank(21, 50), true);
  ASSERT_TRUE(builder.InitCuda().ok());
  AnalysisWindow window(16000, 0.25);
  AudioSignal too_short{AMatrix<double>::Filled(window.size, 1, 0.0), 16000};
  EXPECT_EQ(builder.Build(too_short, window).status().code(),
            absl::StatusCode::kInvalidArgument);
}
#endif
}  // namespace
}  // namespace Visqol
