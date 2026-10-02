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

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "comparison_patches_selector.h"
#include "gtest/gtest.h"
#include "misc_audio.h"
#include "visqol_api.h"

namespace Visqol {
namespace {
ImagePatch RandomPatch(size_t rows, size_t cols, std::mt19937* rng) {
  ImagePatch patch(rows, cols);
  std::uniform_real_distribution<double> values(-80.0, 20.0);
  for (auto& value : patch) value = values(*rng);
  return patch;
}

TEST(CandidateSimilaritiesTest, DefaultMatchesIndividualScores) {
  NeurogramSimiliarityIndexMeasure cpu;
  std::mt19937 rng(42);
  auto reference = RandomPatch(32, 30, &rng);
  std::vector<ImagePatch> candidates;
  for (int i = 0; i < 8; ++i) candidates.push_back(RandomPatch(32, 30, &rng));
  auto scores = cpu.MeasureCandidateSimilarities(reference, candidates, 2, 5);
  ASSERT_TRUE(scores.ok()) << scores.status();
  ASSERT_EQ(scores->size(), 5);
  for (size_t i = 0; i < scores->size(); ++i) {
    EXPECT_DOUBLE_EQ(
        (*scores)[i],
        cpu.MeasurePatchSimilarity(reference, candidates[i + 2]).similarity);
  }
  EXPECT_FALSE(
      cpu.MeasureCandidateSimilarities(reference, candidates, 7, 2).ok());
  EXPECT_FALSE(
      cpu.MeasureCandidateSimilarities(reference, candidates, 9, 0).ok());
  auto empty = cpu.MeasureCandidateSimilarities(reference, candidates, 8, 0);
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
}

TEST(CandidateSimilaritiesTest, SelectorPropagatesScoringFailure) {
  class FailingComparator : public NeurogramSimiliarityIndexMeasure {
   public:
    absl::StatusOr<std::vector<double>> MeasureCandidateSimilarities(
        const ImagePatch&, const std::vector<ImagePatch>&, size_t,
        size_t) const override {
      return absl::InternalError("test scoring failure");
    }
  };
  ComparisonPatchesSelector selector(std::make_unique<FailingComparator>());
  auto patch = ImagePatch::Filled(32, 30, 0.0);
  auto result =
      selector.FindMostOptimalDegPatches({patch}, {0}, patch, 0.02, 1);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
}

#ifndef VISQOL_TEST_CUDA
TEST(CudaCandidateSimilaritiesTest, CpuBuildReportsUnsupported) {
  CudaPatchSimilarityComparator gpu;
  EXPECT_EQ(gpu.Init().code(), absl::StatusCode::kUnimplemented);
}
#else
TEST(CudaCandidateSimilaritiesTest, MatchesCpuAcrossWindowCountsAndDimensions) {
  CudaPatchSimilarityComparator gpu;
  ASSERT_TRUE(gpu.Init().ok());
  NeurogramSimiliarityIndexMeasure cpu;
  std::mt19937 rng(42);
  for (const auto& shape : std::vector<std::pair<size_t, size_t>>{
           {1, 1}, {3, 7}, {21, 20}, {32, 30}}) {
    auto reference = RandomPatch(shape.first, shape.second, &rng);
    auto spectrum = RandomPatch(shape.first, 515, &rng);
    auto candidates = cpu.PrepareCandidateSearch(spectrum, shape.second);
    auto prepared = gpu.PrepareCandidateSearch(spectrum, shape.second);
    ASSERT_TRUE(candidates.ok());
    ASSERT_TRUE(prepared.ok());
    auto actual =
        gpu.MeasureCandidateSimilarities(reference, *prepared, 2, 513);
    auto expected =
        cpu.MeasureCandidateSimilarities(reference, *candidates, 2, 513);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_TRUE(expected.ok());
    ASSERT_EQ(actual->size(), expected->size());
    for (size_t i = 0; i < actual->size(); ++i) {
      EXPECT_NEAR((*actual)[i], (*expected)[i], 1e-10) << "candidate " << i;
    }
  }
}

TEST(CudaCandidateSimilaritiesTest, SilentAndNearlyConstantPatches) {
  CudaPatchSimilarityComparator gpu;
  ASSERT_TRUE(gpu.Init().ok());
  NeurogramSimiliarityIndexMeasure cpu;
  for (double level : {0.0, -80.0, 0.1, 100.0}) {
    auto reference = ImagePatch::Filled(32, 30, level);
    auto spectrum = ImagePatch::Filled(32, 31, level);
    spectrum(0, 1) += 1e-8;
    auto candidates = cpu.PrepareCandidateSearch(spectrum, 30);
    auto prepared = gpu.PrepareCandidateSearch(spectrum, 30);
    ASSERT_TRUE(candidates.ok());
    ASSERT_TRUE(prepared.ok());
    auto actual = gpu.MeasureCandidateSimilarities(reference, *prepared, 0, 2);
    auto expected =
        cpu.MeasureCandidateSimilarities(reference, *candidates, 0, 2);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_TRUE(expected.ok());
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_NEAR((*actual)[i], (*expected)[i], 1e-10);
    }
  }
}

TEST(CudaCandidateSimilaritiesTest, ValidatesInputAndInitialization) {
  CudaPatchSimilarityComparator gpu;
  auto reference = ImagePatch::Filled(32, 30, 0.0);
  EXPECT_EQ(
      gpu.MeasureCandidateSimilarities(reference, {}, 0, 1).status().code(),
      absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(gpu.Init().ok());
  EXPECT_FALSE(gpu.PrepareCandidateSearch(ImagePatch(0, 0), 30).ok());
  EXPECT_FALSE(gpu.PrepareCandidateSearch(reference, 0).ok());
  auto candidates = gpu.PrepareCandidateSearch(reference, 30);
  ASSERT_TRUE(candidates.ok());
  EXPECT_FALSE(
      gpu.MeasureCandidateSimilarities(reference, *candidates, 30, 1).ok());
  EXPECT_FALSE(gpu.MeasureCandidateSimilarities(ImagePatch::Filled(21, 20, 0.0),
                                                *candidates, 0, 1)
                   .ok());
  auto empty = gpu.MeasureCandidateSimilarities(reference, *candidates, 30, 0);
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
}

TEST(CudaCandidateSimilaritiesTest, PreparedSpectrumMatchesPackedCandidates) {
  CudaPatchSimilarityComparator gpu;
  ASSERT_TRUE(gpu.Init().ok());
  NeurogramSimiliarityIndexMeasure cpu;
  std::mt19937 rng(83);
  auto reference = RandomPatch(32, 30, &rng);
  EXPECT_FALSE(gpu.MeasureCandidateSimilarities(reference, {}, 0, 1).ok());
  for (size_t frames : {size_t{7}, size_t{90}, size_t{600}, size_t{17}}) {
    auto spectrum = RandomPatch(32, frames, &rng);
    auto prepared = gpu.PrepareCandidateSearch(spectrum, 30);
    auto expected_candidates = cpu.PrepareCandidateSearch(spectrum, 30);
    ASSERT_TRUE(prepared.ok());
    ASSERT_TRUE(expected_candidates.ok());
    EXPECT_TRUE(prepared->empty());
    for (size_t first : {size_t{0}, frames - 1}) {
      const size_t count = frames - first;
      auto expected = cpu.MeasureCandidateSimilarities(
          reference, *expected_candidates, first, count);
      auto actual =
          gpu.MeasureCandidateSimilarities(reference, *prepared, first, count);
      ASSERT_TRUE(actual.ok()) << actual.status();
      ASSERT_TRUE(expected.ok());
      ASSERT_EQ(actual->size(), expected->size());
      for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR((*actual)[i], (*expected)[i], 1e-10) << "candidate " << i;
      }
    }
    EXPECT_FALSE(
        gpu.MeasureCandidateSimilarities(reference, {}, frames, 1).ok());
    EXPECT_TRUE(
        gpu.MeasureCandidateSimilarities(reference, {}, frames, 0)->empty());
  }
}

TEST(CudaCandidateSimilaritiesTest, RepeatedUploadsAreReadyForImmediateScoring) {
  CudaPatchSimilarityComparator gpu;
  ASSERT_TRUE(gpu.Init().ok());
  NeurogramSimiliarityIndexMeasure cpu;
  std::mt19937 rng(103);
  const auto reference = RandomPatch(32, 30, &rng);
  // Reuse allocations so cudaMalloc cannot mask missing stream ordering.
  // Score the end of a large pageable upload and release its host storage first.
  constexpr size_t kFrames = 65536;
  for (int repeat = 0; repeat < 8; ++repeat) {
    double expected;
    {
      auto spectrum = ImagePatch::Filled(32, kFrames, repeat * 0.1);
      const auto tail = RandomPatch(32, 30, &rng);
      std::copy(tail.cbegin(), tail.cend(),
                spectrum.begin() + (kFrames - 30) * 32);
      expected = cpu.MeasurePatchSimilarity(reference, tail).similarity;
      const auto prepared = gpu.PrepareCandidateSearch(spectrum, 30);
      ASSERT_TRUE(prepared.ok()) << prepared.status();
    }
    const auto actual =
        gpu.MeasureCandidateSimilarities(reference, {}, kFrames - 30, 1);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_EQ(actual->size(), 1);
    EXPECT_NEAR(actual->front(), expected, 1e-10);
  }
}

TEST(CudaCandidateSimilaritiesTest, SelectorMatchesCpuAlignmentAndStatistics) {
  std::mt19937 rng(17);
  auto spectrogram = RandomPatch(32, 100, &rng);
  std::vector<size_t> indices{0, 30, 60};
  std::vector<ImagePatch> reference;
  for (size_t index : indices) {
    reference.push_back(spectrogram.GetSpan(0, 31, index, index + 29));
  }
  ComparisonPatchesSelector cpu(
      std::make_unique<NeurogramSimiliarityIndexMeasure>());
  auto comparator = std::make_unique<CudaPatchSimilarityComparator>();
  ASSERT_TRUE(comparator->Init().ok());
  ComparisonPatchesSelector gpu(std::move(comparator));
  auto expected =
      cpu.FindMostOptimalDegPatches(reference, indices, spectrogram, 0.02, 2);
  auto actual =
      gpu.FindMostOptimalDegPatches(reference, indices, spectrogram, 0.02, 2);
  ASSERT_TRUE(actual.ok()) << actual.status();
  ASSERT_TRUE(expected.ok());
  ASSERT_EQ(actual->size(), expected->size());
  for (size_t i = 0; i < actual->size(); ++i) {
    EXPECT_DOUBLE_EQ((*actual)[i].similarity, (*expected)[i].similarity);
    EXPECT_DOUBLE_EQ((*actual)[i].deg_patch_start_time,
                     (*expected)[i].deg_patch_start_time);
    EXPECT_DOUBLE_EQ((*actual)[i].deg_patch_end_time,
                     (*expected)[i].deg_patch_end_time);
    for (size_t row = 0; row < 32; ++row) {
      EXPECT_DOUBLE_EQ((*actual)[i].freq_band_means(row),
                       (*expected)[i].freq_band_means(row));
      EXPECT_DOUBLE_EQ((*actual)[i].freq_band_stddevs(row),
                       (*expected)[i].freq_band_stddevs(row));
      EXPECT_DOUBLE_EQ((*actual)[i].freq_band_deg_energy(row),
                       (*expected)[i].freq_band_deg_energy(row));
    }
  }
}
TEST(CudaCandidateSimilaritiesTest, AudioAndSpeechApiMatchCpuResults) {
  for (bool speech : {false, true}) {
    std::vector<double> ref_data, deg_data;
    if (speech) {
      // Generate samples at 16 kHz; the bundled CA01 WAV fixtures are 48 kHz.
      constexpr size_t kRate = 16000;
      ref_data.resize(2 * kRate);
      deg_data.resize(ref_data.size());
      for (size_t i = 0; i < ref_data.size(); ++i) {
        const double time = static_cast<double>(i) / kRate;
        ref_data[i] = 0.3 * std::sin(2.0 * M_PI * 440.0 * time) +
                      0.1 * std::sin(2.0 * M_PI * 880.0 * time);
        deg_data[i] = ref_data[i] + 0.01 * std::sin(2.0 * M_PI * 3000.0 * time);
      }
    } else {
      auto ref_audio = MiscAudio::LoadAsMono(FilePath(
          "testdata/conformance_testdata_subset/contrabassoon48_stereo.wav"));
      auto deg_audio = MiscAudio::LoadAsMono(
          FilePath("testdata/conformance_testdata_subset/"
                   "contrabassoon48_stereo_24kbps_aac.wav"));
      ref_data = ref_audio.data_matrix.ToVector();
      deg_data = deg_audio.data_matrix.ToVector();
    }
    absl::Span<double> ref(ref_data), deg(deg_data);
    VisqolConfig config;
    config.mutable_audio()->set_sample_rate(speech ? 16000 : 48000);
    config.mutable_options()->set_use_speech_scoring(speech);
    config.mutable_options()->set_use_lattice_model(speech);
    VisqolApi cpu;
    ASSERT_TRUE(cpu.Create(config).ok());
    auto expected = cpu.Measure(ref, deg);
    config.mutable_options()->set_use_cuda(true);
    VisqolApi gpu;
    ASSERT_TRUE(gpu.Create(config).ok());
    auto actual = gpu.Measure(ref, deg);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_TRUE(expected.ok()) << expected.status();
    // Spectrograms come from the GPU here, so allow for FMA contraction in the
    // CPU build. Patch selection and timing should still match exactly.
    EXPECT_NEAR(actual->moslqo(), expected->moslqo(), 1e-6);
    EXPECT_NEAR(actual->vnsim(), expected->vnsim(), 1e-9);
    ASSERT_EQ(actual->patch_sims_size(), expected->patch_sims_size());
    for (int i = 0; i < actual->patch_sims_size(); ++i) {
      const auto& a = actual->patch_sims(i);
      const auto& e = expected->patch_sims(i);
      EXPECT_NEAR(a.similarity(), e.similarity(), 1e-9);
      EXPECT_DOUBLE_EQ(a.ref_patch_start_time(), e.ref_patch_start_time());
      EXPECT_DOUBLE_EQ(a.ref_patch_end_time(), e.ref_patch_end_time());
      EXPECT_DOUBLE_EQ(a.deg_patch_start_time(), e.deg_patch_start_time());
      EXPECT_DOUBLE_EQ(a.deg_patch_end_time(), e.deg_patch_end_time());
      ASSERT_EQ(a.freq_band_means_size(), e.freq_band_means_size());
      for (int band = 0; band < a.freq_band_means_size(); ++band) {
        EXPECT_NEAR(a.freq_band_means(band), e.freq_band_means(band), 1e-9);
      }
    }
  }
}

#endif
}  // namespace
}  // namespace Visqol
