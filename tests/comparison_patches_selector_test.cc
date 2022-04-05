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

#include "comparison_patches_selector.h"

#include <cstddef>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "image_patch_creator.h"
#include "neurogram_similiarity_index_measure.h"
#include "patch_similarity_comparator.h"

namespace Visqol {

class ComparisonPatchesSelectorPeer {
 public:
  explicit ComparisonPatchesSelectorPeer(const ComparisonPatchesSelector* cps)
      : cps_(cps) {}

  size_t CalcMaxNumPatches(const std::vector<size_t>& ref_patch_indices,
                           size_t max_slide_offset, size_t num_frames) const {
    return cps_->CalcMaxNumPatches(ref_patch_indices, max_slide_offset,
                                   num_frames);
  }
  static AudioSignal Slice(const AudioSignal& in_signal, double start_time,
                           double end_time) {
    return ComparisonPatchesSelector::Slice(in_signal, start_time, end_time);
  }
  absl::StatusOr<std::vector<PatchSimilarityResult>> FindMostOptimalDegPatches(
      const std::vector<ImagePatch>& ref_patches,
      const std::vector<size_t>& ref_patch_indices,
      const AMatrix<double>& spectrogram_data, const double frame_duration,
      const int search_window) const {
    return cps_->FindMostOptimalDegPatches(ref_patches, ref_patch_indices,
                                           spectrogram_data, frame_duration,
                                           search_window);
  }

 private:
  const ComparisonPatchesSelector* const cps_;
};

namespace {

class ComparisonPatchesSelectorTest : public ::testing::Test {
 protected:
  ComparisonPatchesSelectorTest() {}
};

TEST_F(ComparisonPatchesSelectorTest, EndPatches) {
  ComparisonPatchesSelector selector(nullptr);
  ComparisonPatchesSelectorPeer selectorPeer(&selector);

  std::vector<size_t> patchIndices = {0, 15, 30, 45, 60};
  // slideOffset is the maximum initial patch index for degraded.
  size_t slideOffset = 45;
  size_t acceptedNumPatches =
      selectorPeer.CalcMaxNumPatches(patchIndices, slideOffset, 30);

  EXPECT_EQ(patchIndices.size(), acceptedNumPatches);

  slideOffset = 44;
  acceptedNumPatches =
      selectorPeer.CalcMaxNumPatches(patchIndices, slideOffset, 30);

  EXPECT_EQ(patchIndices.size() - 1, acceptedNumPatches);
}

TEST_F(ComparisonPatchesSelectorTest, Slice) {
  auto silence_matrix = AMatrix<double>::Filled(16000 * 3, 1, 0.0);
  // Add an impulse at 1.0 secs
  std::vector<double> impulse_vec(1, 1.0);

  silence_matrix.SetRow(16000, impulse_vec);
  AudioSignal three_seconds_silence{silence_matrix, 16000};

  AudioSignal sliced_signal =
      ComparisonPatchesSelectorPeer::Slice(three_seconds_silence, 0.5, 2.5);

  EXPECT_EQ(sliced_signal.GetDuration(), 2.0);

  // Check that the impulse moved to .5 secs.
  EXPECT_EQ(sliced_signal.data_matrix(7999, 0), 0.0);
  EXPECT_EQ(sliced_signal.data_matrix(8000, 0), 1.0);
  EXPECT_EQ(sliced_signal.data_matrix(8001, 0), 0.0);
}

TEST_F(ComparisonPatchesSelectorTest, OptimalPatches) {
  ComparisonPatchesSelector selector(
      absl::make_unique<NeurogramSimiliarityIndexMeasure>());
  ComparisonPatchesSelectorPeer selectorPeer(&selector);

  // Defining the reference audio matrix
  auto ref_matrix = AMatrix<double>::Filled(3, 10, 0.0);
  std::vector<double> top_row{1, 1, 1, 2, 2, 2, 2, 2, 3, 3};
  std::vector<double> med_row{0, 1, 0, 2, 1, 1, 2, 3, 1, 2};
  std::vector<double> bot_row{0, 1, 0, 2, 1, 1, 2, 3, 1, 2};
  ref_matrix.SetRow(0, top_row);
  ref_matrix.SetRow(1, med_row);
  ref_matrix.SetRow(2, bot_row);

  // Create reference patches from given patch indices
  int patch_size = 1;
  std::vector<size_t> patch_indices{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto patch_creator(absl::make_unique<ImagePatchCreator>(patch_size));
  std::vector<ImagePatch> ref_patches =
      patch_creator->CreatePatchesFromIndices(ref_matrix, patch_indices);

  // Defining the degraded audio matrix
  auto deg_matrix = AMatrix<double>::Filled(3, 30, 0.0);
  std::vector<double> top_deg_row{0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 2,
                                  3, 2, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> med_deg_row{0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 2, 1,
                                  2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> bot_deg_row{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1,
                                  2, 3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0};
  deg_matrix.SetRow(0, top_deg_row);
  deg_matrix.SetRow(1, med_deg_row);
  deg_matrix.SetRow(2, bot_deg_row);

  // Check if the output of the algorithm is consistent with the logic
  double frame_duration = 1.0;
  const int search_window = 8;
  auto res = selectorPeer.FindMostOptimalDegPatches(
      ref_patches, patch_indices, deg_matrix, frame_duration, search_window);
  ASSERT_TRUE(res.ok());
  auto best_patches = res.value();
  EXPECT_DOUBLE_EQ(best_patches[3].deg_patch_start_time, 0);
  EXPECT_DOUBLE_EQ(best_patches[4].deg_patch_start_time, 7);
  EXPECT_DOUBLE_EQ(best_patches[5].deg_patch_start_time, 8);
}

TEST_F(ComparisonPatchesSelectorTest, OutOfOrderMatches) {
  ComparisonPatchesSelector selector(
      absl::make_unique<NeurogramSimiliarityIndexMeasure>());
  ComparisonPatchesSelectorPeer selectorPeer(&selector);

  // Defining the reference audio matrix
  auto ref_matrix = AMatrix<double>::Filled(3, 4, 0.0);
  std::vector<double> top_row{1, 100, 3, 4};
  std::vector<double> med_row{0, 0, 0, 0};
  std::vector<double> bot_row{1, 100, 3, 4};
  ref_matrix.SetRow(0, top_row);
  ref_matrix.SetRow(1, med_row);
  ref_matrix.SetRow(2, bot_row);

  // Create reference patches from given patch indices
  int patch_size = 1;
  std::vector<size_t> patch_indices{0, 1, 2, 3};
  auto patch_creator(absl::make_unique<ImagePatchCreator>(patch_size));
  std::vector<ImagePatch> ref_patches =
      patch_creator->CreatePatchesFromIndices(ref_matrix, patch_indices);

  // Defining the degraded audio matrix
  auto deg_matrix = AMatrix<double>::Filled(3, 4, 0.0);
  std::vector<double> top_deg_row{100, 1, 3, 4};
  std::vector<double> med_deg_row{0, 0, 0, 0};
  std::vector<double> bot_deg_row{100, 1, 3, 4};
  deg_matrix.SetRow(0, top_deg_row);
  deg_matrix.SetRow(1, med_deg_row);
  deg_matrix.SetRow(2, bot_deg_row);

  // Check if the output of the algorithm is consistent with the logic
  double frame_duration = 1.0;
  const int search_window = 60;
  auto res = selectorPeer.FindMostOptimalDegPatches(
      ref_patches, patch_indices, deg_matrix, frame_duration, search_window);
  ASSERT_TRUE(res.ok());
  auto best_patches = res.value();
  EXPECT_DOUBLE_EQ(best_patches[0].deg_patch_start_time, 1);
  EXPECT_DOUBLE_EQ(best_patches[1].deg_patch_start_time, 0);
  EXPECT_DOUBLE_EQ(best_patches[2].deg_patch_start_time, 2);
  EXPECT_DOUBLE_EQ(best_patches[3].deg_patch_start_time, 3);
}

TEST_F(ComparisonPatchesSelectorTest, DifferentResults) {
  ComparisonPatchesSelector selector(
      absl::make_unique<NeurogramSimiliarityIndexMeasure>());
  ComparisonPatchesSelectorPeer selectorPeer(&selector);

  // Defining the reference audio matrix
  auto ref_matrix = AMatrix<double>::Filled(3, 1, 0.0);
  std::vector<double> top_row{1};
  std::vector<double> med_row{1};
  std::vector<double> bot_row{0};
  ref_matrix.SetRow(0, top_row);
  ref_matrix.SetRow(1, med_row);
  ref_matrix.SetRow(2, bot_row);

  // Create reference patches from given patch indices
  int patch_size = 1;
  std::vector<size_t> patch_indices{0};
  auto patch_creator(absl::make_unique<ImagePatchCreator>(patch_size));
  std::vector<ImagePatch> ref_patches =
      patch_creator->CreatePatchesFromIndices(ref_matrix, patch_indices);

  // Defining the degraded audio matrix
  auto deg_matrix = AMatrix<double>::Filled(3, 17, 0.0);
  std::vector<double> top_deg_row{0, 0, 0, 0, 0, 0, 1, 0, 0,
                                  3, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> med_deg_row{0, 0, 0, 0, 0, 0, 1, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> bot_deg_row{0, 0, 0, 0, 0, 0, 0, 1, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0};
  deg_matrix.SetRow(0, top_deg_row);
  deg_matrix.SetRow(1, med_deg_row);
  deg_matrix.SetRow(2, bot_deg_row);

  // Check if the output of the algorithm is consistent with the logic
  double frame_duration = 1.0;
  const int search_window = 60;
  auto res = selectorPeer.FindMostOptimalDegPatches(
      ref_patches, patch_indices, deg_matrix, frame_duration, search_window);
  ASSERT_TRUE(res.ok());
  auto best_patches = res.value();
  EXPECT_DOUBLE_EQ(best_patches[0].deg_patch_start_time, 6);
}

TEST_F(ComparisonPatchesSelectorTest, BigExample) {
  ComparisonPatchesSelector selector(
      absl::make_unique<NeurogramSimiliarityIndexMeasure>());
  ComparisonPatchesSelectorPeer selectorPeer(&selector);

  // Defining the reference audio matrix
  auto ref_matrix = AMatrix<double>::Filled(3, 31, 0.0);
  std::vector<double> top_row{0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2,
                              0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> med_row{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 2, 3,
                              0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> bot_row{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 2, 3,
                              0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0};
  ref_matrix.SetRow(0, top_row);
  ref_matrix.SetRow(1, med_row);
  ref_matrix.SetRow(2, bot_row);

  // Create reference patches from given patch indices
  int patch_size = 2;
  std::vector<size_t> patch_indices{4, 6, 10, 12, 14, 22};
  auto patch_creator(absl::make_unique<ImagePatchCreator>(patch_size));
  std::vector<ImagePatch> ref_patches =
      patch_creator->CreatePatchesFromIndices(ref_matrix, patch_indices);

  // Defining the degraded audio matrix
  auto deg_matrix = AMatrix<double>::Filled(3, 31, 0.0);
  std::vector<double> top_deg_row{0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0,
                                  0, 0, 2, 1, 2, 3, 2, 0, 0, 0, 0,
                                  3, 3, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> med_deg_row{0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                                  0, 0, 2, 0, 1, 2, 3, 0, 0, 0, 0,
                                  1, 2, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> bot_deg_row{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                  0, 0, 2, 0, 1, 2, 3, 0, 0, 0, 0,
                                  1, 2, 0, 0, 0, 0, 0, 0, 0};
  deg_matrix.SetRow(0, top_deg_row);
  deg_matrix.SetRow(1, med_deg_row);
  deg_matrix.SetRow(2, bot_deg_row);

  // Check if the output of the algorithm is consistent with the logic
  double frame_duration = 1.0;
  const int search_window = 60;
  auto res = selectorPeer.FindMostOptimalDegPatches(
      ref_patches, patch_indices, deg_matrix, frame_duration, search_window);
  ASSERT_TRUE(res.ok());
  auto best_patches = res.value();
  EXPECT_DOUBLE_EQ(best_patches[0].deg_patch_start_time, 6);
  EXPECT_DOUBLE_EQ(best_patches[1].deg_patch_start_time, 8);
  EXPECT_DOUBLE_EQ(best_patches[2].deg_patch_start_time, 12);
  EXPECT_DOUBLE_EQ(best_patches[3].deg_patch_start_time, 14);
  EXPECT_DOUBLE_EQ(best_patches[4].deg_patch_start_time, 16);
  EXPECT_DOUBLE_EQ(best_patches[5].deg_patch_start_time, 22);
}

}  // namespace
}  // namespace Visqol
