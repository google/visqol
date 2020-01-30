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

#include <random>

#include "gtest/gtest.h"

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
  static AudioSignal Slice(
      const AudioSignal &in_signal, double start_time, double end_time) {
    return ComparisonPatchesSelector::Slice(in_signal, start_time, end_time);
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
  size_t acceptedNumPatches = selectorPeer.CalcMaxNumPatches(patchIndices,
                                                             slideOffset,
                                                             30);

  EXPECT_EQ(patchIndices.size(), acceptedNumPatches);

  slideOffset = 44;
  acceptedNumPatches = selectorPeer.CalcMaxNumPatches(patchIndices,
                                                      slideOffset,
                                                      30);

  EXPECT_EQ(patchIndices.size() - 1, acceptedNumPatches);
}

TEST_F(ComparisonPatchesSelectorTest, Slice) {
  auto silence_matrix = AMatrix<double>::Filled(16000 * 3, 1, 0.0);
  // Add an impulse at 1.0 secs
  std::vector<double> impulse_vec(1, 1.0);

  silence_matrix.SetRow(16000, impulse_vec);
  AudioSignal three_seconds_silence{silence_matrix, 16000};

  AudioSignal sliced_signal = ComparisonPatchesSelectorPeer::Slice(
      three_seconds_silence, 0.5, 2.5);

  EXPECT_EQ(sliced_signal.GetDuration(), 2.0);

  // Check that the impulse moved to .5 secs.
  EXPECT_EQ(sliced_signal.data_matrix(7999, 0), 0.0);
  EXPECT_EQ(sliced_signal.data_matrix(8000, 0), 1.0);
  EXPECT_EQ(sliced_signal.data_matrix(8001, 0), 0.0);
}

}  // namespace
}  // namespace Visqol
