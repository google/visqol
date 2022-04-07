// Copyright 2021 Google LLC
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

#include "visqol.h"

#include "gtest/gtest.h"
#include "patch_similarity_comparator.h"

namespace Visqol {
namespace {

/**
 *  Test the ViSQOL quantile-based similarity functions.
 */
TEST(VisqolCalculations, CalcPerPatchFreqBandQuantile) {
  Visqol visqol;
  std::vector<PatchSimilarityResult> patch_similarities;

  // Construct a hundred patches with decreasing nsims and two bands.
  // They are decreasing to check that the quantile function doesn't just grab
  // the first few entries.
  for (int i = 100; i > 0; --i) {
    PatchSimilarityResult patch;
    patch.freq_band_means.Resize(2, 1);
    // The first band's fvnsim will range from 1 to 100
    patch.freq_band_means(0) = i;
    // The second band's fvnsim will range from 11 to 110
    patch.freq_band_means(1) = i + 10.0;
    patch_similarities.push_back(patch);
  }
  AMatrix<double> result =
      visqol.CalcPerPatchFreqBandQuantile(patch_similarities, 0.10);
  // The average of the 10% quantile should be the average the integers between
  // 1 and 10.
  EXPECT_EQ(result(0), 5.5);
  // The second band is the same, but offset by 10.
  EXPECT_EQ(result(1), 15.5);
}

}  // namespace
}  // namespace Visqol
