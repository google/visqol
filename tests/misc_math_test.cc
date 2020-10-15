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

#include "misc_math.h"

#include "gtest/gtest.h"

namespace Visqol {
namespace {

// This test was adopted from the ResonanceAudio project:
// https://github.com/resonance-audio/resonance-audio
TEST(MiscMath, NextPowTwoTest) {
  const std::vector<size_t> inputs = {2, 10, 3, 5, 48000, 7, 23, 32};
  const std::vector<size_t> expected = {2, 16, 4, 8, 65536, 8, 32, 32};

  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(expected[i], MiscMath::NextPowTwo(inputs[i]));
  }
}

TEST(MiscMath, ExponentialFromFitTest) {
  // Test some realistic values from NSIM->MOS case where we expect
  // a certain range.
  EXPECT_DOUBLE_EQ(1.4461764166502666,
                   MiscMath::ExponentialFromFit(0.5, 1.15, 4.68, .76));
  EXPECT_DOUBLE_EQ(4.2246774455486502,
                   MiscMath::ExponentialFromFit(1.0, 1.15, 4.68, .76));
}

}  // namespace
}  // namespace Visqol
