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

#include "xcorr.h"

#include "gtest/gtest.h"

namespace Visqol {
namespace {

// The reference signal
const AMatrix<double> kReferenceSignal{std::valarray<double>{
    2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0}};

// A degraded signal with a 2 sample lag.
const AMatrix<double> kDegradedSignalLag2{std::valarray<double>{
    1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2, 2.1}};

// A degraded signal (that is longer than the reference) with a 2 sample lag.
const AMatrix<double> kLongDegradedSignalLag2{
    std::valarray<double>{1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8,
                          2.4, 2.2, 2.2, 2.1, 2.0}};

// A degraded signal (that is shorter than the reference) with a 2 sample lag.
const AMatrix<double> kShortDegradedSignalLag2{std::valarray<double>{
    1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2}};

// A degraded signal with a negative 2 sample lag.
const AMatrix<double> kDegradedSignalNegativeLag2{std::valarray<double>{
    2.0, 2.0, 2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0}};

// These lag values were calculated manually from the simple signals above.
const int64_t kBestLagPositive2 = 2;
const int64_t kBestLagNegative2 = -2;

// Test the calculation of the best lag between a reference and degraded signal.
// Test case where the ref and deg signals are the same length.
TEST(XCorr, BestLagSameLength) {
  ASSERT_TRUE(kReferenceSignal.NumElements() ==
              kDegradedSignalLag2.NumElements());
  const int64_t best_lag =
      XCorr::FindLowestLagIndex(kReferenceSignal, kDegradedSignalLag2);
  ASSERT_EQ(kBestLagPositive2, best_lag);
}

// Test the calculation of the best lag between a reference and degraded signal.
// Test case where the ref signal is shorter than the deg signal.
TEST(XCorr, BestLagRefShorter) {
  ASSERT_TRUE(kReferenceSignal.NumElements() <
              kLongDegradedSignalLag2.NumElements());
  const int64_t best_lag =
      XCorr::FindLowestLagIndex(kReferenceSignal, kLongDegradedSignalLag2);
  ASSERT_EQ(kBestLagPositive2, best_lag);
}

// Test the calculation of the best lag between a reference and degraded signal.
// Test case where the ref signal is longer than the deg signal.
TEST(XCorr, BestLagRefLonger) {
  ASSERT_TRUE(kReferenceSignal.NumElements() >
              kShortDegradedSignalLag2.NumElements());
  const int64_t best_lag =
      XCorr::FindLowestLagIndex(kReferenceSignal, kShortDegradedSignalLag2);
  ASSERT_EQ(kBestLagPositive2, best_lag);
}

// Test the calculation of the best lag between a reference and degraded signal.
// Test case where the lag between the signals is negative.
TEST(XCorr, NegativeBestLag) {
  const int64_t best_lag =
      XCorr::FindLowestLagIndex(kReferenceSignal, kDegradedSignalNegativeLag2);
  ASSERT_EQ(kBestLagNegative2, best_lag);
}

}  // namespace
}  // namespace Visqol
