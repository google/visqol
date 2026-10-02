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

#include <algorithm>
#include <random>

#include "envelope.h"
#include "gtest/gtest.h"
#include "misc_vector.h"

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

// Retain the full-spectrum path as an independent numerical oracle for the
// packed FFT implementation, including padded lengths and zero-valued ties.
TEST(XCorr, PackedMatchesFullSpectrumAndWorkspaceSizeChanges) {
  std::mt19937 rng(91);
  std::uniform_real_distribution<double> random(-1.0, 1.0);
  for (size_t size : {size_t{1}, size_t{2}, size_t{7}, size_t{14}, size_t{32},
                      size_t{33}, size_t{256}, size_t{255}, size_t{14}}) {
    for (double level : {0.0, 1.0}) {
      AMatrix<double> ref(size, 1), deg(size + 3, 1);
      for (auto& value : ref) value = level * random(rng);
      for (auto& value : deg) value = level * random(rng);
      int exponent;
      frexp(static_cast<double>(2 * deg.NumRows() - 1), &exponent);
      const size_t points = size_t{1} << exponent;
      auto manager = std::make_unique<FftManager>(points);
      auto a = FastFourierTransform::Forward1d(manager, ref, points);
      auto b = FastFourierTransform::Forward1d(manager, deg, points);
      for (auto& value : b) value = std::conj(value);
      auto corr =
          FastFourierTransform::Inverse1dConjSym(manager, a.PointWiseProduct(b))
              .ToVector();
      const size_t lag = deg.NumRows() - 1;
      std::vector<double> ordered(corr.end() - lag, corr.end());
      ordered.insert(ordered.end(), corr.begin(), corr.begin() + lag + 1);
      const int64_t expected =
          std::max_element(ordered.begin(), ordered.end()) - ordered.begin() -
          static_cast<int64_t>(lag);
      EXPECT_EQ(XCorr::FindLowestLagIndex(ref, deg), expected);
      if (level != 0.0 && points >= FftManager::kMinFftSize) {
        EXPECT_EQ(XCorr::FindLowestLagIndex(deg, ref), -expected);
      }
    }
  }
}

TEST(Envelope, PackedMatchesFullSpectrumIncludingPaddedAndOddLengths) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> random(-1.0, 1.0);
  for (size_t samples : {size_t{14}, size_t{32}, size_t{33}, size_t{1024},
                         size_t{1023}, size_t{14}}) {
    AMatrix<double> input(samples, 1);
    for (auto& value : input) value = random(rng);
    const double mean = MiscVector::Mean(input);
    auto manager = std::make_unique<FftManager>(samples);
    auto freq = FastFourierTransform::Forward1d(manager, input - mean);
    std::vector<double> scaling(freq.NumRows(), 0.0);
    scaling[0] = 1.0;
    scaling[samples / 2] = samples % 2 == 0 ? 1.0 : 2.0;
    const size_t end =
        samples % 2 ? (freq.NumRows() + 1) / 2 : freq.NumRows() / 2;
    for (size_t i = 1; i < end; ++i) scaling[i] = 2.0;
    for (size_t i = 0; i < freq.NumRows(); ++i) freq(i) *= scaling[i];
    const auto hilbert = FastFourierTransform::Inverse1d(manager, freq);
    const auto actual = Envelope::CalcUpperEnv(input);
    for (size_t i = 0; i < samples; ++i)
      EXPECT_EQ(actual(i), std::abs(hilbert(i)) + mean);
  }
}

}  // namespace
}  // namespace Visqol
