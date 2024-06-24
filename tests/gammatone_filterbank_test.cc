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

#include "gammatone_filterbank.h"

#include "amatrix.h"
#include "equivalent_rectangular_bandwidth.h"
#include "gtest/gtest.h"

namespace Visqol {
namespace {

const size_t kSampleRate = 48000;
const size_t kNumBands = 32;
const double kMinFreq = 50;

// Ensure that the output matrix from the signal filter has the correct
// dimensions.
TEST(ApplyFilterTest, basic_positive_flow) {
  // The contents of this input signal are random figures.
  Eigen::MatrixXd k10Samples (10, 1);
  k10Samples << 0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9;

  auto filter_bank = GammatoneFilterBank{kNumBands, kMinFreq};
  auto erb = EquivalentRectangularBandwidth::MakeFilters(
      kSampleRate, kNumBands, kMinFreq, kSampleRate / 2);
  AMatrix<double> filter_coeffs = AMatrix<double>(erb.filterCoeffs);
  filter_coeffs = filter_coeffs.FlipUpDown();
  filter_bank.SetFilterCoefficients(filter_coeffs);
  auto filtered_signal =
      filter_bank.ApplyFilter(k10Samples);

  ASSERT_EQ(k10Samples.size(), filtered_signal.cols());
  ASSERT_EQ(kNumBands, filtered_signal.rows());
}

}  // namespace
}  // namespace Visqol
