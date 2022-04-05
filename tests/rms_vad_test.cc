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

#include "rms_vad.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

namespace Visqol {
namespace {

const double kTolerance = 0.0001;

struct RmsTestData {
  const std::vector<int16_t> signal;
  const std::vector<double> vad_res;

  RmsTestData(const std::vector<int16_t>& in_signal,
              const std::vector<double>& in_vad_res)
      : signal{in_signal}, vad_res{in_vad_res} {}
};

// Class definition necessary for Value-Parameterized Tests.
class RmsVadTest : public ::testing::TestWithParam<RmsTestData> {};

// Sample data pulled from test signal.
const std::vector<int16_t> kChunk{186,  236,  44, -152, -155, -2, 66, 5,
                                  -108, -107, 14, 141,  151,  31, -90};

// RMS value calculated with Matlab.
const double kChunkRms = 120.7736;

// Chunk size for kSignal.
const size_t kSignalChunkSize = 5;

// Signal with varying quantities of sequential chunks with low activity.
const std::vector<int16_t> kSignal{
    10000, 10000, 10000, 10000, 10000, 10,    10,    10,    10,    10,
    10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,
    10000, 10000, 10000, 10000, 10000, 10,    10,    10,    10,    10,
    10,    10,    10,    10,    10,    10000, 10000, 10000, 10000, 10000,
    10000, 10000, 10000, 10000, 10000, 10,    10,    10,    10,    10,
    10,    10,    10,    10,    10,    10,    10,    10,    10,    10,
    10000, 10000, 10000, 10000, 10000, 10,    10,    10,    10,    10,
    10,    10,    10,    10,    10,    10,    10,    10,    10,    10,
    10,    10,    10,    10,    10,    10000, 10000, 10000, 10000, 10000};

// The vad results for kSignal.
const std::vector<double> kSignalVadRes{1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 0, 1, 1, 1, 0, 0, 1};

/**
 * Ensure that we correctly handle signals that start with low RMS values. The
 * first N chunks in the signal will be marked as containing voice activity.
 * This N value is defined in the code and represents the number of sequential
 * chunks needed before we declare start declaring chunks as lacking voice
 * activity. At the very start of the signal there are no previous chunks to
 * make vad decisions on, so we declare them as having voice activity as we are
 * more interested in avoiding false negatives (and therefore compare patches
 * with no voice activity) than allowing false positives.
 */
const std::vector<int16_t> kSignalLowStart{
    10, 10, 10, 10, 10, 10, 10, 10,    10,    10,    10,    10,   10,
    10, 10, 10, 10, 10, 10, 10, 10000, 10000, 10000, 10000, 10000};

// The vad results for kSignal.
const std::vector<double> kSignalVadResLowStart{1, 1, 0, 0, 1};

/**
 * Test the GetVadResults function with a range of input data.
 */
TEST_P(RmsVadTest, GetVadResultsTests) {
  RmsVad vad;
  size_t counter = 0;
  std::vector<int16_t> chunk;
  for (auto sample : GetParam().signal) {
    chunk.emplace_back(sample);
    if (counter == kSignalChunkSize - 1) {
      vad.ProcessChunk(chunk);
      chunk.clear();
      counter = 0;
    } else {
      counter++;
    }
  }
  ASSERT_TRUE(GetParam().vad_res == vad.GetVadResults());
}

// Initialise the input paramaters for GetVadResultsTests.
INSTANTIATE_TEST_CASE_P(TestParams, RmsVadTest,
                        testing::Values(RmsTestData(kSignal, kSignalVadRes),
                                        RmsTestData(kSignalLowStart,
                                                    kSignalVadResLowStart)));

/**
 * Assert that the ProcessChunk function calculates the correct RMS value for a
 * chunk.
 */
TEST(RmsVadTest, ProcessChunk) {
  RmsVad vad;
  ASSERT_NEAR(kChunkRms, vad.ProcessChunk(kChunk), kTolerance);
}
}  // namespace
}  // namespace Visqol
