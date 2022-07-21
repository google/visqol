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

#include "commandline_parser.h"

#include <string>

#include "gtest/gtest.h"
#include "test_utility.h"

namespace Visqol {
namespace {

const size_t kCountOfBatchFilePairs = 2;
const char kRefFile1[] = "ref_1.wav";
const char kDegFile1[] = "deg_1.wav";
const char kRefFile2[] = "ref_2.wav";
const char kDegFile2[] = "deg_2.wav";

// Test to ensure that a csv batch file can be successfully parsed.
TEST(BuildFilePairPaths, BatchFile) {
  const Visqol::CommandLineArgs cmd_args =
      CommandLineArgsHelper("", "", "testdata/example_batch/batch_input.csv");
  std::vector<ReferenceDegradedPathPair> file_pairs =
      VisqolCommandLineParser::BuildFilePairPaths(cmd_args);
  ASSERT_EQ(file_pairs.size(), kCountOfBatchFilePairs);
  ASSERT_EQ(file_pairs[0].reference.Path(), kRefFile1);
  ASSERT_EQ(file_pairs[0].degraded.Path(), kDegFile1);
  ASSERT_EQ(file_pairs[1].reference.Path(), kRefFile2);
  ASSERT_EQ(file_pairs[1].degraded.Path(), kDegFile2);
}

}  // namespace
}  // namespace Visqol
