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

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "absl/flags/reflection.h"
#include "file_path.h"
#include "gtest/gtest.h"
#include "test_utility.h"

namespace Visqol {
namespace {

std::string CreateTempFile() {
  static int counter = 0;
  std::string path = ::testing::TempDir() + "/tmp_" + std::to_string(counter++);
  std::ofstream(path).close();
  return path;
}

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

TEST(CommandLineParser, SkipMosMapping) {
  absl::FlagSaver flag_saver;
  // Arrange.
  char arg0[] = "program_name";

  std::string ref = CreateTempFile();
  std::string ref_file_flag_string = "--reference_file=" + ref;
  std::vector<char> ref_file_flag(ref_file_flag_string.begin(),
                                  ref_file_flag_string.end());
  ref_file_flag.push_back('\0');

  std::string degraded_file = CreateTempFile();
  std::string degraded_file_flag_string = "--degraded_file=" + degraded_file;
  std::vector<char> degraded_file_flag(degraded_file_flag_string.begin(),
                                       degraded_file_flag_string.end());
  degraded_file_flag.push_back('\0');

  char similarity_to_quality_model_flag[] =
      "--similarity_to_quality_model=non-existent";
  char skip_mos_mapping_flag[] = "--skip_mos_mapping";

  const int kArgc = 5;
  char* argv_array[kArgc + 1] = {arg0,
                                 ref_file_flag.data(),
                                 degraded_file_flag.data(),
                                 similarity_to_quality_model_flag,
                                 skip_mos_mapping_flag,
                                 nullptr};
  char** argv = argv_array;
  // Act.
  auto command_line_flags_or = VisqolCommandLineParser::Parse(kArgc, argv);
  // Assert.
  ASSERT_TRUE(command_line_flags_or.ok());
  auto command_line_flags = command_line_flags_or.value();
  EXPECT_EQ(command_line_flags.reference_signal_path.Path(), ref);
  EXPECT_EQ(command_line_flags.degraded_signal_path.Path(), degraded_file);
  EXPECT_FALSE(
      command_line_flags.similarity_to_quality_mapper_model.has_value());
}

TEST(CommandLineParser, NoSkipMosMapping) {
  absl::FlagSaver flag_saver;
  // Arrange.
  char arg0[] = "program_name";

  std::string ref = CreateTempFile();
  std::string ref_file_flag_string = "--reference_file=" + ref;
  std::vector<char> ref_file_flag(ref_file_flag_string.begin(),
                                  ref_file_flag_string.end());
  ref_file_flag.push_back('\0');

  std::string degraded_file = CreateTempFile();
  std::string degraded_file_flag_string = "--degraded_file=" + degraded_file;
  std::vector<char> deg_file_flag(degraded_file_flag_string.begin(),
                                  degraded_file_flag_string.end());
  deg_file_flag.push_back('\0');

  char no_skip_mos_mapping_flag[] = "--skip_mos_mapping=false";

  const int kArgc = 4;
  char* argv_array[kArgc + 1] = {arg0, ref_file_flag.data(),
                                 deg_file_flag.data(), no_skip_mos_mapping_flag,
                                 nullptr};
  char** argv = argv_array;
  // Act.
  auto command_line_flags_or = VisqolCommandLineParser::Parse(kArgc, argv);
  // Assert.
  ASSERT_TRUE(command_line_flags_or.ok());
  auto command_line_flags = command_line_flags_or.value();
  EXPECT_EQ(command_line_flags.reference_signal_path.Path(), ref);
  EXPECT_EQ(command_line_flags.degraded_signal_path.Path(), degraded_file);
  // similarity_to_quality_mapper_model should be populated with a default
  // value.
  EXPECT_TRUE(
      command_line_flags.similarity_to_quality_mapper_model.has_value());
}

}  // namespace
}  // namespace Visqol
