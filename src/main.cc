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

#include "absl/base/internal/raw_logging.h"
#include "google/protobuf/stubs/status.h"
#include "google/protobuf/stubs/statusor.h"

#include "commandline_parser.h"
#include "sim_results_writer.h"
#include "visqol_manager.h"

int main(int argc, char **argv) {
  // Parse the command line args.
  auto parse_statusor = Visqol::VisqolCommandLineParser::Parse(argc, argv);
  if (!parse_statusor.ok()) {
    ABSL_RAW_LOG(ERROR, "%s",
        parse_statusor.status().error_message().ToString().c_str());
    return -1;
  }
  Visqol::CommandLineArgs cmd_args = parse_statusor.ValueOrDie();
  auto files_to_compare = Visqol::VisqolCommandLineParser::BuildFilePairPaths(
      cmd_args);

  // Init ViSQOL.
  Visqol::VisqolManager visqol;
  auto init_status = visqol.Init(cmd_args.sim_to_quality_mapper_model,
      cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping);
  if (!init_status.ok()) {
    ABSL_RAW_LOG(ERROR, "%s",
        init_status.error_message().ToString().c_str());
    return -1;
  }

  // Run ViSQOL.
  auto sim_result_msgs = visqol.Run(files_to_compare);

  // Write the results.
  for (const auto& sim_result_msg : sim_result_msgs) {
    Visqol::SimilarityResultsWriter::Write(
        cmd_args.verbose, cmd_args.results_output_csv, cmd_args.debug_output_path,
        sim_result_msg, cmd_args.use_speech_mode);
  }

  return 0;
}
