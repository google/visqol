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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "commandline_parser.h"
#include "sim_results_writer.h"
#include "visqol_manager.h"

int main(int argc, char** argv) {
  // Parse the command line args.
  auto parse_statusor = Visqol::VisqolCommandLineParser::Parse(argc, argv);
  if (!parse_statusor.ok()) {
    ABSL_RAW_LOG(ERROR, "%s", parse_statusor.status().ToString().c_str());
    return -1;
  }
  Visqol::CommandLineArgs cmd_args = parse_statusor.value();
  auto files_to_compare =
      Visqol::VisqolCommandLineParser::BuildFilePairPaths(cmd_args);

  // Init ViSQOL.
  Visqol::VisqolManager visqol;
  auto init_status = visqol.Init(
      cmd_args.similarity_to_quality_mapper_model, cmd_args.use_speech_mode,
      cmd_args.use_unscaled_speech_mos_mapping, cmd_args.search_window_radius,
      cmd_args.use_lattice_model);
  if (!init_status.ok()) {
    ABSL_RAW_LOG(ERROR, "%s", init_status.ToString().c_str());
    return -1;
  }

  // Iterate over all signal pairs to compare.
  for (const auto& signal_pair : files_to_compare) {
    // Run comparison on a single signal pair.
    auto status_or = visqol.Run(signal_pair.reference, signal_pair.degraded);
    // If successful write value, else log an error.
    if (status_or.ok()) {
      Visqol::SimilarityResultsWriter::Write(
          cmd_args.verbose, cmd_args.results_output_csv,
          cmd_args.debug_output_path, status_or.value(),
          cmd_args.use_speech_mode, cmd_args.use_lattice_model);
    } else {
      ABSL_RAW_LOG(ERROR, "Error executing ViSQOL: %s.",
                   status_or.status().ToString().c_str());
      // A status of aborted gets thrown when visqol hasn't been init'd.
      // So if that happens we want to quit processing.
      if (status_or.status().code() == absl::StatusCode::kAborted) {
        break;
      }
    }
  }

  return 0;
}
