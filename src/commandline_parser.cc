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

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Placeholder for runfiles.
#include "absl/base/attributes.h"
#include "absl/base/internal/raw_logging.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

ABSL_FLAG(std::string, reference_file, "",
          "The wav file path used as the reference audio.");
ABSL_FLAG(std::string, degraded_file, "",
          "The wav file path used as the degraded audio.");
ABSL_FLAG(std::string, batch_input_csv, "",
          "Used to specify a path to a CSV file with the format: \n"
          "------------------\n"
          "reference,degraded\n"
          "ref1.wav,deg1.wav\n"
          "ref2.wav,deg2.wav\n"
          "------------------\n"
          "If the `batch_input_csv` flag is used, the `reference_file` \n"
          "and `degraded_file` flags will be ignored.");
ABSL_FLAG(std::string, results_csv, "",
          "Used to specify a path that the similarity score results will be "
          "output to \n"
          ". This will be a CSV file with the format:\n"
          "------------------\n"
          "reference,degraded,moslqo\n"
          "ref1.wav,deg1.wav,3.4\n"
          "ref2.wav,deg2.wav,4.1\n");
ABSL_FLAG(bool, verbose, false, "Enables verbose output in the terminal.");
ABSL_FLAG(std::string, output_debug, "",
          "Used to specify a file path where output debug information will be "
          "written\n"
          "to. This debug info contains the full details of the comparison "
          "between the\n"
          "reference and degraded audio signals and is in JSON format. The "
          "file does\n"
          "not need to previously exist. Contents will be appended to the file "
          "if it\n"
          "does already exist or if ViSQOL is run in batch mode.");
ABSL_FLAG(std::string, similarity_to_quality_model, "",
          "The libsvm model to use during comparison. Use this only if you "
          "want to explicitly specify the model file location, otherwise the "
          "default model will be used.");
ABSL_FLAG(bool, use_speech_mode, false,
          "Use a wideband model (sensitive up to 8kHz) with voice activity "
          "detection\n"
          "that normalizes the polynomial NSIM->MOS mapping so that a perfect "
          "NSIM\n"
          "score of 1.0 translates to 5.0.");
ABSL_FLAG(bool, use_lattice_model, true,
          "Use a deep lattice network model to map similarity to quality. "
          "This produces more accurate results for speech (audio mode is "
          "not yet supported.");
ABSL_FLAG(bool, use_unscaled_speech_mos_mapping, false,
          "When used in conjunction with --use_speech_mode, this flag will "
          "prevent a\n"
          "perfect NSIM score of 1.0 being translated to a MOS score of 5.0. "
          "Perfect\n"
          "NSIM scores will instead result in MOS scores of ~4.x.");
ABSL_FLAG(int, search_window_radius, 60,
          "The search_window parameter determines how far the algorithm will "
          "search to discover patch matches. For a given reference frame, it "
          "will look at 2*search_window_radius + 1 patches to find the most "
          "optimal match.");

namespace Visqol {
ABSL_CONST_INIT const char kDefaultAudioModelFile[] =
    "/model/libsvm_nu_svr_model.txt";
ABSL_CONST_INIT const char kDefaultSpeechModelFile[] =
    "/model/"
    "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400"
    "_train1_7_raw.tflite";

absl::StatusOr<CommandLineArgs> VisqolCommandLineParser::Parse(int argc,
                                                               char** argv) {
  absl::SetProgramUsageMessage(
      "Perceptual quality estimator for speech and audio");
  absl::ParseCommandLine(argc, argv);

  bool error_found = false;
  FilePath reference_file;
  FilePath degraded_file;
  FilePath similarity_to_quality_model;
  FilePath result_output_csv;
  FilePath batch_input;
  FilePath debug_output;
  bool verbose;
  bool use_speech;
  bool use_lattice_model;
  bool use_unscaled_mapping;
  int search_window;

  batch_input = FilePath(absl::GetFlag(FLAGS_batch_input_csv));
  if (!batch_input.Path().empty()) {
    error_found |= !FileExists(batch_input);
  } else {
    reference_file = FilePath(absl::GetFlag(FLAGS_reference_file));
    error_found |= !FileExists(reference_file);

    degraded_file = FilePath(absl::GetFlag(FLAGS_degraded_file));
    error_found |= !FileExists(degraded_file);
  }

  result_output_csv = FilePath(absl::GetFlag(FLAGS_results_csv));
  use_unscaled_mapping = absl::GetFlag(FLAGS_use_unscaled_speech_mos_mapping);
  use_speech = absl::GetFlag(FLAGS_use_speech_mode);
  use_lattice_model = absl::GetFlag(FLAGS_use_lattice_model);
  verbose = absl::GetFlag(FLAGS_verbose);
  search_window = absl::GetFlag(FLAGS_search_window_radius);
  debug_output = FilePath(absl::GetFlag(FLAGS_output_debug));

  similarity_to_quality_model =
      FilePath(absl::GetFlag(FLAGS_similarity_to_quality_model));
  // The quality model file is only relevant for SVR in audio mode,
  // so check that before validating
  if (!similarity_to_quality_model.Path().empty() && !use_speech) {
    error_found |= !FileExists(similarity_to_quality_model);
  }

  if (error_found) {
    return absl::InvalidArgumentError(
        "Invalid command line argument detected. "
        "Run with --helpfull for usage.");
  }
  if (similarity_to_quality_model.Path().empty()) {
    if (use_speech) {
      similarity_to_quality_model =
          FilePath(FilePath::currentWorkingDir() + kDefaultSpeechModelFile);
    } else {
      similarity_to_quality_model =
          FilePath(FilePath::currentWorkingDir() + kDefaultAudioModelFile);
    }
    if (!similarity_to_quality_model.Path().empty() &&
        !FileExists(similarity_to_quality_model)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to load the default SVR model ",
          similarity_to_quality_model.Path(),
          ". Specify the correct path using '--similarity_to_quality_model"
          " <path/to/libsvm_nu_svr_model.txt>'?"));
    }
  }

  return CommandLineArgs{
      .reference_signal_path = reference_file,
      .degraded_signal_path = degraded_file,
      .similarity_to_quality_mapper_model = similarity_to_quality_model,
      .results_output_csv = result_output_csv,
      .batch_input_csv = batch_input,
      .debug_output_path = debug_output,
      .verbose = verbose,
      .use_speech_mode = use_speech,
      .use_unscaled_speech_mos_mapping = use_unscaled_mapping,
      .search_window_radius = search_window,
      .use_lattice_model = use_lattice_model};
}

std::vector<ReferenceDegradedPathPair>
VisqolCommandLineParser::ReadFilesToCompare(const FilePath& batch_input_path) {
  std::vector<ReferenceDegradedPathPair> file_paths;
  std::vector<FilePath> line_file_paths;
  std::ifstream fin(batch_input_path.Path());
  std::string item, line;
  const char delimiter = ',';

  if (fin) {
    getline(fin, line);  // skip the header

    while (getline(fin, line)) {
      // getline will read up to \n, so in cases where the line ending is \r\n,
      // we need to manually strip the \r.
      if (line.back() == '\r') {
        line.pop_back();
      }
      std::istringstream in(line);

      while (std::getline(in, item, delimiter)) {
        line_file_paths.push_back(FilePath(item));
      }
      file_paths.push_back({line_file_paths[0], line_file_paths[1]});
      line_file_paths.clear();
    }
    fin.close();
  }

  return file_paths;
}

bool VisqolCommandLineParser::FileExists(const FilePath& path) {
  bool exists = path.Exists();
  if (!exists) {
    ABSL_RAW_LOG(ERROR, "File not found: %s", path.Path().c_str());
  }
  return exists;
}

std::vector<ReferenceDegradedPathPair>
VisqolCommandLineParser::BuildFilePairPaths(const CommandLineArgs& cmd_res) {
  std::vector<ReferenceDegradedPathPair> pairs;
  if (!cmd_res.batch_input_csv.Path().empty()) {
    pairs = ReadFilesToCompare(cmd_res.batch_input_csv);
  } else if (cmd_res.reference_signal_path.Exists() &&
             cmd_res.degraded_signal_path.Exists()) {
    pairs.push_back(
        {cmd_res.reference_signal_path, cmd_res.degraded_signal_path});
  }
  return pairs;
}
}  // namespace Visqol
