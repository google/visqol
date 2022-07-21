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

#include "training_data_file_reader.h"

#include <fstream>
#include <sstream>

#include "absl/base/internal/raw_logging.h"
#include "absl/strings/numbers.h"
#include "file_path.h"

namespace Visqol {
std::vector<std::vector<double>> TrainingDataFileReader::Read(
    const FilePath& data_filepath, const char delimiter) {
  std::vector<std::vector<double>> values;
  std::vector<double> value_line;
  std::ifstream fin(data_filepath.Path());
  std::string item, line;
  double parsed_double;
  while (getline(fin, line)) {
    std::istringstream in(line);

    while (getline(in, item, delimiter)) {
      if (absl::SimpleAtod(item, &parsed_double)) {
        value_line.push_back(parsed_double);
      } else {
        ABSL_RAW_LOG(ERROR, "Error parsing SVR training data for item: '%s'",
                     item.c_str());
      }
    }

    values.push_back(value_line);
    value_line.clear();
  }
  fin.close();
  return values;
}
}  // namespace Visqol
