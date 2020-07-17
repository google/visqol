/*
 * Copyright 2019 Google LLC, Andrew Hines
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef VISQOL_INCLUDE_FILEPATH_H
#define VISQOL_INCLUDE_FILEPATH_H

#include <fstream>
#include <string>

#include "boost/filesystem.hpp"

namespace Visqol {
class FilePath {
 public:
  FilePath() {}

  FilePath(const FilePath &other) { path_ = other.path_; }

  FilePath(const std::string &path) {
    path_ = ::boost::filesystem::path(path).string();
  }

  const std::string Path() const { return path_; }

  bool Exists() const { return ::boost::filesystem::exists(path_); }

  static std::string currentWorkingDir() {
    return ::boost::filesystem::current_path().string();
  }

 private:
  std::string path_;
};

struct ReferenceDegradedPathPair {
  FilePath reference;
  FilePath degraded;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_FILEPATH_H
