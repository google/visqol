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

#ifndef VISQOL_TESTS_TEST_UTILITY_H
#define VISQOL_TESTS_TEST_UTILITY_H

#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "amatrix.h"
#include "commandline_parser.h"
#include "gtest/gtest.h"

namespace Visqol {
inline Visqol::CommandLineArgs CommandLineArgsHelper(
    absl::string_view reference_file, absl::string_view degraded_file,
    absl::string_view batch_file = "", bool speech_mode = false,
    bool unscaled_speech = false, int search_window = 60,
    bool use_lattice = true) {
  const char* model_path =
      speech_mode ? kDefaultSpeechModelFile : kDefaultAudioModelFile;
  const std::string similarity_to_quality_model =
      absl::StrCat(FilePath::currentWorkingDir(), model_path);
  return CommandLineArgs{.reference_signal_path = reference_file,
                         .degraded_signal_path = degraded_file,
                         .similarity_to_quality_mapper_model =
                             FilePath(similarity_to_quality_model),
                         .batch_input_csv = batch_file,
                         .use_speech_mode = speech_mode,
                         .use_unscaled_speech_mos_mapping = unscaled_speech,
                         .search_window_radius = search_window,
                         .use_lattice_model = use_lattice};
}

// Perform basic matrix dimension comparison.
template <typename T>
bool CompareMatrix(const AMatrix<T> matrix_a, const AMatrix<T> matrix_b,
                   std::string* fail_msg) {
  if (matrix_a.NumCols() != matrix_b.NumCols()) {
    *fail_msg = absl::StrCat(
        "Matrices do not match! matrix_a num cols: ", matrix_a.NumCols(),
        " matrix_b num cols: ", matrix_b.NumCols());
    return false;
  } else if (matrix_a.NumRows() != matrix_b.NumRows()) {
    *fail_msg = absl::StrCat(
        "Matrices do not match! matrix_a num rows: ", matrix_a.NumRows(),
        " matrix_b num rows: ", matrix_b.NumRows());
    return false;
  } else {
    return true;
  }
}

// Perform comparison of matrix containing doubles
bool CompareDoubleMatrix(const AMatrix<double> matrix_a,
                         const AMatrix<double> matrix_b, const double tolerance,
                         std::string* fail_msg) {
  bool basic_comp = CompareMatrix(matrix_a, matrix_b, fail_msg);
  if (basic_comp) {
    int i = 0;
    for (auto it_a = matrix_a.cbegin(), it_b = matrix_b.cbegin();
         it_a != matrix_a.cend(); it_a++, it_b++) {
      if (std::abs(*it_a - *it_b) > tolerance) {
        *fail_msg = absl::StrCat("Matrices do not match! At index ", i,
                                 " matrix_a abs val: ", std::abs(*it_a),
                                 " matrix_b abs val: ", std::abs(*it_b));
        return false;
      }
      i++;
    }
  }
  return basic_comp;
}

// Perform comparison of matrix containing complex numbers.
bool CompareComplexMatrix(const AMatrix<std::complex<double>> matrix_a,
                          const AMatrix<std::complex<double>> matrix_b,
                          const double tolerance, std::string* fail_msg) {
  bool basic_comp = CompareMatrix(matrix_a, matrix_b, fail_msg);
  if (basic_comp) {
    int i = 0;
    for (auto it_a = matrix_a.cbegin(), it_b = matrix_b.cbegin();
         it_a != matrix_a.cend(); it_a++, it_b++) {
      if (std::abs(it_a->real() - it_b->real()) > tolerance ||
          std::abs(it_a->imag() - it_b->imag()) > tolerance) {
        *fail_msg = absl::StrCat("Matrices do not match! At index ", i,
                                 " matrix_a abs val: ", std::abs(*it_a),
                                 " matrix_b abs val: ", std::abs(*it_b));
        return false;
      }
      i++;
    }
  }
  return basic_comp;
}

}  // namespace Visqol

#endif  // VISQOL_TESTS_TEST_UTILITY_H
