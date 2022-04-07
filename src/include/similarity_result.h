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

#ifndef VISQOL_INCLUDE_SIMILARITYRESULT_H
#define VISQOL_INCLUDE_SIMILARITYRESULT_H

#include <vector>

#include "file_path.h"
#include "patch_similarity_comparator.h"

namespace Visqol {
/**
 * Struct used for storing debug information related to a specific
 * similarity comparison result.
 */
struct SimilarityDebugInfo {
  /**
   * Vector of results from the comparison of each reference and degraded patch
   * pair.
   */
  std::vector<PatchSimilarityResult> patch_sims;
};

/**
 * Struct used for storing the result of a similarity comparison.
 */
struct SimilarityResult {
  /**
   * The predicted Mean Opinion Score - Listening Quality Objective (scored 1
   * to 5)
   */
  double moslqo;

  /**
   * The mean of the FVNSIM values.
   */
  double vnsim;

  /**
   * Represents the similarity between the two signals for each frequency band
   * that is compared. The size of this vector will therefore equal the number
   * of frequency bands compared. Values are ordered from the lowest frequency
   * band at index 0, running up to the highest.
   */
  std::vector<double> fvnsim;

  /**
   * Represents the similarity between the two signals for each frequency band
   * that is compared for the 10 percentile (0.1 quantile).
   * The size of this vector will therefore equal the number of frequency bands
   * compared. Values are ordered from the lowest frequency band at index 0,
   * running up to the highest.
   */
  std::vector<double> fvnsim10;

  /**
   * Standard deviation in similarity for each frequency.
   */
  std::vector<double> fstdnsim;

  /**
   * Degraded energy for each frequency.
   */
  std::vector<double> fvdegenergy;

  /**
   * Stores the center frequency bands that the above FVNSIM scores correspond
   * to. Values are stored running from the lowest frequency band to the
   * highest.
   */
  std::vector<double> center_freq_bands;

  /**
   * The debug information related to this similarity comparison.
   */
  SimilarityDebugInfo debug_info;

  /**
   * If the reference audio signal was read in from file, this will store the
   * path to this file.
   */
  FilePath reference;

  /**
   * If the degraded audio signal was read in from file, this will store the
   * path to this file.
   */
  FilePath degraded;

  /**
   * If the degraded audio was additionally aligned, this will store the value
   * received from alignment.cc
   */
  double alignment_lag_s;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_SIMILARITYRESULT_H
