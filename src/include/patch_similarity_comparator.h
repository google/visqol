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

#ifndef VISQOL_INCLUDE_PATCHSIMILARITYCOMPARATOR_H
#define VISQOL_INCLUDE_PATCHSIMILARITYCOMPARATOR_H

#include "image_patch_creator.h"

namespace Visqol {
class Spectrogram;

/**
 * Store the result of a comparison between a single reference patch and a
 * single degraded patch.
 */
struct PatchSimilarityResult {
  /**
   * A 1-D matrix with a row for each frequency band compared. Each row stores
   * the similarity score resulting from the comparison between the reference
   * and degraded signals for that particualr frequency band. Used to calculate
   * FVNSIM. Values are stored in order of frequency band, running from the
   * lowest frequency band to the highest.
   */
  AMatrix<double> freq_band_means;

  /**
   * A 1-D matrix with the variance over time of each frequency.
   */
  AMatrix<double> freq_band_stddevs;

  /**
   * A 1-D matrix with the average energy over time of each frequency.
   *
   * The energy here is only for the degraded signal.
   */
  AMatrix<double> freq_band_deg_energy;

  /**
   * The similarity score following a comparison between a single reference
   * patch and a single degraded patch.
   */
  double similarity;

  /**
   * The time (in sec) where this patch starts in the reference signal.
   */
  double ref_patch_start_time;

  /**
   * The time (in sec) where this patch end in the reference signal.
   */
  double ref_patch_end_time;

  /**
   * The time (in sec) where this patch starts in the degraded signal.
   */
  double deg_patch_start_time;

  /**
   * The time (in sec) where this patch ends in the degraded signal.
   */
  double deg_patch_end_time;
};

/**
 * When searching for the best degraded patch to compare to a given reference
 * patch, this struct is used to store the index of the best degraded patch
 * that was found for the comparison, along with the result of that comparison.
 */
struct BestPatchSimilarityMatch {
  /**
   * The result of the comparison between the best matched degraded patch and a
   * given reference patch.
   */
  PatchSimilarityResult result;
};

/**
 * This class provided the logic for comparing two patches.
 */
class PatchSimilarityComparator {
 public:
  /**
   * Destructor for the patch similarity comparator.
   */
  virtual ~PatchSimilarityComparator() {}

  /**
   * For a given reference and degraded patch pair, measure their similarity.
   *
   * @param ref_patch The reference patch.
   * @param deg_patch The degraded patch.
   *
   * @return The patch comparison similarity result.
   */
  virtual PatchSimilarityResult MeasurePatchSimilarity(
      const ImagePatch& ref_patch, const ImagePatch& deg_patch) const = 0;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_PATCHSIMILARITYCOMPARATOR_H
