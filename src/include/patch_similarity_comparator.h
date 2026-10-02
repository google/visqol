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

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
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

  // Prepare one search. The returned candidates are passed to
  // MeasureCandidateSimilarities; backends may instead keep a resident
  // spectrum. Preparation and all scores for a search must be serialized.
  virtual absl::StatusOr<std::vector<ImagePatch>> PrepareCandidateSearch(
      const AMatrix<double>& spectrum, size_t patch_cols) const {
    if (spectrum.NumRows() == 0 || spectrum.NumCols() == 0 || patch_cols == 0) {
      return absl::InvalidArgumentError(
          "Candidate dimensions must be nonempty.");
    }
    std::vector<ImagePatch> patches;
    patches.reserve(spectrum.NumCols());
    for (size_t first = 0; first < spectrum.NumCols(); ++first) {
      ImagePatch patch =
          AMatrix<double>::Filled(spectrum.NumRows(), patch_cols, 0.0);
      const size_t cols = std::min(patch_cols, spectrum.NumCols() - first);
      std::copy_n(spectrum.cbegin() + first * spectrum.NumRows(),
                  cols * spectrum.NumRows(), patch.begin());
      patches.push_back(std::move(patch));
    }
    return patches;
  }

  // Returns NSIM scores for one reference against consecutive candidates.
  // The default implementation preserves CPU behavior.
  virtual absl::StatusOr<std::vector<double>> MeasureCandidateSimilarities(
      const ImagePatch& ref_patch, const std::vector<ImagePatch>& deg_patches,
      size_t first, size_t count) const {
    if (first > deg_patches.size() || count > deg_patches.size() - first) {
      return absl::InvalidArgumentError("Candidate range is out of bounds.");
    }
    std::vector<double> scores;
    scores.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      scores.push_back(
          MeasurePatchSimilarity(ref_patch, deg_patches[first + i]).similarity);
    }
    return scores;
  }
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_PATCHSIMILARITYCOMPARATOR_H
