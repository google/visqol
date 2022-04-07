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

#ifndef VISQOL_INCLUDE_COMPARISON_PATCHES_SELECTOR_H
#define VISQOL_INCLUDE_COMPARISON_PATCHES_SELECTOR_H

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "amatrix.h"
#include "image_patch_creator.h"
#include "patch_similarity_comparator.h"
#include "spectrogram_builder.h"

namespace Visqol {
struct PatchSimilarityResult;

/**
 * This class is used for creating and comparing patches from the degraded
 * spectrogram with a given set of reference patches.
 */
class ComparisonPatchesSelector {
 public:
  /**
   * Constructor that takes a patch similarity comparator for performing the
   * patch comparison.
   */
  ComparisonPatchesSelector(
      std::unique_ptr<PatchSimilarityComparator> sim_comparator);

  /**
   * For each patch provided (from the reference spectrogram) find the most
   * optimal degraded patch within a given search space that matches it such
   * that the cumulative similarity score across all reference patches is
   * maximized. It returns the set of all such comparisons.
   *
   * @param ref_patches A vector containing all of the patches created from the
   *    reference spectrogram.
   * @param ref_patch_indices The indices for the set of reference patches. Each
   *    index corresponds to the index of the column in the reference
   *    spectrogram where this patch starts from.
   * @param spectrogram_data The spectrogram that represents the degraded
   *    signal.
   * @param frame_duration The duration of the frame in seconds.
   * @param search_window_radius The parameter that determines how far you
   *    should search to discover patch matches. For a given reference frame,
   *    one looks at 2*search_window_radius + 1 patches to find the most
   *    optimal match.
   *
   * @return A vector of similarity results. Each similary result is the result
   *    of the comparison between a patch from the reference spectrogram with
   *    its corresponding patch in the degraded spectrogram.
   */
  absl::StatusOr<std::vector<PatchSimilarityResult>> FindMostOptimalDegPatches(
      const std::vector<ImagePatch>& ref_patches,
      const std::vector<size_t>& ref_patch_indices,
      const AMatrix<double>& spectrogram_data, const double frame_duration,
      const int search_window_radius) const;

  /**
   * Given roughly aligned ref/deg patches, realign the original audio within
   * the patch size so that they are maximally locally aligned.
   *
   * @param sim_results A vector of PatchSimilarityResults
   * @param ref_signal The reference signal used to create the
   *    PatchSimilarityResults.
   * @param deg_signal The degraded signal used to create the
   *    PatchSimilarityResults.
   * @param spect_builder A pointer to a SpectrogramBuilder.
   * @param window An AnalysisWindow used to create the spectrogram
   *
   * @return A StatusOr that may contain a vector of new, finely aligned
   *    PatchSimilarityResults.
   */
  absl::StatusOr<std::vector<PatchSimilarityResult>>
  FinelyAlignAndRecreatePatches(
      const std::vector<PatchSimilarityResult>& sim_results,
      const AudioSignal& ref_signal, const AudioSignal& deg_signal,
      SpectrogramBuilder* spect_builder, const AnalysisWindow& window) const;

 private:
  /**
   * Extract a subregion of an audio signal.
   *
   * @param in_signal An AudioSignal.
   * @param start_time The start time of the sliced audio in seconds.
   * @param end_time The end time of the sliced audio in seconds.
   *
   * @return A new AudioSignal containing the subregion.
   */
  static AudioSignal Slice(const AudioSignal& in_signal, double start_time,
                           double end_time);

  /**
   * Based on the input locations and dimensions, build a patch from the input
   * degraded spectrogram.
   *
   * @param spectrogram_data The spectrogram that represents the degraded
   *    signal.
   * @param window_beginning The index of the column in the degraded spectrogram
   *    where this patch should start from. This column index is inclusive.
   * @param window_end The index of the column in the degraded spectrogram
   *    where this patch should end. This column index is inclusive.
   * @param window_height The number of rows needed in the output patch.
   * @param window_width The number of columns needed in the output patch.
   *
   * @return The patch built from the degraded spectrogram.
   */
  ImagePatch BuildDegradedPatch(const AMatrix<double>& spectrogram_data,
                                int window_beginning, size_t window_end,
                                size_t window_height,
                                size_t window_width) const;

  /**
   * For a given patch from the reference spectrogram, find the most optimal
   * degraded patch, such that it maximizes the cumulative similarity score
   * calculated from the 0th patch index to the current one from within the
   * given bounds(search_window patches on either side) in the degraded
   * spectrogram.
   *
   * This function takes the provided ref_frame_index, constructs all patches in
   * the degarded signal that occur in the given bounds, comparing it to the
   * provided reference patch and stores the cumulative similarity score formed
   * till this reference patch in the cumulative_similarity_dp vector. The
   * backtrace vector is used to store the offset where the previous reference
   * frame matched the best. Returns nothing but populates the
   * cumulative_similarity_dp vector and backtrace vector accordingly.
   *
   * @param spectrogram_data The spectrogram that represents the degraded
   *    signal.
   * @param ref_patch The reference patch to find the best match for.
   * @param cumulative_similarity_dp A 2D array to record the cumulative
   *    similarity scores from reference patches to degraded patches.
   * @param backtrace A 2D array to record the matching patch information of
   *    previous patch indices.
   * @param ref_patch_indices The indices for the set of reference patches. Each
   *    index corresponds to the index of the column in the reference
   *    spectrogram where this patch starts from.
   * @param patch_index The patch number in the multiple patches created by
   *    patch_creator.
   * @param search_window The search space parameter that determines how far you
   *    should search in to discover patch matches. For a given reference frame,
   *    one looks at 2*search_window + 1 frames to find the most optimal match.
   *
   * @return The function returns nothing. It's purpose is to populate the
   *    cumulative_similarity_dp and backtrace vectors.
   */
  void FindMostOptimalDegPatch(
      const AMatrix<double>& spectrogram_data, const ImagePatch& ref_patch,
      std::vector<ImagePatch>& deg_patches,
      std::vector<std::vector<double>>& cumulative_similarity_dp,
      std::vector<std::vector<int>>& backtrace,
      const std::vector<size_t>& ref_patch_indices, int patch_index,
      const int search_window) const;

  /**
   * Calculate the maximum number of patches that the degraded spectrogram can
   * support.
   *
   * @param ref_patch_indices The indices for the set of reference patches. Each
   *    index corresponds to the index of the column in the reference
   *    spectrogram where this patch starts from.
   * @param num_frames_in_deg_spectro The number of frames in the degraded
   *    spectrogram.
   * @param num_frames_per_patch The number of frames in a patch.
   *
   * @return The maximum number of patches that the degraded spectrogram can
   *    support.
   */
  size_t CalcMaxNumPatches(const std::vector<size_t>& ref_patch_indices,
                           size_t num_frames_in_deg_spectro,
                           size_t num_frames_per_patch) const;

  /**
   * Friend class used for unit testing.
   */
  friend class ComparisonPatchesSelectorPeer;

  /**
   * The patch comparator to use for comparisons.
   */
  const std::unique_ptr<PatchSimilarityComparator> sim_comparator_;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_COMPARISON_PATCHES_SELECTOR_H
