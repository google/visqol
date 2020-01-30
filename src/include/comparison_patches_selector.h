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

#include "google/protobuf/stubs/statusor.h"

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
   * similar degraded patch that matches it and compare the two patches. Return
   * the set of all such comparisons.
   *
   * @param ref_patches A vector containing all of the patches created from the
   *    reference spectrogram.
   * @param ref_patch_indices The indices for the set of reference patches. Each
   *    index corresponds to the index of the column in the reference
   *    spectrogram where this patch starts from.
   * @param spectrogram_data The spectrogram that represents the degraded
   *    signal.
   *
   * @return A vector of similarity results. Each similary result is the result
   *    of the comparison between a patch from the reference spectrogram with
   *    its corresponding patch in the degraded spectrogram.
   */
  google::protobuf::util::StatusOr<std::vector<PatchSimilarityResult>>
      FindMostSimilarDegPatches(const std::vector<ImagePatch> &ref_patches,
          const std::vector<size_t> &ref_patch_indices,
          const AMatrix<double> &spectrogram_data,
          const double frame_duration) const;

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
  google::protobuf::util::StatusOr<std::vector<PatchSimilarityResult>>
      FinelyAlignAndRecreatePatches(
          const std::vector<PatchSimilarityResult>& sim_results,
          const AudioSignal &ref_signal,
          const AudioSignal &deg_signal,
          SpectrogramBuilder *spect_builder,
          const AnalysisWindow &window) const;

 private:
  /**
   * Extract a subregion of an audio signal.
   *
   * @param in_signal An AudioSignal.
   * @param start_time The start time of the sliced audio in seconds.
   * @param end_time The end time of the sliced audio in seconds.
   * @return A new AudioSignal containing the subregion.
   */
  static AudioSignal Slice(const AudioSignal &in_signal, double start_time,
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
  ImagePatch BuildDegradedPatch(const AMatrix<double> &spectrogram_data,
                                int window_beginning, size_t window_end,
                                size_t window_height, size_t window_width)
                                const;

  /**
   * For a given patch from the reference spectrogram, find the most similar
   * degraded patch from within the given bounds in the degraded spectrogram.
   *
   * This function takes the provided ref_frame_index, constructs patches that
   * occur before and after, comparing it to the provided reference patch. This
   * process is repreated until the patch that starts at the provided end index
   * has been crated and compared. The result of the most similar patch
   * comparison is returned.
   *
   * @param spectrogram_data The spectrogram that represents the degraded
   *    signal.
   * @param ref_patch The reference patch to find the best match for.
   * @param ref_frame_index The index of the column in the spectrogram data from
   *    around which to construct patches for comparison.
   * @param frame_duration The duration of a frame in seconds.
   *
   * @return The similarity result from the comparison of the most similar
   *    degraded patch with the reference patch.
   */
  PatchSimilarityResult FindMostSimilarDegPatch(
      const AMatrix<double> &spectrogram_data, const ImagePatch &ref_patch,
      int ref_frame_index, const double frame_duration) const;

  /**
   * Calculate the maximum number of patches that the degraded spectrogram can
   * support.
   *
   * @param ref_patch_indices The indices for the set of reference patches. Each
   *    index corresponds to the index of the column in the reference
   *    spectrogram where this patch starts from.
   * @param num_deg_frames The number of frames in the degraded spectrogram.
   * @param num_frames The number of frames in a patch.
   *
   * @return The maximum number of patches that the degraded spectrogram can
   *    support.
   */
  size_t CalcMaxNumPatches(const std::vector<size_t> &ref_patch_indices,
                           size_t max_slide_offset, size_t num_frames) const;

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
