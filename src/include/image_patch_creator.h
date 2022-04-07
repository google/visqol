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

#ifndef VISQOL_INCLUDE_IMAGEPATCHCREATOR_H
#define VISQOL_INCLUDE_IMAGEPATCHCREATOR_H

#include <vector>

#include "absl/status/statusor.h"
#include "amatrix.h"
#include "analysis_window.h"
#include "audio_signal.h"

namespace Visqol {
using ImagePatch = AMatrix<double>;

/**
 * Class used for creating patches from a spectrogram.
 */
class ImagePatchCreator {
 public:
  /**
   * Constructor for the patch creator for patches of the specified size.
   *
   * @param patch_size The required patch size.
   */
  explicit ImagePatchCreator(size_t patch_size) : patch_size_(patch_size) {}

  virtual ~ImagePatchCreator() {}

  /**
   * For the given spectrogram, create a vector of patch indices. Each index
   * corresponds to the index of the column in the spectrogram where the patch
   * starts from.
   *
   * @param spectrogram The spectrogram to create patch indices for.
   * @param ref_signal The reference audio signal.
   * @param window The analysis window that was used when building the
   *    spectrogram.
   *
   * @param If successful, the vector of patch indices is returned. Else, an
   *    error status.
   */
  virtual absl::StatusOr<std::vector<size_t>> CreateRefPatchIndices(
      const AMatrix<double>& spectrogram, const AudioSignal& ref_signal,
      const AnalysisWindow& window) const;

  /**
   * For a given spectrogram and vector of patch indices, create a vector of
   * patches.
   *
   * @param spectrogram The spectrogram to create patches from.
   * @param patch_indices The indices for the set of patches. Each index
   *    corresponds to the index of the column in the spectrogram where this
   *    patch starts from.
   *
   * @return The vector of patches.
   */
  std::vector<ImagePatch> CreatePatchesFromIndices(
      const AMatrix<double>& spectrogram,
      const std::vector<size_t>& patch_indices) const;

 protected:
  /**
   * The number of frames that each patch should contain. A single frame is
   * represented by a single column in the spectrogram.
   */
  size_t patch_size_;

 private:
  /**
   * For the given spectrogram, create a vector of patch indices. Each index
   * corresponds to the index of the column in the spectrogram where the patch
   * starts from.
   *
   * @param spectrogram The spectrogram to create patch indices for.
   *
   * @param If successful, the vector of patch indices is returned. Else, an
   *    error status.
   */
  absl::StatusOr<std::vector<size_t>> CreateRefPatchIndices(
      const AMatrix<double>& spectrogram) const;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_IMAGEPATCHCREATOR_H
