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

#include "image_patch_creator.h"

#include <utility>
#include <vector>

#include "absl/base/internal/raw_logging.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "amatrix.h"
#include "analysis_window.h"
#include "audio_signal.h"

namespace Visqol {
absl::StatusOr<std::vector<size_t>> ImagePatchCreator::CreateRefPatchIndices(
    const AMatrix<double>& spectrogram, const AudioSignal& ref_signal,
    const AnalysisWindow& window) const {
  return CreateRefPatchIndices(spectrogram);
}

std::vector<ImagePatch> ImagePatchCreator::CreatePatchesFromIndices(
    const AMatrix<double>& spectrogram,
    const std::vector<size_t>& patch_indices) const {
  size_t start_col, end_col;
  size_t num_patches = patch_indices.size();
  std::vector<ImagePatch> patches;
  ImagePatch patch;
  for (size_t i = 0; i < num_patches; i++) {
    start_col = *(patch_indices.begin() + i);
    end_col = start_col + patch_size_ - 1;
    patch = spectrogram.GetColumns(start_col, end_col);
    patches.push_back(std::move(patch));
  }
  return patches;
}

absl::StatusOr<std::vector<size_t>> ImagePatchCreator::CreateRefPatchIndices(
    const AMatrix<double>& spectrogram) const {
  std::vector<size_t> refPatchIndices;
  auto spectrum_length = spectrogram.NumCols();
  auto init_patch_index = patch_size_ / 2;
  // Ensure that the spectrum is at least as big as a single patch
  if (spectrum_length < patch_size_ + init_patch_index) {
    return absl::InvalidArgumentError(
        absl::StrCat("Reference spectrum size (", spectrum_length,
                     ") smaller than minimum patch size (",
                     patch_size_ + init_patch_index, ")."));
  }
  refPatchIndices.reserve(spectrum_length / patch_size_);
  // If we get to this point, the spectrogram can support at least a single
  // patch, so allow for at least one patch to be created.
  auto max_index = (init_patch_index < (spectrum_length - patch_size_))
                       ? spectrum_length - patch_size_
                       : init_patch_index + 1;
  for (size_t i = init_patch_index; i < max_index; i += patch_size_) {
    refPatchIndices.push_back(i - 1);
  }
  return refPatchIndices;
}
}  // namespace Visqol
