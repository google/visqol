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

#include "vad_patch_creator.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "absl/status/statusor.h"
#include "amatrix.h"
#include "analysis_window.h"
#include "audio_signal.h"
#include "misc_math.h"
#include "rms_vad.h"

namespace Visqol {

const double VadPatchCreator::kFramesWithVAThreshold = 1.0;

std::vector<double> VadPatchCreator::GetVoiceActivity(
    const AudioSignal& signal, const size_t start_sample,
    const size_t total_samples, const size_t frame_len) const {
  RmsVad rms_vad;
  auto sig = signal.data_matrix.GetColumn(0);
  auto patch = sig.GetRows(start_sample, start_sample + total_samples - 1);

  std::vector<int16_t> frame;
  frame.reserve(frame_len);

  for (auto floatVal : patch) {
    // Check the bounds.
    floatVal = floatVal * (1 << 15);
    floatVal =
        std::max(-1.0 * (1 << 15), std::min(1.0 * ((1 << 15) - 1), floatVal));
    frame.emplace_back(floatVal);
    if (frame.size() == frame_len) {
      rms_vad.ProcessChunk(frame);
      frame.clear();
    }
  }

  return rms_vad.GetVadResults();
}

absl::StatusOr<std::vector<size_t>> VadPatchCreator::CreateRefPatchIndices(
    const AMatrix<double>& spectrogram, const AudioSignal& ref_signal,
    const AnalysisWindow& window) const {
  const auto norm_mat = MiscMath::Normalize(ref_signal.data_matrix);
  const AudioSignal norm_sig{norm_mat, ref_signal.sample_rate};
  const double frame_size = window.size * window.overlap;
  const size_t patch_sample_len = patch_size_ * frame_size;
  const size_t spectrum_length = spectrogram.NumCols();
  const size_t first_patch_idx = patch_size_ / 2 - 1;
  const size_t patch_count = (spectrum_length - first_patch_idx) / patch_size_;
  const size_t total_sample_count = patch_count * patch_sample_len;
  std::vector<size_t> refPatchIndices;
  refPatchIndices.reserve(patch_count);

  // Pass the reference signal to the VAD to determine which frames have voice
  // activity.
  const auto vad_res = GetVoiceActivity(norm_sig, first_patch_idx,
                                        total_sample_count, frame_size);

  // Based on the frame VAD data, determine which reference patches to include
  // in the comparison.
  size_t patch_idx = first_patch_idx;
  for (size_t i = 0; i < patch_count; i++) {
    // Take a slice of the VAD data so that we only look at the frames
    // contained within this patch.
    auto first = vad_res.begin() + i * patch_size_;
    auto last = first + patch_size_;
    std::vector<double> patch_vad(first, last);

    // Determine how many frames within this patch contain voice activity.
    auto frames_with_va =
        std::accumulate(patch_vad.begin(), patch_vad.end(), 0.0);

    if (frames_with_va >= kFramesWithVAThreshold) {
      refPatchIndices.push_back(patch_idx);
    }
    patch_idx += patch_size_;
  }

  return refPatchIndices;
}
}  // namespace Visqol
