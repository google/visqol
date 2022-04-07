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

#ifndef VISQOL_INCLUDE_VADPATCHCREATOR_H
#define VISQOL_INCLUDE_VADPATCHCREATOR_H

#include <vector>

#include "absl/status/statusor.h"
#include "analysis_window.h"
#include "audio_signal.h"
#include "image_patch_creator.h"

namespace Visqol {
/**
 * A class that takes in a signal and returns voice activity data for it.
 *
 * The main task of the VAD for speech mode is to reject silence on the
 * reference file. This implementation utilizes a simple RMS approach. The VAD
 * is only run on the reference audio, which is less likely to have significant
 * noise.
 *
 * The VAD threshold was determined through experimentation.
 */
class VadPatchCreator : public ImagePatchCreator {
 public:
  /**
   * The number of frames that must have voice activity present in the patch
   * for us to include the patch in the comparison.
   */
  static const double kFramesWithVAThreshold;

  // Docs inherited from parent.
  explicit VadPatchCreator(size_t patch_size) : ImagePatchCreator(patch_size) {}

  // Docs inherited from parent.
  absl::StatusOr<std::vector<size_t>> CreateRefPatchIndices(
      const AMatrix<double>& spectrogram, const AudioSignal& ref_signal,
      const AnalysisWindow& window) const override;

  /**
   * For a given input signal and sample bounds, break the signal up into
   * frames and for each frame determine if there is voice activity present.
   *
   * @param signal The input signal.
   * @param start_sample The sample to start running the VAD on in the input
   *    signal.
   * @param total_samples The total number of samples to include in the VAD,
   *    starting from the given starting sample.
   * @param frame_len The length of each frame that should be pased to the
   *    VAD.
   *
   * @return A vector containing the VAD results for each frame taken from
   *    the input signal. Results are in order of frame position within the
   *    signal i.e. first frame is at index 0. If a given frame had voice
   *    activity, a value of 1.0 will be stored at that index in the vector.
   *    Else a value of 0.0 is stored.
   */
  std::vector<double> GetVoiceActivity(const AudioSignal& signal,
                                       const size_t start_sample,
                                       const size_t total_samples,
                                       const size_t frame_len) const;
};

}  // namespace Visqol
#endif  // VISQOL_INCLUDE_VADPATCHCREATOR_H
