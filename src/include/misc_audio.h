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

#ifndef VISQOL_INCLUDE_MISCAUDIO_H
#define VISQOL_INCLUDE_MISCAUDIO_H

#include <vector>

#include "absl/types/optional.h"
#include "amatrix.h"
#include "audio_signal.h"
#include "file_path.h"
#include "misc_math.h"
#include "spectrogram.h"

namespace Visqol {

class MiscAudio {
 public:
  /**
   * The number of channels in a mono signal.
   */
  static const size_t kNumChanMono;

  /**
   * Constant value for a sample of zero.
   */
  static const double kZeroSample;

  /**
   * The sound pressure level reference point which is equal to the threshold
   * of human hearing (20 micropascals).
   */
  static const double kSplReferencePoint;

  /**
   * For a given pair of reference and degraded audio signals, scale the sound
   * pressure level (spl) of the degraded signal so that it matches the spl of
   * the reference signal.
   *
   * @param reference The reference signal whose spl is to be matched.
   * @param degraded The degraded signal whose spl is to be scaled.
   *
   * @return The new degraded signal with an spl that has been scaled to match
   *    that of the input reference signal.
   */
  static AudioSignal ScaleToMatchSoundPressureLevel(
      const AudioSignal& reference, const AudioSignal& degraded);

  /**
   * For a given audio file, load it in mono. Files with more than 1 channel
   * will be downmixed to mono.
   *
   * Currently only WAV files are supported.
   *
   * @param path The path to the audio file to load.
   *
   * @return The mono audio signal.
   */
  static AudioSignal LoadAsMono(const FilePath& path);

  /**
   * For a given audio stream, load it in mono. Audio with more than 1 channel
   * will be downmixed to mono.
   *
   * Currently only WAV streams are supported.
   *
   * @param string_stream Audio stream to load.
   *
   * @param filepath Optional filepath for logging purposes.
   *
   * @return The mono audio signal.
   */
  static AudioSignal LoadAsMono(
      std::stringstream* string_stream,
      absl::optional<std::string> filepath = absl::nullopt);

  /**
   * Performs some basic preparation on the input spectrograms so that they are
   * suitable for comparison to each other.
   *
   * @param reference The reference spectrogram.
   * @param degraded The degraded spectrogram.
   */
  static void PrepareSpectrogramsForComparison(Spectrogram& reference,
                                               Spectrogram& degraded);

 private:
  /**
   * For a given audio signal, downmix it to mono. If already mono, no work is
   * performed. The downmixing is performed by simply averaging the value of
   * each sample at each index across the channels.
   *
   * Assumes that channels are stored in columns with the samples in rows.
   *
   * @param signal The audio signal to downmix to mono.
   *
   * @return The downmixed mono audio signal.
   */
  static AudioSignal ToMono(const AudioSignal& signal);

  /**
   * For a given data matrix, downmix it to mono. If already mono, no work is
   * performed. The downmixing is performed by simply averaging the value of
   * each sample at each index across the channels.
   *
   * Assumes that channels are stored in columns with the samples in rows.
   *
   * @param signal The data matrix to downmix to mono.
   *
   * @return The downmixed mono data matrix.
   */
  static AMatrix<double> ToMono(const AMatrix<double>& signal);

  /**
   * For a given audio signal, calculate (in dB) the sound pressure level.
   *
   * Sound pressure level is calculated using the following formula:
   * 20log(p/pref) where 'p' is the signal's sound pressure and 'pref' is a
   * reference equal to the threshold of human hearing.
   *
   * @param signal The signal to calculate sound pressure level for.
   *
   * @return The signal's sound pressure level in dB.
   */
  static double CalcSoundPressureLevel(const AudioSignal& signal);

  /**
   * Takes a vector that contains samples from multiple channels that have been
   * interleaved and extracts each individual channel from the interleaved
   * vector. Each extracted channel is placed in an individual vector and then
   * all channels are returned in a vector of vectors.
   *
   * As an example, for a stero file, the interleaving is assumed to be in the
   * format: [C1-S1, C2-S1, C1-S2, C2-S2, C1-S3, C2-S3] where C is Channel
   * and S is Sample.
   *
   * @param num_channels The number of channels in the interleaved vector.
   * @param interleaved_vector The vector of multi-channel interleaved samples.
   *
   * @return The extracted channels stored in individual vectors.
   */
  static std::vector<std::vector<double>> ExtractMultiChannel(
      const int num_channels, const std::vector<double>& interleaved_vector);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_MISCAUDIO_H
