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

#ifndef VISQOL_INCLUDE_ALIGNMENT_H
#define VISQOL_INCLUDE_ALIGNMENT_H

#include <utility>

namespace Visqol {
struct AudioSignal;

/**
 * Perform an alignment on two given signals. Used to adjust for codec initial
 * padding.
 */
class Alignment {
 public:
  /**
   * For a given reference signal, align a second degraded signal with it,
   * returning a new aligned degraded signal.
   *
   * @param reference_signal The reference signal.
   * @param degraded_signal The degraded signal to align.
   * @return A tuple of the aligned degraded signal and its lag in seconds.
   */
  static std::tuple<AudioSignal, double> GloballyAlign(
      const AudioSignal& reference_signal, const AudioSignal& degraded_signal);
  /**
   * Aligns a degraded signal to the reference signal, truncating them to
   * be the same length.
   *
   * @param reference_signal The reference signal.
   * @param degraded_signal The degraded signal.
   * @return A std::tuple of two new signals and the lag of the degraded.
   *   The start position will be what it was for the reference_signal, and the
   *   durations will be truncated as needed so that they are the same length.
   **/
  static std::tuple<AudioSignal, AudioSignal, double> AlignAndTruncate(
      const AudioSignal& reference_signal, const AudioSignal& degraded_signal);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_ALIGNMENT_H
