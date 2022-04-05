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

#ifndef VISQOL_INCLUDE_ANALYSIS_WINDOW_H
#define VISQOL_INCLUDE_ANALYSIS_WINDOW_H

#include <math.h>

#include <cstddef>
#include <valarray>

namespace Visqol {
/**
 * Struct used for creating windows during spectrogram creation.
 */
struct AnalysisWindow {
  /**
   * The desired size of the temporal window in seconds.
   */
  double window_duration;

  /**
   * The size of the window.
   */
  size_t size;

  /**
   * The overlap of the window.
   */
  double overlap;

  /**
   * Constructs an instance of this AnalysisWindow struct.
   *
   * @param sample_rate The sample rate of the signal to be analyzed.
   * @param win_overlap The overlap of the window.
   */
  AnalysisWindow(const size_t sample_rate, const double win_overlap,
                 double window_duration = .08);

  std::valarray<double> ApplyHannWindow(
      const std::valarray<double>& signal) const;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_ANALYSIS_WINDOW_H
