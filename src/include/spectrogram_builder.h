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

#ifndef VISQOL_INCLUDE_SPECTROGRAMBUILDER_H
#define VISQOL_INCLUDE_SPECTROGRAMBUILDER_H

#include "absl/status/statusor.h"
#include "analysis_window.h"
#include "audio_signal.h"
#include "spectrogram.h"

namespace Visqol {

/**
 * This class is used to build a spectrogram representation of a given signal.
 */
class SpectrogramBuilder {
 public:
  /**
   * The deconstructor for the spectrogram builder.
   */
  virtual ~SpectrogramBuilder() {}

  /**
   * For a given signal, build a spectrogram representation utilising the
   * provided window.
   *
   * The input signal will be divided up into segments, the length of which are
   * specified by the analysis window. Each segment is then windowed with a
   * Hamming window of that length. The overlap of each window is also
   * specified by the analysis window.
   *
   * @param signal The signal to produce a spectrogram representation of.
   * @param window The analysis window that specifies the length and overlap of
   *    each Hamming window.
   *
   * @return The spectrogram representation of the input signal.
   */
  virtual absl::StatusOr<Spectrogram> Build(const AudioSignal& signal,
                                            const AnalysisWindow& window) = 0;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_SPECTROGRAMBUILDER_H
