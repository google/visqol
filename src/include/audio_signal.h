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

#ifndef VISQOL_INCLUDE_AUDIO_SIGNAL_H
#define VISQOL_INCLUDE_AUDIO_SIGNAL_H

#include "amatrix.h"

namespace Visqol {

/**
 * This struct represents an audio signal.
 */
struct AudioSignal {
  /**
   * This matrix is used to store samples. Columns represent channels with each
   * row storing a single sample.
   */
  AMatrix<double> data_matrix;

  /**
   * The sample rate of the audio signal.
   */
  size_t sample_rate;

  /**
   * Get the duration (in seconds) of the signal.
   */
  double GetDuration() const {
    return (data_matrix.NumRows() / static_cast<double>(sample_rate));
  }
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_AUDIO_SIGNAL_H
