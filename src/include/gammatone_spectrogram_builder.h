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

#ifndef VISQOL_INCLUDE_GAMMATONESPECTROGRAMBUILDER_H
#define VISQOL_INCLUDE_GAMMATONESPECTROGRAMBUILDER_H

#include "absl/status/statusor.h"
#include "gammatone_filterbank.h"
#include "spectrogram_builder.h"

namespace Visqol {

/**
 * This class provides a Gammatone filter based implementation for building a
 * spectrogram representation of a signal. Gammatone filters were designed to
 * match with experimental observations of how mammalian cochlea process
 * auditory signals.
 *
 * Based on Dan Ellis Matlab implementation:
 * https://uk.mathworks.com/matlabcentral/fileexchange/23053-gammatone-based--auditory--spectrograms/content/gammatonegram/gammatonegram.m
 */
class GammatoneSpectrogramBuilder : public SpectrogramBuilder {
 public:
  /**
   * The maximum frequency to be used during speech mode.
   */
  static const double kSpeechModeMaxFreq;

  /**
   * Constructs an instance of this GammatoneSpectrogramBuilder using the
   * provided GammatoneFilterBank.
   *
   * @param filter_bank The gamatone filter bank to apply to the signal.
   */
  explicit GammatoneSpectrogramBuilder(const GammatoneFilterBank& filter_bank,
                                       const bool use_speech_mode);

  // Docs inherited from parent.
  absl::StatusOr<Spectrogram> Build(const AudioSignal& signal,
                                    const AnalysisWindow& window) override;

 private:
  /**
   * The gammatone filter bank to apply to the signal.
   */
  GammatoneFilterBank filter_bank_;

  /**
   * If true, build the spectrogram for speech mode.
   */
  bool speech_mode_;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_GAMMATONESPECTROGRAMBUILDER_H
