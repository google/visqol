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

#ifndef VISQOL_INCLUDE_VISQOLAUDIO_H
#define VISQOL_INCLUDE_VISQOLAUDIO_H

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "analysis_window.h"
#include "audio_signal.h"
#include "comparison_patches_selector.h"
#include "file_path.h"
#include "image_patch_creator.h"
#include "similarity_result.h"
#include "similarity_to_quality_mapper.h"
#include "spectrogram.h"
#include "spectrogram_builder.h"

namespace Visqol {

/**
 * This class is used for performing a similarity comparison on two input
 * signals
 */
class Visqol {
 public:
  /**
   * Perform a comparison on two audio signals. Their similarity is calculated
   * and converted to a quality score using the given similarity to quality
   * mapper.
   *
   * @param ref_signal The reference signal for comparison.
   * @param deg_signal The degraded signal for comparison.
   * @param spect_builder The spectrogram builder used for building spectrograms
   *    from the above input signals.
   * @param window The Hamming window used for analysis of the signals.
   * @param patch_creator Used for creating patches for comparison from the
   *    signal's spectrograms.
   * @param comparison_patches_selector Used for selecting and comparing patches
   *    from the degraded signal with those from the reference signal.
   * @param sim_to_qual_mapper Used to convert a similarity score to a quality
   *    score.
   * @param search_window This parameter is used to determine how far the
   *    algorithm will search in order to find the most optimal match.
   *
   * @return If the comparison was successful, return the similarity result and
   *    associated debug info. Else, return an error status.
   */
  absl::StatusOr<SimilarityResult> CalculateSimilarity(
      const AudioSignal& ref_signal, AudioSignal& deg_signal,
      SpectrogramBuilder* spect_builder, const AnalysisWindow& window,
      const ImagePatchCreator* patch_creator,
      const ComparisonPatchesSelector* comparison_patches_selector,
      const SimilarityToQualityMapper* sim_to_qual_mapper,
      const int search_window) const;

  /**
   * Produces a set of FVNSIM scores, which represent the similarity between
   * the two signals for each frequency band. This is done by calculating the
   * mean of a certain quantile's similarity score for each frequency band for
   * the lowest patches in the quantile.
   * This is not a true quantile, because if there are not enough patches in the
   * signals, the lowest patch will always be returned.  So rather, the true
   * quantile is max(.1, 1 / num_patches).
   *
   * @param sim_match_info The similarity scores for each patch comparison.
   * @param quantile The quantile from 0.0 to 1.0.  Patches that are above this
   *    quantile are not included in the computation.
   *
   * @return The resulting set of FVNSIM scores.
   */
  AMatrix<double> CalcPerPatchFreqBandQuantile(
      absl::Span<const PatchSimilarityResult> sim_match_info,
      double quantile) const;

 private:
  /**
   * For a given set of FVNSIM scores, which represent the similarity between
   * the two signals for each frequency band, perform a similarity to quality
   * mapping to produce a quality score i.e. a mean opinion score (MOS).
   *
   * @param fvnsim An AMatrix of neurogram similarity (NSIM) score means
   *     indexed by frequency band.  This typically comes from the patch
   *     comparison step.
   * @param fvnsim10 An AMatrix of neurogram similarity (NSIM) score means
   *     indexed by frequency band averaged over all the patches that fall
   *     in the 10th lowest quantile.
   * @param fstdnsim An AMatrix of neurogram similarity score standard
   *     deviations indexed by frequency band.
   * @param fvdegenergy An AMatrix of mean energy in the degraded signal
   *     indexed by frequency band.
   * @param mapper The similarity to quality mapper to use for the prediction.
   *
   * @return The predicted MOS score.
   */
  double PredictMos(const AMatrix<double>& fvnsim,
                    const AMatrix<double>& fvnsim10,
                    const AMatrix<double>& fstdnsim,
                    const AMatrix<double>& fvdegenergy,
                    const SimilarityToQualityMapper* mapper) const;

  /**
   * Produces a set of FVNSIM scores, which represent the similarity between
   * the two signals for each frequency band. This is done by calculating the
   * mean similarity score for each frequency band across all compared patches.
   *
   * @param sim_match_info The similarity scores for each patch comparison.
   *
   * @return The resulting set of FVNSIM scores.
   */
  AMatrix<double> CalcPerPatchMeanFreqBandMeans(
      const std::vector<PatchSimilarityResult>& sim_match_info) const;

  /**
   * Produces a set of FVNSIM scores' stddev in similarity between
   * the two signals for each frequency band. This is done by calculating the
   * mean similarity score for each frequency band across all compared patches.
   *
   * @param sim_match_info The similarity scores for each patch comparison.
   * @param frame_duration The frame duration in seconds.
   *
   * @return The resulting set of FSTDNSIM scores.
   */
  AMatrix<double> CalcPerPatchMeanFreqBandStdDevs(
      const std::vector<PatchSimilarityResult>& sim_match_info,
      const double frame_duration) const;

  /**
   * Produces a set of per frequency energies in the degraded signal.
   *
   * @param sim_match_info The similarity scores for each patch comparison.
   *
   * @return The resulting set of per frequency energies.
   */
  AMatrix<double> CalcPerPatchMeanFreqBandDegradedEnergy(
      const std::vector<PatchSimilarityResult>& sim_match_info) const;

  /**
   * This function alters the resulting MOS-LQO score in cases where the audio
   * files are massively dissimilar e.g. two completely different audio files.
   *
   * @param vnsim The mean of the FVNSIM scores.
   * @param moslqo The MOS-LQO that was produced for the given set of FVNSIM
   *    scores.
   *
   * @return If the given VNSIM is below a certain threshold value, a constant
   *    MOS-LQO of 1 is returned. Else, the input MOS-LQO is returned.
   */
  double AlterForSimilarityExtremes(double vnsim, double moslqo) const;

  /**
   * Calculate the duration of a frame in seconds.
   *
   * @param frame_size The size of the frame in samples.
   * @param sample_rate The sample rate for the signal that the frame belongs
   *    to.
   *
   * @return The duration of the frame in seconds.
   */
  double CalcFrameDuration(const size_t frame_size,
                           const size_t sample_rate) const;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_VISQOLAUDIO_H
