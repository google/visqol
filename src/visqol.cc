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

#include "visqol.h"

#include <utility>
#include <vector>

#include "amatrix.h"
#include "comparison_patches_selector.h"
#include "file_path.h"
#include "image_patch_creator.h"
#include "misc_audio.h"
#include "patch_similarity_comparator.h"
#include "similarity_result.h"
#include "similarity_to_quality_mapper.h"
#include "spectrogram.h"
#include "spectrogram_builder.h"
#include "absl/base/internal/raw_logging.h"
#include "absl/status/statusor.h"

namespace Visqol {
absl::StatusOr<SimilarityResult> Visqol::CalculateSimilarity(
    const AudioSignal &ref_signal, AudioSignal &deg_signal,
    SpectrogramBuilder *spect_builder, const AnalysisWindow &window,
    const ImagePatchCreator *patch_creator,
    const ComparisonPatchesSelector *comparison_patches_selector,
    const SimilarityToQualityMapper *sim_to_qual_mapper,
    const int search_window) const {
  /////////////////// Stage 1: Preprocessing ///////////////////
  deg_signal = MiscAudio::ScaleToMatchSoundPressureLevel(ref_signal,
      deg_signal);

  // build the reference spectrogram.
  const auto ref_spectro_result = spect_builder->Build(ref_signal, window);
  if (!ref_spectro_result.ok()) {
    ABSL_RAW_LOG(ERROR, "Error building reference spectrogram: %s",
                 ref_spectro_result.status().ToString().c_str());
    return ref_spectro_result.status();
  }

  // build the degraded spectrogram.
  const auto deg_spectro_result = spect_builder->Build(deg_signal, window);
  if (!deg_spectro_result.ok()) {
    ABSL_RAW_LOG(ERROR, "Error building degraded spectrogram: %s",
                 deg_spectro_result.status().ToString().c_str());
    return deg_spectro_result.status();
  }

  Spectrogram ref_spectrogram = ref_spectro_result.value();
  Spectrogram deg_spectrogram = deg_spectro_result.value();
  MiscAudio::PrepareSpectrogramsForComparison(ref_spectrogram, deg_spectrogram);

  /////////////// Stage 2: Feature selection and similarity measure ////////////
  auto ref_patch_result = patch_creator->CreateRefPatchIndices(
      ref_spectrogram.Data(), ref_signal, window);
  if (!ref_patch_result.ok()) {
    ABSL_RAW_LOG(ERROR, "Error creating reference patch indices: %s",
                 ref_patch_result.status().ToString().c_str());
    return ref_patch_result.status();
  }
  auto ref_patch_indices = ref_patch_result.value();
  const double frame_duration = CalcFrameDuration(window.size * window.overlap,
                                                  ref_signal.sample_rate);

  auto ref_patches = patch_creator->CreatePatchesFromIndices(
      ref_spectrogram.Data(), ref_patch_indices);
  auto most_sim_patch_result =
      comparison_patches_selector->FindMostOptimalDegPatches(
          ref_patches, ref_patch_indices, deg_spectrogram.Data(),
          frame_duration, search_window);
  if (!most_sim_patch_result.ok()) {
    return most_sim_patch_result.status();
  }
  auto sim_match_info = most_sim_patch_result.value();

  // Realign the patches in time domain subsignals that start at the coarse
  // patch times.
  auto realign_result =
      comparison_patches_selector->FinelyAlignAndRecreatePatches(
          sim_match_info, ref_signal, deg_signal, spect_builder,
          window);
  if (!realign_result.ok()) {
    return realign_result.status();
  }

  sim_match_info = realign_result.value();

  auto fvnsim = CalcPerPatchMeanFreqBandMeans(sim_match_info);
  double moslqo = PredictMos(fvnsim, sim_to_qual_mapper);

  // calc vnsim
  double sum = 0;
  for (auto &d : fvnsim) {
    sum += d;
  }
  double vnsim = sum / fvnsim.NumRows();

  moslqo = AlterForSimilarityExtremes(vnsim, moslqo);

  // gather results
  SimilarityDebugInfo d;
  d.patch_sims = std::move(sim_match_info);
  SimilarityResult r;
  r.vnsim = vnsim;
  r.fvnsim = fvnsim.ToVector();
  r.moslqo = moslqo;
  r.debug_info = std::move(d);
  r.center_freq_bands = ref_spectrogram.GetCenterFreqBands();
  return r;
}

double Visqol::PredictMos(const AMatrix<double> &fvnsim,
                               const SimilarityToQualityMapper *mapper) const {
  double predicted_quality = mapper->PredictQuality(fvnsim.ToVector());
  return predicted_quality;
}

// calc fvnsim
AMatrix<double> Visqol::CalcPerPatchMeanFreqBandMeans(
    const std::vector<PatchSimilarityResult> &sim_match_info) const {
  size_t num_freq_bands = sim_match_info[0].freq_band_means.NumRows();
  auto fvnsim = AMatrix<double>::Filled(num_freq_bands, 1, 0.0);
  for (auto &p : sim_match_info) {
    auto &patch_freq_band_means = p.freq_band_means;
    for (size_t band = 0; band < patch_freq_band_means.NumRows(); band++) {
      fvnsim(band) += patch_freq_band_means(band);
    }
  }
  return fvnsim / sim_match_info.size();
}

double Visqol::AlterForSimilarityExtremes(double vnsim,
                                               double moslqo) const {
  // Stop totally dissimilar signals from getting a good score.
  // The SVM is trained on the same songs with different quality.
  // In situations where it's given a fvnsim for two completely
  // different songs, is returns a pretty random moslqo.
  // This is is prevent that unwanted behaviour.
  // The threshold is based on the Andrew's voice data poly curve
  // and probably deserves more investigation.

  // Also, if the two signals are very similar, make it a perfect score.
  // It's not perfect from the SVR because people regularly mistake the
  // reference for a slightly worse than perfect signal, often giving
  // scores of 4.8 and 4.9 instead of 5. This function just rounds it.
  if (vnsim < 0.15) {
    moslqo = 1;
  }
  // It makes sense to have very similar audio clips give perfect score,
  // but subjective tests showed that people doubt the quality of the
  // reference, giving, say 4.8 instead of 5 MOS, so to be more like
  // people, a perfect score probably shouldn't be returned.
  // if (vnsim > 0.99)
  //{
  //  moslqo = 5;
  //}
  return moslqo;
}

double Visqol::CalcFrameDuration(const size_t frame_size,
                                      const size_t sample_rate) const {
  return frame_size / static_cast<double>(sample_rate);
}
}  // namespace Visqol
