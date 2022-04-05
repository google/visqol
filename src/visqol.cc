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

#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/base/internal/raw_logging.h"
#include "absl/status/statusor.h"
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

namespace Visqol {
absl::StatusOr<SimilarityResult> Visqol::CalculateSimilarity(
    const AudioSignal& ref_signal, AudioSignal& deg_signal,
    SpectrogramBuilder* spect_builder, const AnalysisWindow& window,
    const ImagePatchCreator* patch_creator,
    const ComparisonPatchesSelector* comparison_patches_selector,
    const SimilarityToQualityMapper* sim_to_qual_mapper,
    const int search_window) const {
  /////////////////// Stage 1: Preprocessing ///////////////////
  deg_signal =
      MiscAudio::ScaleToMatchSoundPressureLevel(ref_signal, deg_signal);

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
  const double frame_duration =
      CalcFrameDuration(window.size * window.overlap, ref_signal.sample_rate);

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
          sim_match_info, ref_signal, deg_signal, spect_builder, window);
  if (!realign_result.ok()) {
    return realign_result.status();
  }

  sim_match_info = realign_result.value();

  AMatrix<double> fvnsim = CalcPerPatchMeanFreqBandMeans(sim_match_info);
  AMatrix<double> fvnsim10 = CalcPerPatchFreqBandQuantile(sim_match_info, 0.10);
  AMatrix<double> fstdnsim =
      CalcPerPatchMeanFreqBandStdDevs(sim_match_info, frame_duration);
  AMatrix<double> fvdegenergy =
      CalcPerPatchMeanFreqBandDegradedEnergy(sim_match_info);
  double moslqo =
      PredictMos(fvnsim, fvnsim10, fstdnsim, fvdegenergy, sim_to_qual_mapper);

  // calc vnsim
  double sum = 0;
  for (double& d : fvnsim) {
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
  r.fvnsim10 = fvnsim10.ToVector();
  r.fstdnsim = fstdnsim.ToVector();
  r.fvdegenergy = fvdegenergy.ToVector();
  r.moslqo = moslqo;
  r.debug_info = std::move(d);
  r.center_freq_bands = ref_spectrogram.GetCenterFreqBands();
  return r;
}

double Visqol::PredictMos(const AMatrix<double>& fvnsim,
                          const AMatrix<double>& fvnsim10,
                          const AMatrix<double>& fstdnsim,
                          const AMatrix<double>& fvdegenergy,
                          const SimilarityToQualityMapper* mapper) const {
  return mapper->PredictQuality(fvnsim.ToVector(), fvnsim10.ToVector(),
                                fstdnsim.ToVector(), fvdegenergy.ToVector());
}

AMatrix<double> Visqol::CalcPerPatchFreqBandQuantile(
    absl::Span<const PatchSimilarityResult> sim_match_info,
    double quantile) const {
  size_t num_freq_bands = sim_match_info[0].freq_band_means.NumRows();
  AMatrix<double> fvnsim_quantile =
      AMatrix<double>::Filled(num_freq_bands, 1, 0.0);
  for (int band = 0; band < num_freq_bands; ++band) {
    std::vector<double> band_nsims(sim_match_info.size());
    for (int patch_idx = 0; patch_idx < sim_match_info.size(); ++patch_idx) {
      band_nsims[patch_idx] = sim_match_info[patch_idx].freq_band_means(band);
    }
    // Sort the band to get quantile.
    std::sort(band_nsims.begin(), band_nsims.end());
    // Check if the quantile exists.
    int num_elements_in_quantile =
        std::max(1, static_cast<int>(band_nsims.size() * quantile));
    fvnsim_quantile(band) = std::accumulate(
        band_nsims.begin(), band_nsims.begin() + num_elements_in_quantile, 0.0);
    fvnsim_quantile(band) /= num_elements_in_quantile;
  }

  return fvnsim_quantile;
}

AMatrix<double> Visqol::CalcPerPatchMeanFreqBandMeans(
    const std::vector<PatchSimilarityResult>& sim_match_info) const {
  size_t num_freq_bands = sim_match_info[0].freq_band_means.NumRows();
  auto fvnsim = AMatrix<double>::Filled(num_freq_bands, 1, 0.0);
  for (const PatchSimilarityResult& patch : sim_match_info) {
    for (size_t band = 0; band < patch.freq_band_means.NumRows(); band++) {
      fvnsim(band) += patch.freq_band_means(band);
    }
  }
  return fvnsim / sim_match_info.size();
}

AMatrix<double> Visqol::CalcPerPatchMeanFreqBandDegradedEnergy(
    const std::vector<PatchSimilarityResult>& sim_match_info) const {
  // All patches have the same number of frequency bands.
  size_t num_freq_bands = sim_match_info[0].freq_band_deg_energy.NumRows();
  // Create an empty matrix to accumulate the energy over the patches.
  AMatrix<double> total_fvdegenergy =
      AMatrix<double>::Filled(num_freq_bands, 1, 0.0);
  for (const PatchSimilarityResult& patch : sim_match_info) {
    for (size_t band = 0; band < patch.freq_band_deg_energy.NumRows(); band++) {
      total_fvdegenergy(band) += patch.freq_band_deg_energy(band);
    }
  }
  return total_fvdegenergy / sim_match_info.size();
}

AMatrix<double> Visqol::CalcPerPatchMeanFreqBandStdDevs(
    const std::vector<PatchSimilarityResult>& sim_match_info,
    const double frame_duration) const {
  const size_t num_freq_bands = sim_match_info[0].freq_band_means.NumRows();
  AMatrix<double> fvnsim = AMatrix<double>::Filled(num_freq_bands, 1, 0.0);

  AMatrix<double> contribution =
      AMatrix<double>::Filled(num_freq_bands, 1, 0.0);
  for (const PatchSimilarityResult& patch : sim_match_info) {
    for (size_t band = 0; band < patch.freq_band_means.NumRows(); band++) {
      fvnsim(band) += patch.freq_band_means(band);
    }
  }
  fvnsim = fvnsim / sim_match_info.size();

  // Now that we have the global mean, we can compute the combined
  // variance/stddev.
  int total_frame_count = 0;
  for (const PatchSimilarityResult& patch : sim_match_info) {
    const double secs_in_patch =
        (patch.ref_patch_end_time - patch.ref_patch_start_time);
    const int frame_count =
        static_cast<int>(std::ceil(secs_in_patch / frame_duration));
    total_frame_count += frame_count;
    for (size_t band = 0; band < patch.freq_band_means.NumRows(); band++) {
      // Calculate the total variance by combining mean and stddev for each
      // patch. https://en.wikipedia.org/wiki/Pooled_variance
      const double stddev = patch.freq_band_stddevs(band);
      const double mean = patch.freq_band_means(band);
      contribution(band) += (frame_count - 1) * stddev * stddev;
      contribution(band) += frame_count * mean * mean;
    }
  }

  AMatrix<double> result =
      ((contribution - (fvnsim.PointWiseProduct(fvnsim) * total_frame_count)) /
       (total_frame_count - 1));
  // Square root is not defined for AMatrix, so we can't include it as above.
  // Instead, use a transform on the elements. Also, filter out negative numbers
  // due to precision issues that would cause NaNs.
  std::transform(result.begin(), result.end(), result.begin(), [&](double& d) {
    if (d < 0.0) {
      return 0.;
    } else {
      return sqrt(d);
    }
  });
  return result;
}

double Visqol::AlterForSimilarityExtremes(double vnsim, double moslqo) const {
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
