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

#include "comparison_patches_selector.h"

#include <assert.h>

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/internal/raw_logging.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "alignment.h"
#include "amatrix.h"
#include "audio_signal.h"
#include "image_patch_creator.h"
#include "misc_audio.h"
#include "patch_similarity_comparator.h"
#include "status_macros.h"

namespace Visqol {
ComparisonPatchesSelector::ComparisonPatchesSelector(
    std::unique_ptr<PatchSimilarityComparator> sim_comparator)
    : sim_comparator_{std::move(sim_comparator)} {}

absl::Status ComparisonPatchesSelector::FindMostOptimalDegPatch(
    const AMatrix<double>& spectrogram_data, const ImagePatch& ref_patch,
    std::vector<ImagePatch>& deg_patches,
    std::vector<std::vector<double>>& cumulative_similarity_dp,
    std::vector<std::vector<int>>& backtrace,
    const std::vector<size_t>& ref_patch_indices, int patch_index,
    const int search_window) const {
  int ref_frame_index = ref_patch_indices[patch_index];
  const int first = std::max(0, ref_frame_index - search_window);
  const int last = std::min(static_cast<int>(spectrogram_data.NumCols()) - 1,
                            ref_frame_index + search_window);
  if (first > last) return absl::OkStatus();
  std::vector<double> scores;
  VISQOL_ASSIGN_OR_RETURN(scores,
                          sim_comparator_->MeasureCandidateSimilarities(
                              ref_patch, deg_patches, first, last - first + 1));

  // For a given reference frame index, this function compares the given
  // reference patch with all possible degraded patches in the search window and
  // populates the cumulative_similarity_dp vector accordingly. For more details
  // : https://en.wikipedia.org/wiki/Dynamic_time_warping

  double highest_sim = std::numeric_limits<double>::lowest();
  int past_best = -1;
  int next_predecessor =
      patch_index > 0
          ? std::max(0, static_cast<int>(ref_patch_indices[patch_index - 1]) -
                            search_window)
          : 0;
  for (int slide_offset = first; slide_offset <= last; ++slide_offset) {
    double similarity = scores[slide_offset - first];

    int past_slide_offset = -1;
    if (patch_index > 0) {
      // Maintain the best predecessor as the candidate position advances.
      // Ties select the largest index, matching the original reverse scan.
      while (next_predecessor < slide_offset) {
        const double value =
            cumulative_similarity_dp[patch_index - 1][next_predecessor];
        if (value > highest_sim || (past_best >= 0 && value == highest_sim)) {
          highest_sim = value;
          past_best = next_predecessor;
        }
        ++next_predecessor;
      }
      past_slide_offset = past_best;
      similarity += highest_sim;
      // If the current reference patch experienced a packet loss, then the
      // cumulative similarity score till the previous patch might be more and
      // in that case no matching patch for the current reference patch is found
      // in the degraded window.
      if (cumulative_similarity_dp[patch_index - 1][slide_offset] >
          similarity) {
        similarity = cumulative_similarity_dp[patch_index - 1][slide_offset];
        past_slide_offset = slide_offset;
      }
    }
    cumulative_similarity_dp[patch_index][slide_offset] = similarity;
    backtrace[patch_index][slide_offset] = past_slide_offset;
  }
  return absl::OkStatus();
}

size_t ComparisonPatchesSelector::CalcMaxNumPatches(
    const std::vector<size_t>& ref_patch_indices,
    size_t num_frames_in_deg_spectro, size_t num_frames_per_patch) const {
  size_t num_patches = ref_patch_indices.size();

  if (num_patches) {
    // The last patch can start up to half a patch away.
    while (ref_patch_indices[num_patches - 1] -
               floor(num_frames_per_patch / 2) >
           num_frames_in_deg_spectro) {
      num_patches--;
    }
  }

  return num_patches;
}

absl::StatusOr<std::vector<PatchSimilarityResult>>
ComparisonPatchesSelector::FindMostOptimalDegPatches(
    const std::vector<ImagePatch>& ref_patches,
    const std::vector<size_t>& ref_patch_indices,
    const AMatrix<double>& spectrogram_data, const double frame_duration,
    const int search_window_radius) const {
  const size_t num_frames_per_patch = ref_patches[0].NumCols();
  const size_t num_frames_in_deg_spectro = spectrogram_data.NumCols();
  const double patch_duration = frame_duration * num_frames_per_patch;
  const int search_window = search_window_radius * num_frames_per_patch;
  const size_t num_patches = CalcMaxNumPatches(
      ref_patch_indices, num_frames_in_deg_spectro, num_frames_per_patch);

  if (!num_patches) {
    return absl::Status(
        absl::StatusCode::kCancelled,
        "Degraded file was too short, different, or misaligned to score any "
        "of the reference patches.");
  } else if (num_patches < ref_patch_indices.size()) {
    ABSL_RAW_LOG(
        WARNING,
        "Warning: Dropping %zu (of %zu) reference patches "
        "due to the degraded file being misaligned or too short. If too many "
        "patches are dropped, the score will be less meaningful.",
        ref_patch_indices.size() - num_patches, ref_patch_indices.size());
  }
  // The vector to store the similarity results
  std::vector<PatchSimilarityResult> bestDegPatches(num_patches);
  std::vector<std::vector<double>> cumulative_similarity_dp(
      ref_patch_indices.size(),
      std::vector<double>(spectrogram_data.NumCols()));
  std::vector<std::vector<int>> backtrace(
      ref_patch_indices.size(), std::vector<int>(spectrogram_data.NumCols()));
  std::vector<ImagePatch> deg_patches;
  VISQOL_ASSIGN_OR_RETURN(deg_patches,
                          sim_comparator_->PrepareCandidateSearch(
                              spectrogram_data, num_frames_per_patch));
  // Attempt to get a good alignment with backtracking.
  for (size_t patch_index = 0; patch_index < num_patches; patch_index++) {
    // Find the best alignment to the ref patch within a distance of
    // search_window on each side of the hard-aligned deg signal.
    VISQOL_RETURN_IF_ERROR(FindMostOptimalDegPatch(
        spectrogram_data, ref_patches[patch_index], deg_patches,
        cumulative_similarity_dp, backtrace, ref_patch_indices, patch_index,
        search_window));
  }
  double max_similarity_score = std::numeric_limits<double>::lowest();
  // The patch index for the last reference patch.
  int last_index = num_patches - 1;
  // The last_offset stores the offset at which the last reference patch got the
  // maximal similarity score over all the reference patches.
  int last_offset;
  int lower_limit = std::max(
      0, static_cast<int>(ref_patch_indices[last_index] - search_window));
  // The for loop is used to find the offset which maximizes the similarity
  // score across all the patches.
  for (int slide_offset = lower_limit;
       slide_offset <= ref_patch_indices[last_index] + search_window;
       slide_offset++) {
    if (slide_offset >= num_frames_in_deg_spectro) {
      // The frame offset for degraded start patch cannot be more than the
      // number of frames in the degraded spectrogram.
      break;
    }
    if (cumulative_similarity_dp[last_index][slide_offset] >
        max_similarity_score) {
      max_similarity_score = cumulative_similarity_dp[last_index][slide_offset];
      last_offset = slide_offset;
    }
  }

  for (int patch_index = num_patches - 1; patch_index >= 0; patch_index--) {
    // This sets the reference and degraded patch start and end times.
    ImagePatch ref_patch = ref_patches[patch_index];
    ImagePatch deg_patch = BuildDegradedPatch(
        spectrogram_data, last_offset, last_offset + ref_patch.NumCols() - 1,
        ref_patch.NumRows(), ref_patch.NumCols());
    bestDegPatches[patch_index] =
        sim_comparator_->MeasurePatchSimilarity(ref_patch, deg_patch);
    // This condition is true only if no matching patch was found for the given
    // reference patch. In this case, the matched patch is essentially set to
    // NULL (which is different from a silent patch).
    if (last_offset == backtrace[patch_index][last_offset]) {
      bestDegPatches[patch_index].deg_patch_start_time = 0.0;
      bestDegPatches[patch_index].deg_patch_end_time = 0.0;
      bestDegPatches[patch_index].similarity = 0.0;
      int num_rows = bestDegPatches[patch_index].freq_band_means.NumRows();
      int num_cols = bestDegPatches[patch_index].freq_band_means.NumCols();
      bestDegPatches[patch_index].freq_band_means =
          bestDegPatches[patch_index].freq_band_means.Filled(num_rows, num_cols,
                                                             0.0);
    } else {
      bestDegPatches[patch_index].deg_patch_start_time =
          last_offset * frame_duration;
      bestDegPatches[patch_index].deg_patch_end_time =
          bestDegPatches[patch_index].deg_patch_start_time + patch_duration;
    }
    bestDegPatches[patch_index].ref_patch_start_time =
        ref_patch_indices[patch_index] * frame_duration;
    bestDegPatches[patch_index].ref_patch_end_time =
        bestDegPatches[patch_index].ref_patch_start_time + patch_duration;
    last_offset = backtrace[patch_index][last_offset];
  }
  return bestDegPatches;
}

ImagePatch ComparisonPatchesSelector::BuildDegradedPatch(
    const AMatrix<double>& spectrogram_data, int window_beginning,
    size_t window_end, size_t window_height, size_t window_width) const {
  ImagePatch deg_patch{window_height, window_width};
  std::vector<double> row;

  // We should allow negative starts to allow the degraded signal to come first.
  // Each row is a frequency band.

  int first_real_frame = std::max(0, window_beginning);
  // This is an inclusive end, so subtract 1.
  int last_real_frame = std::min(window_end, spectrogram_data.NumCols() - 1);
  for (size_t rowIndex = 0; rowIndex < spectrogram_data.NumRows(); rowIndex++) {
    row =
        spectrogram_data.RowSubset(rowIndex, first_real_frame, last_real_frame);

    // Insert silence at front for negative indices.
    if (window_beginning < 0) {
      row.insert(row.begin(), 0 - window_beginning, 0.0);
    }

    if (window_end > spectrogram_data.NumCols() - 1) {
      row.insert(row.end(), window_end - (spectrogram_data.NumCols() - 1), 0.0);
    }

    deg_patch.SetRow(rowIndex, std::move(row));
  }
  return deg_patch;
}

AudioSignal ComparisonPatchesSelector::Slice(const AudioSignal& in_signal,
                                             double start_time,
                                             double end_time) {
  int start_index = std::max(0, (int)(start_time * in_signal.sample_rate));
  // The end_index is inclusive for GetRows().
  int end_index = std::min((int)(in_signal.data_matrix.NumRows() - 1),
                           (int)(end_time * in_signal.sample_rate));

  // Note that the underlying armadillo rows() uses an unconventional inclusive
  // end index.
  auto sliced_matrix =
      in_signal.data_matrix.GetRows(start_index, end_index - 1);
  // Adds silence at the end of degraded patch, if required for alignment.
  auto end_time_diff =
      end_time * in_signal.sample_rate - in_signal.data_matrix.NumRows();
  if (end_time_diff > 0) {
    auto postsilence_matrix = AMatrix<double>::Filled(end_time_diff, 1, 0.0);
    sliced_matrix = postsilence_matrix.JoinVertically(sliced_matrix);
  }
  if (start_time < 0) {
    auto presilence_matrix = AMatrix<double>::Filled(
        -1 * start_time * in_signal.sample_rate, 1, 0.0);
    sliced_matrix = presilence_matrix.JoinVertically(sliced_matrix);
  }
  AudioSignal sliced_signal{sliced_matrix, in_signal.sample_rate};
  return sliced_signal;
}

absl::StatusOr<std::vector<PatchSimilarityResult>>
ComparisonPatchesSelector::FinelyAlignAndRecreatePatches(
    const std::vector<PatchSimilarityResult>& sim_results,
    const AudioSignal& ref_signal, const AudioSignal& deg_signal,
    SpectrogramBuilder* spect_builder, const AnalysisWindow& window) const {
  std::vector<PatchSimilarityResult> realigned_results(sim_results.size());

  // Bound host/device workspace while combining independent local spectra.
  constexpr size_t kPatchBatchSize = 16;
  for (size_t batch = 0; batch < sim_results.size(); batch += kPatchBatchSize) {
    const size_t end = std::min(sim_results.size(), batch + kPatchBatchSize);
    std::vector<AudioSignal> signals;
    std::vector<size_t> indices;
    std::vector<double> lags;
    for (size_t i = batch; i < end; ++i) {
      const auto& sim_result = sim_results[i];
      if (sim_result.deg_patch_start_time == sim_result.deg_patch_end_time &&
          sim_result.deg_patch_start_time == 0.0) {
        realigned_results[i] = sim_result;
        continue;
      }
      auto ref_patch_audio = Slice(ref_signal, sim_result.ref_patch_start_time,
                                   sim_result.ref_patch_end_time);
      auto deg_patch_audio = Slice(deg_signal, sim_result.deg_patch_start_time,
                                   sim_result.deg_patch_end_time);
      auto aligned =
          Alignment::AlignAndTruncate(ref_patch_audio, deg_patch_audio);
      signals.push_back(std::move(std::get<0>(aligned)));
      signals.push_back(std::move(std::get<1>(aligned)));
      indices.push_back(i);
      lags.push_back(std::get<2>(aligned));
    }
    auto spectra_result = spect_builder->BuildBatch(signals, window);
    if (!spectra_result.ok()) {
      ABSL_RAW_LOG(ERROR, "Error building realigned spectrograms: %s",
                   spectra_result.status().ToString().c_str());
      return spectra_result.status();
    }
    std::vector<Spectrogram> spectra = std::move(spectra_result).value();
    for (size_t item = 0; item < indices.size(); ++item) {
      const size_t i = indices[item];
      const auto& sim_result = sim_results[i];
      const double lag = lags[item];
      const double new_ref_duration = signals[item * 2].GetDuration();
      const double new_deg_duration = signals[item * 2 + 1].GetDuration();
      auto& ref_spectrogram = spectra[item * 2];
      auto& deg_spectrogram = spectra[item * 2 + 1];
      MiscAudio::PrepareSpectrogramsForComparison(ref_spectrogram,
                                                  deg_spectrogram);
      // 4. Recreate an aligned degraded patch from the new spectrogram.
      auto new_ref_patch = ref_spectrogram.Data();

      auto new_deg_patch = deg_spectrogram.Data();
      // 5. Update the similarity result with the new patch.
      auto new_sim_result =
          sim_comparator_->MeasurePatchSimilarity(new_ref_patch, new_deg_patch);
      // Compare to the old result and take the max.
      if (new_sim_result.similarity < sim_result.similarity) {
        realigned_results[i] = sim_result;
      } else {
        if (lag > 0.) {
          new_sim_result.ref_patch_start_time =
              sim_result.ref_patch_start_time + lag;
          new_sim_result.deg_patch_start_time = sim_result.deg_patch_start_time;
        } else {
          new_sim_result.ref_patch_start_time = sim_result.ref_patch_start_time;
          new_sim_result.deg_patch_start_time =
              sim_result.deg_patch_start_time - lag;
        }
        new_sim_result.ref_patch_end_time =
            new_sim_result.ref_patch_start_time + new_ref_duration;
        new_sim_result.deg_patch_end_time =
            new_sim_result.deg_patch_start_time + new_deg_duration;
        realigned_results[i] = new_sim_result;
      }
    }
  }
  return realigned_results;
}
}  // namespace Visqol
