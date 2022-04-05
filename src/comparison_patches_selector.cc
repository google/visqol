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

namespace Visqol {
ComparisonPatchesSelector::ComparisonPatchesSelector(
    std::unique_ptr<PatchSimilarityComparator> sim_comparator)
    : sim_comparator_{std::move(sim_comparator)} {}

void ComparisonPatchesSelector::FindMostOptimalDegPatch(
    const AMatrix<double>& spectrogram_data, const ImagePatch& ref_patch,
    std::vector<ImagePatch>& deg_patches,
    std::vector<std::vector<double>>& cumulative_similarity_dp,
    std::vector<std::vector<int>>& backtrace,
    const std::vector<size_t>& ref_patch_indices, int patch_index,
    const int search_window) const {
  // The similarity threshold below which the two patch matches are not a good
  // match.
  int ref_frame_index = ref_patch_indices[patch_index];
  ImagePatch deg_patch;
  PatchSimilarityResult sim_result;

  // For a given reference frame index, this function compares the given
  // reference patch with all possible degraded patches in the search window and
  // populates the cumulative_similarity_dp vector accordingly. For more details
  // : https://en.wikipedia.org/wiki/Dynamic_time_warping
  // Try Viterbi, if optimization is needed.

  for (int slide_offset = ref_frame_index - search_window;
       slide_offset <= ref_frame_index + search_window; slide_offset++) {
    if (slide_offset < 0) {
      // The degraded patch index cannot be less than 0.
      slide_offset = -1;
      continue;
    }
    if (slide_offset >= spectrogram_data.NumCols()) {
      // The start of the degraded is past the end of the spectrogram, so
      // nothing left to compare.
      break;
    }
    deg_patch = deg_patches[slide_offset];
    sim_result = sim_comparator_->MeasurePatchSimilarity(ref_patch, deg_patch);

    int past_slide_offset = -1;
    double highest_sim = std::numeric_limits<double>::lowest();
    // There's no need to backtrace for the first patch index.
    if (patch_index > 0) {
      // The lower_limit parameter tells us how far we should go
      // back to look for a possible match for the previous patch index
      // (patch_index - 1). The current value of lower_limit is used because the
      // search space for the previous patch index  is
      // (ref_patch_indices[patch_index - 1] - search_window,
      // ref_patch_indices[patch_index - 1] + search_window).
      int lower_limit = ref_patch_indices[patch_index - 1] - search_window;
      lower_limit = std::max(lower_limit, 0);
      // The back_offset parameter determines all the offsets that should be
      // considered while calculating the highest cumulative similarity score
      // achieved till patch_index - 1. Since two reference patches should
      // not map to the exact same degraded patch, the initial value of
      // back_offset is set to slide_offset - 1.
      int back_offset = slide_offset - 1;
      for (; back_offset >= lower_limit; back_offset--) {
        // The current for loop is used to find out the highest cumulative score
        // achieved till the previous ref_patch_index.
        if (cumulative_similarity_dp[patch_index - 1][back_offset] >
            highest_sim) {
          highest_sim = cumulative_similarity_dp[patch_index - 1][back_offset];
          past_slide_offset = back_offset;
        }
      }
      sim_result.similarity += highest_sim;
      // If the current reference patch experienced a packet loss, then the
      // cumulative similarity score till the previous patch might be more and
      // in that case no matching patch for the current reference patch is found
      // in the degraded window.
      if (cumulative_similarity_dp[patch_index - 1][slide_offset] >
          sim_result.similarity) {
        sim_result.similarity =
            cumulative_similarity_dp[patch_index - 1][slide_offset];
        past_slide_offset = slide_offset;
      }
    }
    cumulative_similarity_dp[patch_index][slide_offset] = sim_result.similarity;
    backtrace[patch_index][slide_offset] = past_slide_offset;
  }
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
  std::vector<ImagePatch> deg_patches(spectrogram_data.NumCols());
  for (size_t slide_offset = 0; slide_offset < spectrogram_data.NumCols();
       slide_offset++) {
    deg_patches[slide_offset] =
        BuildDegradedPatch(spectrogram_data, slide_offset,
                           slide_offset + ref_patches[0].NumCols() - 1,
                           ref_patches[0].NumRows(), ref_patches[0].NumCols());
  }
  // Attempt to get a good alignment with backtracking.
  for (size_t patch_index = 0; patch_index < num_patches; patch_index++) {
    // Find the best alignment to the ref patch within a distance of
    // search_window on each side of the hard-aligned deg signal.
    FindMostOptimalDegPatch(spectrogram_data, ref_patches[patch_index],
                            deg_patches, cumulative_similarity_dp, backtrace,
                            ref_patch_indices, patch_index, search_window);
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

  // The patches are already matched.  Iterate over each pair.
  for (size_t i = 0; i < sim_results.size(); ++i) {
    auto sim_result = sim_results[i];
    if (sim_result.deg_patch_start_time == sim_result.deg_patch_end_time &&
        sim_result.deg_patch_start_time == 0.0) {
      realigned_results[i] = sim_result;
      continue;
    }

    // 1. The sim results keep track of the start and end points of each matched
    // pair.  Extract the audio for this segment.
    auto ref_patch_audio = Slice(ref_signal, sim_result.ref_patch_start_time,
                                 sim_result.ref_patch_end_time);
    auto deg_patch_audio = Slice(deg_signal, sim_result.deg_patch_start_time,
                                 sim_result.deg_patch_end_time);
    // 2. For any pair, we want to shift the degraded signal to be maximally
    // aligned.
    auto aligned_result =
        Alignment::AlignAndTruncate(ref_patch_audio, deg_patch_audio);
    AudioSignal ref_audio_aligned = std::get<0>(aligned_result);
    AudioSignal deg_audio_aligned = std::get<1>(aligned_result);
    double lag = std::get<2>(aligned_result);

    double new_ref_duration = ref_audio_aligned.GetDuration();
    double new_deg_duration = deg_audio_aligned.GetDuration();
    // 3. Compute a new spectrogram for the degraded audio.
    const auto ref_spectro_result =
        spect_builder->Build(ref_audio_aligned, window);
    if (!ref_spectro_result.ok()) {
      ABSL_RAW_LOG(ERROR, "Error building ref spectrogram: %s",
                   ref_spectro_result.status().ToString().c_str());
      return ref_spectro_result.status();
    }
    Spectrogram ref_spectrogram = ref_spectro_result.value();

    const auto deg_spectro_result =
        spect_builder->Build(deg_audio_aligned, window);
    if (!deg_spectro_result.ok()) {
      ABSL_RAW_LOG(ERROR, "Error building degraded spectrogram: %s",
                   deg_spectro_result.status().ToString().c_str());
      return deg_spectro_result.status();
    }
    Spectrogram deg_spectrogram = deg_spectro_result.value();

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
  return realigned_results;
}
}  // namespace Visqol
