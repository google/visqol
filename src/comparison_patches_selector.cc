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
#include "util/task/status.h"
#include "util/task/statusor.h"

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

PatchSimilarityResult ComparisonPatchesSelector::FindMostSimilarDegPatch(
    const AMatrix<double>& spectrogram_data, const ImagePatch& ref_patch,
    int ref_frame_index, const double frame_duration) const {
  ImagePatch deg_patch;
  PatchSimilarityResult sim_result;
  PatchSimilarityResult best_sim_result;
  int best_slide_offset = 0;
  const int num_frames_per_patch = ref_patch.NumCols();
  const double patch_duration = frame_duration * num_frames_per_patch;
  double highest_sim = std::numeric_limits<double>::lowest();

  // For each possible index in the given range, build a degraded patch and
  // check it for similarity to the given reference patch.

  // The ref_frame_index means ref patch minus half a patch, and the end index
  // if ref patch + half.  This means we may need to add silence.
  for (int slide_offset = ref_frame_index - (num_frames_per_patch / 2);
       slide_offset <= ref_frame_index + (num_frames_per_patch / 2);
       slide_offset++) {
    if (slide_offset > (int)spectrogram_data.NumCols() - 1) {
      // The start of the degraded is past the end of the spectrogram, so
      // nothing left to compare.
      break;
    }
    deg_patch = BuildDegradedPatch(spectrogram_data, slide_offset,
        slide_offset + ref_patch.NumCols() - 1, ref_patch.NumRows(),
        ref_patch.NumCols());
    sim_result = sim_comparator_->MeasurePatchSimilarity(ref_patch, deg_patch);

    if (sim_result.similarity > highest_sim) {
      highest_sim = sim_result.similarity;
      best_sim_result = sim_result;
      best_slide_offset = slide_offset;
    }
  }

  PatchSimilarityResult res;
  res = std::move(best_sim_result);
  // Set the degraded patch start and end time.
  res.deg_patch_start_time = best_slide_offset * frame_duration;
  res.deg_patch_end_time = res.deg_patch_start_time + patch_duration;
  return res;
}

util::StatusOr<std::vector<PatchSimilarityResult>>
ComparisonPatchesSelector::FindMostSimilarDegPatches(
    const std::vector<ImagePatch>& ref_patches,
    const std::vector<size_t>& ref_patch_indices,
    const AMatrix<double>& spectrogram_data,
    const double frame_duration) const {
  const size_t num_frames_per_patch = ref_patches[0].NumCols();
  const size_t num_frames_in_deg_spectro = spectrogram_data.NumCols();
  const double patch_duration = frame_duration * num_frames_per_patch;
  // Allow going up to the end.
  const size_t num_patches = CalcMaxNumPatches(ref_patch_indices,
      num_frames_in_deg_spectro, num_frames_per_patch);

  if (!num_patches) {
    return util::Status(
        google::protobuf::util::error::Code::CANCELLED ,
        "Degraded file was too short, different, or misaligned to score any "
        "of the reference patches.");
  } else if (num_patches < ref_patch_indices.size()) {
    ABSL_RAW_LOG(WARNING, "Warning: Dropping %zu (of %zu) reference patches "
        "due to the degraded file being misaligned or too short. If too many "
        "patches are dropped, the score will be less meaningful.",
        ref_patch_indices.size() - num_patches, ref_patch_indices.size());
  }

  std::vector<PatchSimilarityResult> bestDegPatches(num_patches);

  // Attempt to get a good alignment without backtracking.
  for (size_t patch_index = 0; patch_index < num_patches; patch_index++) {
    // Find the best alignment to the ref patch within 1 patch length of the
    // hard-aligned deg signal.
    bestDegPatches[patch_index] =
        FindMostSimilarDegPatch(spectrogram_data, ref_patches[patch_index],
                                ref_patch_indices[patch_index], frame_duration);

    // Set the reference patch start and end time.
    bestDegPatches[patch_index].ref_patch_start_time =
        ref_patch_indices[patch_index] * frame_duration;

    bestDegPatches[patch_index].ref_patch_end_time =
        bestDegPatches[patch_index].ref_patch_start_time + patch_duration;
  }
  return bestDegPatches;
}

size_t ComparisonPatchesSelector::CalcMaxNumPatches(
    const std::vector<size_t>& ref_patch_indices, size_t num_deg_frames,
    size_t num_frames) const {
  size_t num_patches = ref_patch_indices.size();

  if (num_patches) {
    // The last patch can start up to half a patch away.
    while (ref_patch_indices[num_patches - 1] - floor(num_frames / 2) >
           num_deg_frames) {
      num_patches--;
    }
  }

  return num_patches;
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
  int last_real_frame =  std::min(window_end, spectrogram_data.NumCols() - 1);
  for (size_t rowIndex = 0; rowIndex < spectrogram_data.NumRows(); rowIndex++) {
    row = spectrogram_data.RowSubset(rowIndex, first_real_frame,
                                     last_real_frame);

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

AudioSignal ComparisonPatchesSelector::Slice(
    const AudioSignal &in_signal, double start_time, double end_time)
{
  int start_index = std::max(0, (int)(start_time * in_signal.sample_rate));
  // The end_index is inclusive for GetRows().
  int end_index = std::min((int)(in_signal.data_matrix.NumRows() - 1),
                           (int)(end_time * in_signal.sample_rate));

  // Note that the underlying armadillo rows() uses an unconventional inclusive
  // end index.
  auto sliced_matrix = in_signal.data_matrix.GetRows(
       start_index, end_index - 1);
  if (start_time < 0) {
    auto presilence_matrix = AMatrix<double>::Filled(-1 * start_time *
                                                     in_signal.sample_rate,
                                                     1, 0.0);
    sliced_matrix = presilence_matrix.JoinVertically(sliced_matrix);
  }
  AudioSignal sliced_signal{sliced_matrix, in_signal.sample_rate};
  return sliced_signal;
}

util::StatusOr<std::vector<PatchSimilarityResult>>
ComparisonPatchesSelector::FinelyAlignAndRecreatePatches(
    const std::vector<PatchSimilarityResult>& sim_results,
    const AudioSignal &ref_signal,
    const AudioSignal &deg_signal,
    SpectrogramBuilder *spect_builder,
    const AnalysisWindow &window) const {
  std::vector<PatchSimilarityResult> realigned_results(sim_results.size());

  // The patches are already matched.  Iterate over each pair.
  for (size_t i = 0; i < sim_results.size(); ++i) {
    auto sim_result = sim_results[i];
    // 1. The sim results keep track of the start and end points of each matched
    // pair.  Extract the audio for this segment.
    auto ref_patch_audio = Slice(ref_signal, sim_result.ref_patch_start_time,
                                 sim_result.ref_patch_end_time);
    auto deg_patch_audio = Slice(deg_signal, sim_result.deg_patch_start_time,
                                 sim_result.deg_patch_end_time);
    // 2. For any pair, we want to shift the degraded signal to be maximally
    // aligned.
    auto aligned_result = Alignment::AlignAndTruncate(ref_patch_audio,
                                                      deg_patch_audio);
    AudioSignal ref_audio_aligned = std::get<0>(aligned_result);
    AudioSignal deg_audio_aligned = std::get<1>(aligned_result);
    double lag = std::get<2>(aligned_result);

    double new_ref_duration = ref_audio_aligned.GetDuration();
    double new_deg_duration = deg_audio_aligned.GetDuration();
    // 3. Compute a new spectrogram for the degraded audio.
    const auto ref_spectro_result = spect_builder->Build(ref_audio_aligned,
                                                         window);
    if (!ref_spectro_result.ok()) {
      ABSL_RAW_LOG(ERROR, "Error building ref spectrogram: %s",
                   ref_spectro_result.status().ToString().c_str());
      return ref_spectro_result.status();
    }
    Spectrogram ref_spectrogram = ref_spectro_result.ValueOrDie();

    const auto deg_spectro_result = spect_builder->Build(deg_audio_aligned,
                                                         window);
    if (!deg_spectro_result.ok()) {
      ABSL_RAW_LOG(ERROR, "Error building degraded spectrogram: %s",
                   deg_spectro_result.status().ToString().c_str());
      return deg_spectro_result.status();
    }
    Spectrogram deg_spectrogram = deg_spectro_result.ValueOrDie();

    MiscAudio::PrepareSpectrogramsForComparison(ref_spectrogram,
                                                deg_spectrogram);
    // 4. Recreate an aligned degraded patch from the new spectrogram.
    auto new_ref_patch = ref_spectrogram.Data();

    auto new_deg_patch = deg_spectrogram.Data();
    // 5. Update the similarity result with the new patch.
    auto new_sim_result = sim_comparator_->MeasurePatchSimilarity(
        new_ref_patch, new_deg_patch);
    // Compare to the old result and take the max.
    if (new_sim_result.similarity < sim_result.similarity) {
      realigned_results[i] = sim_result;
    } else {
      if (lag > 0.) {
        new_sim_result.ref_patch_start_time = sim_result.ref_patch_start_time
                                              + lag;
        new_sim_result.deg_patch_start_time = sim_result.deg_patch_start_time;
      } else {
        new_sim_result.ref_patch_start_time = sim_result.ref_patch_start_time;
        new_sim_result.deg_patch_start_time = sim_result.deg_patch_start_time
                                              - lag;
      }
      new_sim_result.ref_patch_end_time = new_sim_result.ref_patch_start_time
                                          + new_ref_duration;
      new_sim_result.deg_patch_end_time = new_sim_result.deg_patch_start_time
                                          + new_deg_duration;
      realigned_results[i] = new_sim_result;
    }
  }
  return realigned_results;
}
}  // namespace Visqol
