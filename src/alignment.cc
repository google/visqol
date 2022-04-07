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

#include "alignment.h"

#include <memory>
#include <utility>

#include "amatrix.h"
#include "audio_signal.h"
#include "envelope.h"
#include "xcorr.h"

namespace Visqol {
std::tuple<AudioSignal, AudioSignal, double> Alignment::AlignAndTruncate(
    const AudioSignal& reference_signal, const AudioSignal& degraded_signal) {
  std::tuple<AudioSignal, double> alignment_result =
      Alignment::GloballyAlign(reference_signal, degraded_signal);
  AudioSignal aligned_degraded_signal = std::get<0>(alignment_result);
  double lag = std::get<1>(alignment_result);
  const AMatrix<double>& reference_matrix = reference_signal.data_matrix;
  // Take the aligned degraded matrix.
  const AMatrix<double>& degraded_matrix = aligned_degraded_signal.data_matrix;
  AMatrix<double> new_reference_matrix = reference_matrix;
  AMatrix<double> new_degraded_matrix = degraded_matrix;

  // Truncate the two aligned signals to match lengths.
  // If the lag is positive or negative, the starts are aligned.
  // (The front of degraded_signal is zero padded or truncated).
  if (reference_matrix.NumRows() > degraded_matrix.NumRows()) {
    new_reference_matrix =
        reference_matrix.GetRows(0, degraded_matrix.NumRows() - 1);
  } else if (reference_matrix.NumRows() < degraded_matrix.NumRows()) {
    // For positive lag, the beginning of ref is now aligned with zeros, so
    // that amount should be truncated.
    new_reference_matrix = reference_matrix.GetRows(
        static_cast<int>(lag * reference_signal.sample_rate),
        reference_matrix.NumRows() - 1);
    // Truncate the zeros off the deg matrix as well.
    new_degraded_matrix = degraded_matrix.GetRows(
        static_cast<int>(lag * degraded_signal.sample_rate),
        reference_matrix.NumRows() - 1);
  }

  AudioSignal new_degraded_signal{new_degraded_matrix,
                                  degraded_signal.sample_rate};
  AudioSignal new_reference_signal{new_reference_matrix,
                                   reference_signal.sample_rate};
  return std::make_tuple(new_reference_signal, new_degraded_signal, lag);
}

std::tuple<AudioSignal, double> Alignment::GloballyAlign(
    const AudioSignal& reference_signal, const AudioSignal& degraded_signal) {
  const AMatrix<double>& reference_matrix = reference_signal.data_matrix;
  const AMatrix<double>& degraded_matrix = degraded_signal.data_matrix;
  AMatrix<double> reference_upper_env =
      Envelope::CalcUpperEnv(reference_matrix);
  AMatrix<double> degraded_upper_env = Envelope::CalcUpperEnv(degraded_matrix);
  int64_t best_lag =
      XCorr::FindLowestLagIndex(reference_upper_env, degraded_upper_env);

  // Limit the lag to half a patch.
  if (best_lag == 0 ||
      std::abs(best_lag) >
          static_cast<double>(reference_matrix.NumRows()) / 2.0) {
    return std::make_tuple(degraded_signal, 0);
  } else {
    // align degraded matrix
    AMatrix<double> new_degraded_matrix;
    // If the same point of the reference comes after the degraded
    // (negative lag), truncate the rows before the refrence.
    // If the reference comes before the degraded, prepend zeros
    // to the degraded.
    if (best_lag < 0) {
      new_degraded_matrix = degraded_matrix.GetRows(
          std::abs(best_lag), degraded_matrix.NumRows() - 1);
    } else {
      new_degraded_matrix = AMatrix<double>::Filled(best_lag, 1, 0.0);
      new_degraded_matrix = new_degraded_matrix.JoinVertically(degraded_matrix);
    }
    AudioSignal new_degraded_signal{std::move(new_degraded_matrix),
                                    degraded_signal.sample_rate};
    return std::make_tuple(
        new_degraded_signal,
        best_lag / static_cast<double>(degraded_signal.sample_rate));
  }
}
}  // namespace Visqol
