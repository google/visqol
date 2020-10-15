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
    const AudioSignal &ref_signal, const AudioSignal &deg_signal) {
  auto alignment_result = Alignment::GloballyAlign(ref_signal, deg_signal);
  AudioSignal aligned_deg_signal = std::get<0>(alignment_result);
  double lag = std::get<1>(alignment_result);
  auto &ref_matrix = ref_signal.data_matrix;
  // Take the aligned degraded matrix.
  auto &deg_matrix = aligned_deg_signal.data_matrix;
  AMatrix<double> new_ref_matrix = ref_matrix;
  AMatrix<double> new_deg_matrix = deg_matrix;

  // Truncate the two aligned signals to match lengths.
  // If the lag is positive or negative, the starts are aligned.
  // (The front of deg_signal is zero padded or truncated).
  if (ref_matrix.NumRows() > deg_matrix.NumRows()) {
    new_ref_matrix = ref_matrix.GetRows(0, deg_matrix.NumRows() - 1);
  } else if (ref_matrix.NumRows() < deg_matrix.NumRows()) {
    // For positive lag, the beginning of ref is now aligned with zeros, so
    // that amount should be truncated.
    new_ref_matrix = ref_matrix.GetRows(int(lag * ref_signal.sample_rate),
                                        ref_matrix.NumRows() - 1);
    // Truncate the zeros off the deg matrix as well.
    new_deg_matrix = deg_matrix.GetRows((int)(lag * deg_signal.sample_rate),
                                        ref_matrix.NumRows() - 1);
  }

  AudioSignal new_deg_signal{new_deg_matrix, deg_signal.sample_rate};
  AudioSignal new_ref_signal{new_ref_matrix, ref_signal.sample_rate};
  return std::make_tuple(new_ref_signal, new_deg_signal, lag);
}

std::tuple<AudioSignal, double> Alignment::GloballyAlign(
    const AudioSignal &ref_signal, const AudioSignal &deg_signal) {
  auto &ref_matrix = ref_signal.data_matrix;
  auto &deg_matrix = deg_signal.data_matrix;
  auto ref_upper_env = Envelope::CalcUpperEnv(ref_matrix);
  auto deg_upper_env = Envelope::CalcUpperEnv(deg_matrix);
  auto best_lag = XCorr::CalcBestLag(ref_upper_env, deg_upper_env);

  // Limit the lag to half a patch.
  if (best_lag == 0 || std::abs(best_lag) > (double) ref_matrix.NumRows() / 2) {
    return std::make_tuple(deg_signal, 0);
  } else {
    // align degraded matrix
    AMatrix<double> new_deg_matrix;
    // If the same point of the reference comes after the degraded
    // (negative lag), truncate the rows before the refrence.
    // If the reference comes before the degraded, prepend zeros
    // to the degraded.
    if (best_lag < 0) {
      new_deg_matrix = deg_matrix.GetRows(std::abs(best_lag),
                                          deg_matrix.NumRows() - 1);
    } else {
      new_deg_matrix = AMatrix<double>::Filled(best_lag, 1, 0.0);
      new_deg_matrix = new_deg_matrix.JoinVertically(deg_matrix);
    }
    AudioSignal new_deg_signal{std::move(new_deg_matrix),
                               deg_signal.sample_rate};
    return std::make_tuple(new_deg_signal, best_lag / (double) deg_signal.sample_rate);
  }
}
}  // namespace Visqol
