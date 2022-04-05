// Copyright 2021 Google LLC
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

#include "analysis_window.h"

#include <vector>

#include "absl/base/macros.h"

namespace Visqol {
AnalysisWindow::AnalysisWindow(const size_t sample_rate,
                               const double win_overlap, double window_duration)
    : window_duration(window_duration), overlap(win_overlap) {
  size = static_cast<size_t>(round(sample_rate * window_duration));
}

std::valarray<double> AnalysisWindow::ApplyHannWindow(
    const std::valarray<double>& signal) const {
  // It is assumed `signal` is the same length so we don't zero pad the end.
  ABSL_ASSERT(signal.size() == size);

  // Precompute the hann window function.
  std::vector<double> window(size);
  for (int i = 0; i < size; ++i) {
    window[i] = 0.5 - (0.5 * cos(2.0 * M_PI * i / (size - 1)));
  }

  std::valarray<double> windowed_signal(signal.size());
  for (int i = 0; i < size; ++i) {
    windowed_signal[i] = window[i] * signal[i];
  }
  return windowed_signal;
}

}  // namespace Visqol
