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

#include "spectrogram.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "amatrix.h"
#include "misc_audio.h"

namespace Visqol {
Spectrogram::Spectrogram(AMatrix<double>&& data) : data_{std::move(data)} {}

void Spectrogram::ConvertToDb() {
  std::transform(data_.begin(), data_.end(), data_.begin(),
                 Spectrogram::ConvertSampleToDb);
}

double Spectrogram::Minimum() const {
  return *std::min_element(data_.cbegin(), data_.cend());
}

void Spectrogram::SubtractFloor(double floor) {
  std::transform(data_.begin(), data_.end(), data_.begin(),
                 [&](double d) { return d - floor; });
}

void Spectrogram::RaiseFloor(double new_floor) {
  std::transform(data_.begin(), data_.end(), data_.begin(),
                 [&](double d) { return std::max(new_floor, d); });
}

void Spectrogram::RaiseFloorPerFrame(double noise_threshold,
                                     Spectrogram& other) {
  // Go over each frame, and clip the quiet regions below noise_threshold from
  // the peak of the highest in ref/deg for that frame.
  // Signals with activity have peaks that are typically in the -10dB range.
  // 'Silent' ambient noise frames are typically in the -1000dB to -25dB range.
  // This means most of the action is at the -25 to -10dB range.
  size_t min_cols = std::min(data_.NumCols(), other.data_.NumCols());
  for (size_t i = 0; i < min_cols; i++) {
    auto our_frame = data_.GetColumn(i);
    auto other_frame = other.data_.GetColumn(i);
    // Find the max value per frame.
    double our_max = *std::max_element(our_frame.cbegin(), our_frame.cend());
    double other_max =
        *std::max_element(other_frame.cbegin(), other_frame.cend());
    double any_max = std::max(our_max, other_max);
    double floor_db = any_max - noise_threshold;

    // Raise the floor by some amount under the max.
    std::transform(our_frame.begin(), our_frame.end(), our_frame.begin(),
                   [&](double d) { return std::max(floor_db, d); });
    std::transform(other_frame.begin(), other_frame.end(), other_frame.begin(),
                   [&](double d) { return std::max(floor_db, d); });
    data_.SetColumn(i, std::move(our_frame));
    other.data_.SetColumn(i, std::move(other_frame));
  }
}

double Spectrogram::ConvertSampleToDb(const double sample) {
  // Get the absolute value of the sample. If the sample is zero, use epsilon.
  const auto abs_sample = std::abs(sample) == 0
                              ? std::numeric_limits<double>::epsilon()
                              : std::abs(sample);
  // Convert the sample to decibels and return.
  return 10 * std::log10(abs_sample);
}

void Spectrogram::SetCenterFreqBands(
    const std::vector<double>& center_freq_bands) {
  center_freq_bands_ = center_freq_bands;
}

const std::vector<double> Spectrogram::GetCenterFreqBands() const {
  return center_freq_bands_;
}
}  // namespace Visqol
