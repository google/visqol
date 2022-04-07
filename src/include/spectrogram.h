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

#ifndef VISQOL_INCLUDE_SPECTROGRAM_H
#define VISQOL_INCLUDE_SPECTROGRAM_H

#include <vector>

#include "amatrix.h"

namespace Visqol {

/**
 * This class represents a spectrogram representation of a signal.
 */
class Spectrogram {
 public:
  /**
   * Public no-arg constructor needed for StatusOr instantiation.
   */
  Spectrogram() {}

  /**
   * Constructs a spectrogram object using the input matrix.
   */
  explicit Spectrogram(AMatrix<double>&& data);

  /**
   * Return the smallest value in the spectrogram's matrix.
   *
   * @return The smallest value in the spectrogram's matrix.
   */
  double Minimum() const;

  /**
   * Raises the floor at each frame to noise_threshold below the maximum value
   * in this spectrogram and another spectrogram.  Both spectrograms are
   * affected this way (and share the same floor per frame).
   *
   * @param noise_threshold The new relative floor.
   * @param other A spectrogram to compare against.
   */
  void RaiseFloorPerFrame(double noise_threshold, Spectrogram& other);

  /**
   * For each element in the spectrogram's matrix, subtract the provided floor
   * value and update that element with the new post-subtraction value.
   *
   * @param floor The value to be subtracted from each element in the
   *    spectrogram's matrix.
   */
  void SubtractFloor(double floor);

  /**
   * Converts and updates the data in the spectrogram's matrix to decibels.
   */
  void ConvertToDb();

  /**
   * Used for getting a const reference to the spectrogram's matrix.
   *
   * @return A const reference to the spectrogram's matrix.
   */
  const AMatrix<double>& Data() const { return data_; }

  /**
   * Set the center frequency bands that were used to construct this
   * spectrogram. This setter is therefore called after the spectrogram has
   * already been created to record the center frequencies. Calling this setter
   * before spectrogram creation will not result in the center frequencies that
   * the caller sets being used for spectrogram creation.
   *
   * @param center_freq_bands The center frequency bands that were used to
   *    create this spectrogram. These center frequency bands must be ordered
   *    running from the lowest frequency band to the highest.
   */
  void SetCenterFreqBands(const std::vector<double>& center_freq_bands);

  /**
   * Get the center frequency bands that were used to create this spectrogram.
   *
   * @return The center frequency bands that were used to create this
   *    spectrogram.
   */
  const std::vector<double> GetCenterFreqBands() const;

  /**
   * Raises the floor of the spectrogram by taking the max of any item
   * and `new_floor`
   *
   * @param new_floor the value of the new minimum value for the spectrogram.
   */
  void RaiseFloor(double new_floor);

 private:
  /**
   * The matrix used for storing spectrogram data.
   */
  AMatrix<double> data_;

  /**
   * Convert a single sample to decibels.
   *
   * @param sample The sample to convert to decibel.
   *
   * @return The decibel value for the given sample.
   */
  static double ConvertSampleToDb(const double sample);

  /**
   * The center frequency of each frequency band (represented by the rows) in
   * this Spectrogram. The center frequencies are stored from lowest to highest.
   */
  std::vector<double> center_freq_bands_;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_SPECTROGRAM_H
