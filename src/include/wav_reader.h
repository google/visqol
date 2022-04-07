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

#ifndef VISQOL_INCLUDE_WAV_READER_H_
#define VISQOL_INCLUDE_WAV_READER_H_

#include <cstdint>
#include <istream>

#include "misc_math.h"

namespace Visqol {

/**
 *  Basic RIFF WAVE decoder that supports multichannel 16-bit PCM.
 *
 *  This class was adapted from the ResonanceAudio project:
 *  https://github.com/resonance-audio/resonance-audio
 */
class WavReader {
 public:
  /**
   * Constructor decodes WAV header.
   *
   * @param binary_stream Binary input stream to read from.
   */
  explicit WavReader(std::istream* binary_stream);

  /**
   * True if WAV header was successfully parsed.
   */
  bool IsHeaderValid() const;

  /**
   * Returns the total number of samples defined in the WAV header. Note that
   * the actual number of samples in the file can differ.
   */
  size_t GetNumTotalSamples() const;

  /**
   * Returns number of channels.
   */
  size_t GetNumChannels() const;

  /**
   * Returns sample rate in Hertz.
   */
  int GetSampleRateHz() const;

  /**
   * Returns the duration of the wav file in seconds.
   */
  double GetDuration() const;

  /**
   * Reads samples from WAV file into target buffer.
   *
   * @param num_samples Number of samples to read.
   * @param target_buffer Target buffer to write to.
   * @return Number of decoded samples.
   */
  size_t ReadSamples(size_t num_samples, int16_t* target_buffer);

 private:
  /**
   * Calculate the total number of bytes in the data stream.
   *
   * @return The total number of bytes in the data stream.
   */
  int64_t GetCountOfBytesInStream();

  /**
   * Parses WAV header.
   *
   * @return True on success.
   */
  bool ParseHeader();

  /**
   * Helper method to read binary data from input stream.
   *
   * @param size Number of bytes to read.
   * @param target_ptr Target buffer to write to.
   * @return Number of bytes read.
   */
  size_t ReadBinaryDataFromStream(void* target_ptr, size_t size);

  /**
   * Binary input stream.
   */
  std::istream* binary_stream_;

  /**
   * Flag indicating if the WAV header was parsed successfully.
   */
  bool init_;

  /**
   * Number of audio channels.
   */
  size_t num_channels_;

  /**
   * Sample rate in Hertz.
   */
  int sample_rate_hz_;

  /**
   * Total number of samples.
   */
  size_t num_total_samples_;

  /**
   * Number of remaining samples in WAV file.
   */
  size_t num_remaining_samples_;

  /**
   * Bytes per sample as defined in the WAV header.
   */
  size_t bytes_per_sample_;

  /**
   * Offset into data stream where PCM data begins.
   */
  uint64_t pcm_offset_bytes_;

  /**
   * Total number of bytes in data stream.
   */
  int64_t bytes_in_stream_;
};

}  // namespace Visqol

#endif  // VISQOL_INCLUDE_WAV_READER_H_
