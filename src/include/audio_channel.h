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

#ifndef VISQOL_INCLUDE_AUDIO_CHANNEL_H
#define VISQOL_INCLUDE_AUDIO_CHANNEL_H

#include <cstring>
#include <vector>

#include "pffft.h"

namespace Visqol {

/**
 * This class provides a memory aligned buffer that is used to represent a
 * single audio channel. It is used by the fast Fourier Transform
 * functionality. This class is the owner of the memory it allocates and is
 * responsible for the lifetime management of this memory.
 */
class AudioChannel {
 public:
  /**
   * For a given input size, initialise an alligned memory buffer of floats.
   * The allignment is "simd-compatible" alignment i.e. 16 bytes on x86 and
   * powerpc CPUs. Samples are stored as floats.
   *
   * @param size The number of samples required in this channel.
   */
  void Init(const size_t size) {
    size_ = size;
    size_t n_bytes = sizeof(float) * size_;
    alligned_buffer_ = reinterpret_cast<float*>(pffft_aligned_malloc(n_bytes));
    begin_itr_ = alligned_buffer_;
  }

  /**
   * This destructor is responsible for freeing memory allocated during the
   * initialisation of the class.
   */
  ~AudioChannel() {
    if (alligned_buffer_ != nullptr) {
      pffft_aligned_free(alligned_buffer_);
    }
  }

  /**
   * Array subscript operator. Return a reference to the sample at the given
   * index.
   *
   * @param index The index of the sample to return.
   *
   * @return A reference to the sample.
   */
  float& operator[](size_t index) { return *(begin() + index); }

  /**
   * Const Array subscript operator. Return a const reference to the sample at
   * the given index.
   *
   * @param index The index of the sample to return.
   *
   * @return A const reference to the sample.
   */
  const float& operator[](size_t index) const { return *(begin() + index); }

  /**
   * Returns the number of samples in this channel.
   *
   * @return The number of samples.
   */
  size_t size() const { return size_; }

  /**
   * Returns a float pointer to the first sample in the channel.
   *
   * @return A float pointer to the first sample.
   */
  float* begin() { return begin_itr_; }

  /**
   * Returns a float pointer to the last sample in the channel.
   *
   * @return A float pointer to the last sample.
   */
  float* end() { return begin_itr_ + size_; }

  /**
   * Returns a const float pointer to the first sample in the channel.
   *
   * @return A const float pointer to the first sample.
   */
  const float* begin() const { return begin_itr_; }

  /**
   * Returns a const float pointer to the last sample in the channel.
   *
   * @return A const float pointer to the last sample.
   */
  const float* end() const { return begin_itr_ + size_; }

  /**
   * Set all samples in the AudioChannel to 0.
   */
  void Clear() { memset(begin(), 0, sizeof(float) * size_); }

 private:
  /**
   *  The aligned memory buffer.
   */
  float* alligned_buffer_ = nullptr;

  /**
   *  Pointer to first element in the buffer.
   */
  float* begin_itr_;

  /**
   * The number of samples in this AudioChannel.
   */
  size_t size_;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_AUDIO_CHANNEL_H
