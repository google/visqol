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

#include "fft_manager.h"

#include <assert.h>

#include <algorithm>

#include "misc_math.h"

// Prevent Visual Studio from complaining about std::copy_n.
#if defined(_WIN32)
#define _SCL_SECURE_NO_WARNINGS
#endif

namespace Visqol {

const size_t FftManager::kMinFftSize = 32;
const size_t FftManager::kPffftMaxStackSize = 16384;

FftManager::FftManager(size_t samples_per_channel)
    : fft_size_(
          std::max(MiscMath::NextPowTwo(samples_per_channel), kMinFftSize)),
      samples_per_channel_(samples_per_channel),
      inverse_fft_scale_(1.0f / static_cast<float>(fft_size_)) {
  if (fft_size_ > kPffftMaxStackSize) {
    // Allocate memory for work space factors etc, Size reccomended by pffft.
    const size_t num_bytes = 2 * fft_size_ * sizeof(float);
    pffft_workspace_ =
        reinterpret_cast<float*>(pffft_aligned_malloc(num_bytes));
  }

  time_channel_.Init(samples_per_channel_);
  freq_channel_.Init(fft_size_);

  fft_ = pffft_new_setup(static_cast<int>(fft_size_), PFFFT_REAL);
}

FftManager::~FftManager() {
  pffft_destroy_setup(fft_);
  if (pffft_workspace_ != nullptr) {
    pffft_aligned_free(pffft_workspace_);
  }
}

void FftManager::FreqFromTimeDomain(const AudioChannel& time_channel,
                                    AudioChannel* freq_channel) {
  assert(freq_channel->size() == fft_size_);
  assert(time_channel.size() <= fft_size_);

  // Perform forward FFT transform.
  if (time_channel.size() == fft_size_) {
    pffft_transform_ordered(fft_, time_channel.begin(), freq_channel->begin(),
                            pffft_workspace_, PFFFT_FORWARD);
  } else {
    AudioChannel temp_zeropad_buffer_;
    temp_zeropad_buffer_.Init(fft_size_);
    temp_zeropad_buffer_.Clear();
    std::copy_n(time_channel.begin(), samples_per_channel_,
                temp_zeropad_buffer_.begin());
    pffft_transform_ordered(fft_, temp_zeropad_buffer_.begin(),
                            freq_channel->begin(), pffft_workspace_,
                            PFFFT_FORWARD);
  }
}

void FftManager::TimeFromFreqDomain(const AudioChannel& freq_channel,
                                    AudioChannel* time_channel) {
  assert(freq_channel.size() == fft_size_);

  // Perform reverse FFT transform.
  const size_t time_channel_size = time_channel->size();
  if (time_channel_size == fft_size_) {
    pffft_transform(fft_, freq_channel.begin(), time_channel->begin(),
                    pffft_workspace_, PFFFT_BACKWARD);
  } else {
    AudioChannel temp_freq_buffer_;
    temp_freq_buffer_.Init(fft_size_);
    temp_freq_buffer_.Clear();
    auto& temp_channel = temp_freq_buffer_;
    pffft_transform(fft_, freq_channel.begin(), temp_channel.begin(),
                    pffft_workspace_, PFFFT_BACKWARD);
    std::copy_n(temp_channel.begin(), samples_per_channel_,
                time_channel->begin());
  }
}

void FftManager::ApplyReverseFftScaling(AudioChannel* time_channel) {
  assert(time_channel->size() == samples_per_channel_ ||
         time_channel->size() == fft_size_);

  SimdScalarMultiply(time_channel->size(), inverse_fft_scale_,
                     time_channel->begin(), time_channel->begin());
}

void FftManager::GetPffftFormatFreqBuffer(const AudioChannel& input,
                                          AudioChannel* output) {
  assert(input.size() == fft_size_);
  assert(output->size() == fft_size_);

  pffft_zreorder(fft_, input.begin(), output->begin(), PFFFT_BACKWARD);
}

void FftManager::SimdScalarMultiply(size_t length, float gain,
                                    const float* input, float* output) {
  const SimdVector* input_vector = reinterpret_cast<const SimdVector*>(input);
  SimdVector* output_vector = reinterpret_cast<SimdVector*>(output);
  const SimdVector gain_vector = SIMD_LOAD_ONE_FLOAT(gain);

  for (size_t i = 0; i < GetNumChunks(length); ++i) {
    output_vector[i] = SIMD_MULTIPLY(gain_vector, input_vector[i]);
  }

  // Apply gain to samples at the end that were missed by the SIMD chunking.
  const size_t leftover_samples = GetLeftoverSamples(length);
  for (size_t i = length - leftover_samples; i < length; ++i) {
    output[i] = input[i] * gain;
  }
}

}  // namespace Visqol
