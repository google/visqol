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

#ifndef VISQOL_INCLUDE_FFT_MANAGER_H
#define VISQOL_INCLUDE_FFT_MANAGER_H

#if defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(_M_IX86)
#include <xmmintrin.h>
typedef __m128 SimdVector;
#define SIMD_LENGTH 4
#define SIMD_MULTIPLY(a, b) _mm_mul_ps(a, b)
#define SIMD_LOAD_ONE_FLOAT(p) _mm_set1_ps(p)
#elif defined(__arm__) || defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
typedef float32x4_t SimdVector;
#define SIMD_LENGTH 4
#define SIMD_MULTIPLY(a, b) vmulq_f32(a, b)
#define SIMD_LOAD_ONE_FLOAT(p) vld1q_dup_f32(&(p))
#endif

#include "audio_channel.h"
#include "pffft.h"

namespace Visqol {

/**
 * This class acts as a wrapper for the PFFFT library operations. It is not
 * responsible for managing the lifecycles of the AudioChannels it operates on.
 * It is responsible for memory that it allocates for performing the FFT.
 *
 * This class was adapted from the ResonanceAudio project:
 * https://github.com/resonance-audio/resonance-audio
 */
class FftManager {
 public:
  /**
   * Minimum required FFT size.
   */
  static const size_t kMinFftSize;

  /**
   * If FFT size is less than 2^14 use the stack, else allocate memory. This is
   * the recomendation of the author of the PFFFT library.
   */
  static const size_t kPffftMaxStackSize;

  /**
   * Constructs a FftManager instance. The number of samples that are contained
   * in the input channel that the forward fft will be (or has been) performed
   * on is taken as an input argument. This is used to determine the fft size -
   * PFFFT requires fft size to be a power of 2. This class is not thread safe.
   *
   * @param samples_per_channel The number of samples in the input time domain
   *    channel associated with this manager.
   */
  explicit FftManager(size_t samples_per_channel);

  /**
   * Destroy the manager and any memory associated with it.
   */
  ~FftManager();

  /**
   * For a given input AudioChannel in the time domain perform a forward fft
   * and convert it to the frequency domain, ordered canonically.
   *
   * @param time_channel The input time domain channel.
   * @param freq_channel The output frequency domain channel, ordered
   *    canonically.
   */
  void FreqFromTimeDomain(const AudioChannel& time_channel,
                          AudioChannel* freq_channel);

  /**
   * For a given input AudioChannel in the frequency domain, canonically
   * ordered, perform an inverse fft and convert it to the time domain.
   *
   * @param freq_channel The input frequency domain channel, ordered
   *    canonically.
   * @param time_channel The output time domain channel.
   */
  void TimeFromFreqDomain(const AudioChannel& freq_channel,
                          AudioChannel* time_channel);

  /**
   * Apply a 1/fft_size_ scaling to the time domain output.
   *
   * @param time_channel Time domain channel to be scaled.
   */
  void ApplyReverseFftScaling(AudioChannel* time_channel);

  /**
   * For a given input channel in the frequency domain ordered cannonically,
   * produce a new channel ordered according to the pffft native ordering.
   *
   * @param input The input channel in the frequency domain, ordered
   *    cannonically.
   * @param output The output channel in the frequency domain, ordered
   *    with the pffft ordering.
   */
  void GetPffftFormatFreqBuffer(const AudioChannel& input,
                                AudioChannel* output);

  /**
   * Get the FFT size associated with this manager. The fft size will be a
   * power of 2 value.
   *
   * @return The fft size.
   */
  size_t GetFftSize() const { return fft_size_; }

  /**
   * Get the number of samples in the input time domain channel that this
   * manager was created to work with.
   */
  size_t GetSamplesPerChannel() const { return samples_per_channel_; }

  /**
   * Get a reference to the time channel.
   */
  AudioChannel& GetTimeChannel() { return time_channel_; }

  /**
   * Get a reference to the freq channel.
   */
  AudioChannel& GetFreqChannel() { return freq_channel_; }

 private:
  /**
   * Perform a scalar multiplication on a SIMD alligned input buffer.
   *
   * @param length The length of the input buffer.
   * @param gain The scalar.
   * @param input A float pointer to the first element in the input buffer.
   * @param output A float pointer to the first element in the output buffer.
   */
  void SimdScalarMultiply(size_t length, float gain, const float* input,
                          float* output);
  inline size_t GetNumChunks(size_t length) { return length / SIMD_LENGTH; }
  inline size_t GetLeftoverSamples(size_t length) {
    return length % SIMD_LENGTH;
  }

  /**
   * The FFT size used during this manager's ops. Will be a power of 2.
   */
  const size_t fft_size_;

  /**
   * The number of samples in the input time domain channel that this
   * manager was created to work with.
   */
  const size_t samples_per_channel_;

  /**
   * Inverse scale to be applied to buffers when being transformed from
   * frequency to time domain.
   */
  const float inverse_fft_scale_;

  /**
   *  Stored the PFFFT state for performaing operations.
   */
  PFFFT_Setup* fft_;

  /**
   * Used to store an audio channel in the time domain.
   */
  AudioChannel time_channel_;

  /**
   * Used to store an audio channel in the frequency domain.
   */
  AudioChannel freq_channel_;

  /**
   * Workspace for pffft. This pointer should be set to null for |fft_size_|
   * less than 2^14. In which case the stack is used. This is the
   * recommendation by the author of the pffft library.
   */
  float* pffft_workspace_ = nullptr;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_FFT_MANAGER_H
