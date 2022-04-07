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

#ifndef VISQOL_INCLUDE_RMS_VAD_H
#define VISQOL_INCLUDE_RMS_VAD_H

#include <cstdint>
#include <vector>

namespace Visqol {

/**
 * This class provides a simple Root Mean Square (RMS) implementation for Voice
 * Activity Detection (VAD).
 *
 * The signal to be tested should be broken into chunks and each chunk passed
 * sequentially to the ProcessChunk function. Once all chunks have been
 * processed, GetVadResults can be called to get the results of which chunks
 * have voice activity.
 */
class RmsVad {
 public:
  /**
   * Constant value stored in results vector when voice activity is present.
   */
  static const double kVoiceActivityPresent;

  /**
   * Constant value stored in results vector when voice activity is absent.
   */
  static const double kVoiceActivityAbsent;

  /**
   * The number of sequential chunks with an RMS below the threshold before we
   * declare there to be no voice activity at this point in the signal.
   */
  static const std::size_t kSilentChunkCount;

  /**
   * The threshold RMS value for a chunk to be declared as having voice
   * activity present.
   */
  static const double kRmsThreshold;

  /**
   * Public no-args constructor. Responsbile for setting the first
   * kSilentChunkCount vad results for the first chunk to positive i.e. that
   * voice activity was detected. This is done to avoid false negatives where
   * we state there is no voice activity when there is in fact some. It will
   * also constitute a very small percentage of total samples provided.
   */
  RmsVad();

  /**
   * For a given input signal chunk, calculate the RMS value for the chunk and
   * compare to the threshold RMS value.
   *
   * Once all chunks have been process, GetVadResults can be called to get the
   * results.
   *
   * @param chunk The chunk to process.
   *
   * @return The RMS value for this chunk.
   */
  double ProcessChunk(const std::vector<int16_t>& chunk);

  /**
   * Get the results for VAD for each chunk.
   *
   * @return The VAD results.
   */
  std::vector<double> GetVadResults();

 private:
  /**
   * The results of the RMS threshold comparison for each chunk.
   */
  std::vector<double> each_chunk_result_;

  /**
   * The results for the VAD for each chunk.
   */
  std::vector<double> vad_results_;

  /**
   * Helper function for calculating the RMS of a chunk.
   *
   * @param chunk The chunk to calculate the RMS for.
   *
   * @return The RMS for the chunk.
   */
  double CalcRootMeanSquare(const std::vector<int16_t>& chunk);

  /**
   * If we detect a chunk with a RMS below the threshold, we only mark it as
   * lacking voice activity if the previous kSilentChunkCounts are also below
   * the RMS threshold.
   *
   * @param idx The index of the chunk that we want to detect previous chunk
   *    values for.
   *
   * @return True if the previous chunks are also below the RMS threshold. Else,
   *    false.
   */
  bool CheckPreviousChunksForSilence(const std::size_t idx);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_RMS_VAD_H
