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

#include "rms_vad.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace Visqol {

const std::size_t RmsVad::kSilentChunkCount = 3;
const double RmsVad::kRmsThreshold = 5000.0;
const double RmsVad::kVoiceActivityPresent = 1.0;
const double RmsVad::kVoiceActivityAbsent = 0.0;

RmsVad::RmsVad() {
  // Set the first kSilentChunkCount results to voice activity detected.
  // Helps to avoid false negatives.
  for (std::size_t i = 0; i < kSilentChunkCount - 1; i++) {
    vad_results_.push_back(kVoiceActivityPresent);
  }
}

double RmsVad::ProcessChunk(const std::vector<int16_t>& chunk) {
  const double rms = CalcRootMeanSquare(chunk);
  if (rms < kRmsThreshold) {
    each_chunk_result_.push_back(kVoiceActivityAbsent);
  } else {
    each_chunk_result_.push_back(kVoiceActivityPresent);
  }
  return rms;
}

std::vector<double> RmsVad::GetVadResults() {
  for (std::size_t i = kSilentChunkCount - 1; i < each_chunk_result_.size();
       i++) {
    // If this chunk is below the RMS threshold and the previous
    // kSilentChunkCount chunks are also below threshold, then mark this chunk
    // as lacking voice activity.
    if (!each_chunk_result_[i] && CheckPreviousChunksForSilence(i)) {
      vad_results_.push_back(kVoiceActivityAbsent);
    } else {
      vad_results_.push_back(kVoiceActivityPresent);
    }
  }
  return vad_results_;
}

double RmsVad::CalcRootMeanSquare(const std::vector<int16_t>& chunk) {
  double square = 0.0;
  double mean = 0.0;
  for (std::size_t i = 0; i < chunk.size(); i++) {
    square += std::pow(chunk[i], 2);
  }
  mean = square / static_cast<double>(chunk.size());
  return std::sqrt(mean);
}

bool RmsVad::CheckPreviousChunksForSilence(const std::size_t idx) {
  bool previous_chunks_silent = true;
  for (std::size_t j = 1; j < kSilentChunkCount; j++) {
    if (each_chunk_result_[idx - j] == kVoiceActivityPresent) {
      previous_chunks_silent = false;
      break;
    }
  }
  return previous_chunks_silent;
}

}  // namespace Visqol
