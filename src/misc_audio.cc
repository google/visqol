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

#include "misc_audio.h"

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "absl/base/internal/raw_logging.h"
#include "wav_reader.h"

namespace Visqol {

const size_t MiscAudio::kNumChanMono = 1;
const double MiscAudio::kZeroSample = 0.0;
const double MiscAudio::kSplReferencePoint = 0.00002;
const double kNoiseFloorRelativeToPeakDb = 45.;
const double kNoiseFloorAbsoluteDb = -45.;

AudioSignal MiscAudio::ScaleToMatchSoundPressureLevel(
    const AudioSignal& reference, const AudioSignal& degraded) {
  const double ref_spl = MiscAudio::CalcSoundPressureLevel(reference);
  const double deg_spl = MiscAudio::CalcSoundPressureLevel(degraded);
  const double scale_factor = std::pow(10, (ref_spl - deg_spl) / 20);
  const auto scaled_mat = degraded.data_matrix * scale_factor;
  AudioSignal scaled_sig{std::move(scaled_mat), degraded.sample_rate};
  return scaled_sig;
}

double MiscAudio::CalcSoundPressureLevel(const AudioSignal& signal) {
  const AMatrix<double>& data_matrix = signal.data_matrix;
  double sum = 0;
  std::for_each(data_matrix.cbegin(), data_matrix.cend(),
                [&](const double& datum) { sum += std::pow(datum, 2); });
  const double sound_pressure = std::sqrt(sum / data_matrix.NumElements());
  return 20 * std::log10(sound_pressure / kSplReferencePoint);
}

// Combines the data from all channels into a single channel.
AMatrix<double> MiscAudio::ToMono(const AMatrix<double>& signal) {
  // If already Mono, nothing to do.
  if (signal.NumCols() > kNumChanMono) {
    auto mono_mat =
        AMatrix<double>::Filled(signal.NumRows(), kNumChanMono, kZeroSample);
    for (size_t chan_i = 0; chan_i < signal.NumCols(); chan_i++) {
      for (size_t sample_i = 0; sample_i < signal.NumRows(); sample_i++) {
        mono_mat(sample_i, 0) += signal(sample_i, chan_i);
      }
    }
    return mono_mat / signal.NumCols();
  } else {
    return signal;
  }
}

// Combines the data from all channels into a single channel.
AudioSignal MiscAudio::ToMono(const AudioSignal& signal) {
  // If already Mono, nothing to do.
  if (signal.data_matrix.NumCols() > kNumChanMono) {
    const AMatrix<double> sig_mid_mat(MiscAudio::ToMono(signal.data_matrix));
    AudioSignal sig_mid;
    sig_mid.data_matrix = std::move(sig_mid_mat);
    sig_mid.sample_rate = signal.sample_rate;
    return sig_mid;
  } else {
    return signal;
  }
}

AudioSignal MiscAudio::LoadAsMono(const FilePath& path) {
  std::ifstream wav_file(path.Path().c_str(), std::ios::binary);
  if (wav_file) {
    std::stringstream wav_string_stream;
    wav_string_stream << wav_file.rdbuf();
    wav_file.close();
    return LoadAsMono(&wav_string_stream, path.Path());
  } else {
    ABSL_RAW_LOG(ERROR, "Could not find file %s.", path.Path().c_str());
    return AudioSignal();
  }
}

AudioSignal MiscAudio::LoadAsMono(std::stringstream* string_stream,
                                  absl::optional<std::string> filepath) {
  AudioSignal sig;
  WavReader wav_reader(string_stream);
  const size_t num_total_samples = wav_reader.GetNumTotalSamples();

  if (wav_reader.IsHeaderValid() && num_total_samples != 0) {
    std::vector<int16_t> interleaved_samples(num_total_samples);
    const auto num_samp_read =
        wav_reader.ReadSamples(num_total_samples, &interleaved_samples[0]);

    // Certain wav files are 'mostly valid' and have a slight difference with
    // the reported file length.  Warn for these.
    if (num_samp_read != num_total_samples) {
      ABSL_RAW_LOG(WARNING,
                   "Number of samples read (%lu) was less than the expected"
                   " number (%lu).",
                   num_samp_read, num_total_samples);
    }
    if (num_samp_read > 0) {
      const auto interleaved_norm_vec =
          MiscMath::NormalizeInt16ToDouble(interleaved_samples);
      const auto multi_chan_norm_vec = ExtractMultiChannel(
          wav_reader.GetNumChannels(), interleaved_norm_vec);

      const AMatrix<double> outMat(multi_chan_norm_vec);
      sig.data_matrix = outMat;
      sig.sample_rate = wav_reader.GetSampleRateHz();
      sig = MiscAudio::ToMono(sig);
    } else {
      if (filepath.has_value()) {
        ABSL_RAW_LOG(ERROR, "Error reading data for file %s.",
                     filepath->c_str());
      } else {
        ABSL_RAW_LOG(ERROR, "Error reading data from audio stream.");
      }
    }
  } else {
    if (filepath.has_value()) {
      ABSL_RAW_LOG(ERROR, "Error reading header for file %s.",
                   filepath->c_str());
    } else {
      ABSL_RAW_LOG(ERROR, "Error reading header from audio stream.");
    }
  }

  return sig;
}

std::vector<std::vector<double>> MiscAudio::ExtractMultiChannel(
    const int num_channels, const std::vector<double>& interleaved_vector) {
  assert(interleaved_vector.size() % num_channels == 0);
  const size_t sub_vector_size = interleaved_vector.size() / num_channels;
  std::vector<std::vector<double>> multi_channel_vec(
      num_channels, std::vector<double>(sub_vector_size));

  auto itr = interleaved_vector.cbegin();
  for (size_t sample = 0; sample < sub_vector_size; sample++) {
    for (int channel = 0; channel < num_channels; channel++) {
      multi_channel_vec[channel][sample] = *itr;
      itr++;
    }
  }
  return multi_channel_vec;
}

void MiscAudio::PrepareSpectrogramsForComparison(Spectrogram& reference,
                                                 Spectrogram& degraded) {
  reference.ConvertToDb();
  degraded.ConvertToDb();

  // An absolute threshold is also applied.
  reference.RaiseFloor(kNoiseFloorAbsoluteDb);
  degraded.RaiseFloor(kNoiseFloorAbsoluteDb);
  // Apply a per-frame relative threshold.
  // Note that this is not an STFT spectrogram, the spectrogram bins
  // here are each the RMS of a band filter output on the time domain signal.
  reference.RaiseFloorPerFrame(kNoiseFloorRelativeToPeakDb, degraded);
  // Normalize to a 0dB global floor (which is probably kNoiseFloorAbsoluteDb).
  double ref_floor = reference.Minimum();
  double deg_floor = degraded.Minimum();
  double lowest_floor = std::min(ref_floor, deg_floor);

  reference.SubtractFloor(lowest_floor);
  degraded.SubtractFloor(lowest_floor);
}
}  // namespace Visqol
