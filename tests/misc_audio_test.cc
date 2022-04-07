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

#include "file_path.h"
#include "gtest/gtest.h"

namespace Visqol {
namespace {

const size_t kMonoTestsample_rate = 48000;
const size_t kMonoTestNumRows = 131444;
const size_t kMonoTestNumCols = 1;
const double kMonoDuration = 2.74;

const size_t kStereoTestsample_rate = 48000;
const size_t kStereoTestNumRows = 597784;
const size_t kStereoTestNumCols = 1;
const double kStereoDuration = 12.45;

// Duration tolerance of 10ms
const double kDurationTolerance = 0.01;

TEST(LoadAsMono, Mono) {
  FilePath mono_file{"testdata/clean_speech/CA01_01.wav"};

  auto wavreader_audio = Visqol::MiscAudio::LoadAsMono(mono_file);
  ASSERT_EQ(kMonoTestsample_rate, wavreader_audio.sample_rate);
  ASSERT_EQ(kMonoTestNumRows, wavreader_audio.data_matrix.NumRows());
  ASSERT_EQ(kMonoTestNumCols, wavreader_audio.data_matrix.NumCols());
  ASSERT_EQ(kMonoTestNumRows, wavreader_audio.data_matrix.NumElements());
  ASSERT_NEAR(kMonoDuration, wavreader_audio.GetDuration(), kDurationTolerance);
}

TEST(LoadAsMono, MonoFromStream) {
  std::ifstream wav_file("testdata/clean_speech/CA01_01.wav", std::ios::binary);
  std::stringstream wav_string_stream;
  wav_string_stream << wav_file.rdbuf();
  wav_file.close();

  auto wavreader_audio = Visqol::MiscAudio::LoadAsMono(&wav_string_stream);
  ASSERT_EQ(kMonoTestsample_rate, wavreader_audio.sample_rate);
  ASSERT_EQ(kMonoTestNumRows, wavreader_audio.data_matrix.NumRows());
  ASSERT_EQ(kMonoTestNumCols, wavreader_audio.data_matrix.NumCols());
  ASSERT_EQ(kMonoTestNumRows, wavreader_audio.data_matrix.NumElements());
  ASSERT_NEAR(kMonoDuration, wavreader_audio.GetDuration(), kDurationTolerance);
}

TEST(LoadAsMono, MonoFromEmptyStream) {
  std::stringstream wav_string_stream;
  auto wavreader_audio = Visqol::MiscAudio::LoadAsMono(&wav_string_stream);
  // sample_rate will be garbage in this case.
  ASSERT_NE(kMonoTestsample_rate, wavreader_audio.sample_rate);
}

TEST(LoadAsMono, Stereo) {
  FilePath stereo_file{
      "testdata/conformance_testdata_subset/"
      "guitar48_stereo.wav"};

  auto wavreader_audio = Visqol::MiscAudio::LoadAsMono(stereo_file);
  ASSERT_EQ(kStereoTestsample_rate, wavreader_audio.sample_rate);
  ASSERT_EQ(kStereoTestNumRows, wavreader_audio.data_matrix.NumRows());
  ASSERT_EQ(kStereoTestNumCols, wavreader_audio.data_matrix.NumCols());
  ASSERT_EQ(kStereoTestNumRows, wavreader_audio.data_matrix.NumElements());
  ASSERT_NEAR(kStereoDuration, wavreader_audio.GetDuration(),
              kDurationTolerance);
}

}  // namespace
}  // namespace Visqol
