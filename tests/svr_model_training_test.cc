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

#include "absl/flags/flag.h"
#include "file_path.h"
#include "gtest/gtest.h"
#include "misc_vector.h"
#include "support_vector_regression_model.h"
#include "training_data_file_reader.h"

namespace Visqol {
namespace {

const double kTolerance = 2.0;
const FilePath kTargetsPath = FilePath(
    "testdata/svr_training/"
    "training_mat_tcdaudio14_aacvopus15_moslqs.txt");
const FilePath kObservationsPath = FilePath(
    "testdata/svr_training/"
    "training_mat_tcdaudio14_aacvopus15_fvnsims.txt");
const FilePath kDefaultAudioModelFile =
    FilePath(FilePath::currentWorkingDir() + "/model/libsvm_nu_svr_model.txt");

// This is the FVNSIM results for a ViSQOL comparison between
// contrabassoon48_stereo.wav and contrabassoon48_stereo_24kbps_aac.wav
const std::vector<double> kSampleObservation{
    0.853862, 0.680331, 0.535649, 0.639760, 0.029999, 0.058591, 0.077462,
    0.012432, 0.192035, 0.389230, 0.479403, 0.419914, 0.521414, 0.858340,
    0.884218, 0.864682, 0.868514, 0.845271, 0.850559, 0.877882, 0.903985,
    0.887572, 0.920558, 0.920375, 0.954934, 0.945048, 0.952716, 0.986600,
    0.987345, 0.936462, 0.856010, 0.829761};

/**
 * In this test, an SVR model will be initialized using the default model file.
 * A second SVR model will then be initialized with some target (i.e. moslqo)
 * and observation (i.e. fvnsim) data. This second SVR model will use this
 * target and observation data to train itself. This target and observation
 * data is similar to the data that was used to produce the default model file.
 * Therefore, we would expect both SVR models that we initialised to produce
 * similar predictions - which is what this test will test for.
 */
TEST(SupportVectorRegressionModel_Test, vs_model_file) {
  // Init the default model.
  SupportVectorRegressionModel model_default;
  ASSERT_TRUE(model_default.Init(kDefaultAudioModelFile).ok());

  // Init the second model with data to train on.
  auto targets_mat = TrainingDataFileReader::Read(kTargetsPath, ',');
  auto observations_mat = TrainingDataFileReader::Read(kObservationsPath, ',');
  auto targets_vec = MiscVector::ConvertVecOfVecToVec(targets_mat);
  SupportVectorRegressionModel model_trained;
  model_trained.Init(observations_mat, targets_vec);

  // Generate a prediction with the default SVR model.
  double prediction_model_file = model_default.Predict(kSampleObservation);

  // Generate a prediction with the trained SVR model.
  double prediction_targ_obv = model_trained.Predict(kSampleObservation);

  // Don't expect the values to be anywhere near each other.
  EXPECT_NEAR(prediction_model_file, prediction_targ_obv, kTolerance);
}
}  // namespace
}  // namespace Visqol
