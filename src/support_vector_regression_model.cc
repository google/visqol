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

#include "support_vector_regression_model.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "file_path.h"
#include "libsvm_target_observation_convertor.h"
#include "svm.h"

namespace Visqol {

absl::Mutex SupportVectorRegressionModel::load_model_mutex_{};

SupportVectorRegressionModel::SupportVectorRegressionModel()
    : observations_ptr_{nullptr}, num_observations_{0} {}

SupportVectorRegressionModel::~SupportVectorRegressionModel() {
  if (model_) {
    svm_free_model_content(model_);
    svm_free_and_destroy_model(&model_);
    // Free the 2D array of observations.
    for (size_t i = 0; i < num_observations_; ++i) {
      free(observations_ptr_[i]);
    }
    free(observations_ptr_);
  }
}

void SupportVectorRegressionModel::Init(
    const std::vector<MlObservation>& observations,
    const std::vector<MlTarget>& targets) {
  // Assumes all observations have same number of features
  size_t num_features = observations[0].size();

  double* targets_ptr = const_cast<double*>(&targets[0]);

  const LibSvmTargetObservationConvertor conv;
  observations_ptr_ = conv.ConvertObservations(observations, num_features);
  num_observations_ = observations.size();

  // Setup the SVM problem.
  svm_problem problem;
  problem.l = targets.size();
  problem.y = targets_ptr;
  problem.x = observations_ptr_;

  // Setup the SVM parameters.
  svm_parameter param;
  param.C = 0.4;            // cost
  param.svm_type = NU_SVR;  // SVR
  param.kernel_type = RBF;  // radial
  param.nu = 0.6;           // SVR nu

  // These values are the defaults used in the Matlab version
  // as found in svm_model_matlab.c
  param.gamma = 1.0 / static_cast<double>(num_features);
  param.coef0 = 0;
  param.cache_size = 100;  // in MB
  param.shrinking = 1;
  param.probability = 0;
  param.degree = 3;
  param.eps = 1e-3;
  param.p = 0.1;
  param.shrinking = 1;
  param.probability = 0;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;

  model_ = svm_train(&problem, &param);
}

double SupportVectorRegressionModel::Predict(
    const std::vector<double>& observation) const {
  const LibSvmTargetObservationConvertor conv;
  svm_node* obs_node = conv.ConvertObservation(observation);
  const double prediction = svm_predict(model_, obs_node);
  free(obs_node);

  return prediction;
}

absl::Status SupportVectorRegressionModel::Init(const FilePath& model_path) {
  absl::MutexLock lock(&load_model_mutex_);
  model_ = svm_load_model(model_path.Path().c_str());

  if (model_ == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to load the SVR model file: ", model_path.Path()));
  }

  return absl::Status();
}

}  // namespace Visqol
