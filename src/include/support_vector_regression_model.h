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

#ifndef VISQOL_INCLUDE_SUPPORTVECTORREGRESSIONMODEL_H
#define VISQOL_INCLUDE_SUPPORTVECTORREGRESSIONMODEL_H

#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "file_path.h"
#include "machine_learning.h"
#include "svm.h"

namespace Visqol {
/**
 * This class represents a Support Vector Regression model. It utilises the
 * LIBSVM library.
 */
class SupportVectorRegressionModel {
 public:
  /**
   * The constructor for the SVR model.
   */
  SupportVectorRegressionModel();

  /**
   * The deconstructor for the SVR model. Cleans up the LIBSVM allocated memory.
   */
  ~SupportVectorRegressionModel();

  /**
   * Initialize the SVR model using a model file.
   *
   * @param model_path The path to the SVR model file.
   *
   * @return An OK status if the model was successfully initialized with the
   * model file. Else, an error status is returned.
   */
  absl::Status Init(const FilePath& model_path);

  /**
   * Initialize the SVR model using vectors of observations and targets. These
   * vectors are used to train the SVR model and save it for prediction usage.
   *
   * @param observations A vector of observations.
   * @param targets A vector of targets.
   */
  void Init(const std::vector<MlObservation>& observations,
            const std::vector<MlTarget>& targets);

  /**
   * Using the SVR model, predict a quality value for the given observation.
   *
   * @param observation A single observation.
   *
   * @return The predicted value from the observation.
   */
  double Predict(const MlObservation& observation) const;

 private:
  /**
   * The svm model provided by the LIBSVM library.
   */
  svm_model* model_;

  /**
   * Accessing the LibSVM external libraries for loading the model must be
   * performed by one thread at a time. This mutex is used to guard this
   * loading of the model file.
   */
  static absl::Mutex load_model_mutex_;

  /**
   * Pointer to an array of pointers used by the model for training.
   * As per the docs, this memory can't be freed while the model is used.
   */
  svm_node** observations_ptr_;

  /**
   * Number of observations in the training data.
   */
  size_t num_observations_;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_SUPPORTVECTORREGRESSIONMODEL_H
