/*
 * Copyright 2022 Google LLC
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
#include "tflite_quality_mapper.h"

#include "absl/base/internal/raw_logging.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace Visqol {

TFLiteQualityMapper::TFLiteQualityMapper(absl::string_view model_path,
                                         int num_frequency_bands)
    : model_path_(model_path), num_frequency_bands_(num_frequency_bands) {}

absl::Status TFLiteQualityMapper::Init() {
  ABSL_RAW_LOG(INFO, "Loading TF Lattice TFLite model at %s",
               model_path_.c_str());

  model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
  if (model_ == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Could not build TFLite FlatBufferModel from path: ", model_path_));
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  if (builder.SetNumThreads(1) != 0) {
    return absl::UnknownError("Error when calling SetNumThreads(1).");
  }

  // Create the interpreter.
  if (builder(&interpreter_) != kTfLiteOk) {
    return absl::UnknownError("Error when creating the TFLite interpreter.");
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    return absl::ResourceExhaustedError("Could not allocate TFLite tensors.");
  }

  // Use XNN if possible.
  TfLiteXNNPackDelegateOptions opts{.num_threads = 1};
  delegate_ =
      std::unique_ptr<TfLiteDelegate, std::function<void(TfLiteDelegate*)>>(
          TfLiteXNNPackDelegateCreate(&opts), &TfLiteXNNPackDelegateDelete);

  // Allow dynamic tensors.
  // TODO(b/204470960): Remove this flag once the bug is fixed.
  delegate_->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;

  TfLiteStatus xnn_status =
      interpreter_->ModifyGraphWithDelegate(delegate_.get());
  if (xnn_status == kTfLiteDelegateError) {
    ABSL_RAW_LOG(WARNING,
                 "Failed to set XNNPack delegate; continuing without.");
  } else if (xnn_status != kTfLiteOk) {
    return absl::UnknownError("Fatal error when setting XNNPack delegate.");
  }

  return absl::Status();
}

double TFLiteQualityMapper::PredictQuality(
    const std::vector<double>& fvnsim_vector,
    const std::vector<double>& fvnsim10_vector,
    const std::vector<double>& fstdnsim_vector,
    const std::vector<double>& fvdegenergy_vector) const {
  tflite::SignatureRunner* predict_runner =
      interpreter_->GetSignatureRunner("predict");

  std::string name;
  TfLiteTensor* input_tensor;
  const std::vector<std::string> base_names = {"fvnsim", "fvnsim10_",
                                               "fstdnsim", "fvdegenergy"};
  const std::vector<std::vector<double>> features = {
      fvnsim_vector, fvnsim10_vector, fstdnsim_vector, fvdegenergy_vector};

  for (int band_idx = 0; band_idx < num_frequency_bands_; ++band_idx) {
    for (int feature_idx = 0; feature_idx < base_names.size(); ++feature_idx) {
      name = absl::StrCat(base_names[feature_idx], band_idx);
      input_tensor = predict_runner->input_tensor(name.c_str());
      *input_tensor->data.f = features[feature_idx][band_idx];
    }
  }
  // Fill in the final tau quantile feature.
  input_tensor = predict_runner->input_tensor("tau");
  *input_tensor->data.f = 0.5;

  predict_runner->Invoke();
  const TfLiteTensor* output_tensor =
      predict_runner->output_tensor("predictions");
  return *output_tensor->data.f;
}
}  // namespace Visqol
