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
#ifndef THIRD_PARTY_VISQOL_SRC_INCLUDE_TFLITE_QUALITY_MAPPER_H_
#define THIRD_PARTY_VISQOL_SRC_INCLUDE_TFLITE_QUALITY_MAPPER_H_

#include <memory>

#include "absl/status/status.h"
#include "similarity_to_quality_mapper.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace Visqol {

/**
 * This class represents a deep lattice network model quality mapper.
 * This class loads a TFLite model that has been converted with the
 * script at scripts/convert_saved_model_to_tflite.py.
 */
class TFLiteQualityMapper : public SimilarityToQualityMapper {
 public:
  TFLiteQualityMapper(absl::string_view model_path, int num_frequency_bands);
  absl::Status Init() override;

  double PredictQuality(
      const std::vector<double>& fvnsim_vector,
      const std::vector<double>& fvnsim10_vector,
      const std::vector<double>& fstdnsim_vector,
      const std::vector<double>& fvdegenergy_vector) const override;

 private:
  const std::string model_path_;
  const int num_frequency_bands_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<TfLiteDelegate, std::function<void(TfLiteDelegate*)>>
      delegate_;
};

}  // namespace Visqol

#endif  // THIRD_PARTY_VISQOL_SRC_INCLUDE_TFLITE_QUALITY_MAPPER_H_
