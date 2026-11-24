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

#ifndef VISQOL_INCLUDE_MODEL_FILE_PATH_H_
#define VISQOL_INCLUDE_MODEL_FILE_PATH_H_

namespace Visqol {

/**
 * The relative location of the default SVR model file for audio.
 */
inline const char kDefaultAudioModelFile[] = "/model/libsvm_nu_svr_model.txt";

/**
 * The relative location of the default TFLite model file for speech.
 */
inline const char kDefaultSpeechModelFile[] =
    "/model/"
    "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400"
    "_train1_7_raw.tflite";

}  // namespace Visqol

#endif  // THIRD_PARTY_VISQOL_INCLUDE_MODEL_FILE_PATH_H_
