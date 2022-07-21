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

#ifndef VISQOL_INCLUDE_MATLABMATRIXFILEREADER_H
#define VISQOL_INCLUDE_MATLABMATRIXFILEREADER_H

#include <vector>

#include "file_path.h"

namespace Visqol {
/**
 * This class is used to read in files containing test data to be used for
 * training the SVR model.
 *
 * There are two training files needed. One contains targets (i.e. moslqo) and
 * the other contains observations (i.e. fvnsim).
 *
 * Each row in the targets file should hold a single target. Each row in the
 * observations file should hold a single set of observations, delimited with a
 * comma, without spaces between the values. Rows should not have spaces at the
 * end. There also needs to be a blank line at the bottom of the file. Each row
 * index in both files must contain data that matches the row index in the
 * other file i.e. the first row in the targets file must correspond to the
 * first row in the observations file.
 *
 * e.g. targets file:
 *   4.778303
 *   4.827275
 *   3.817091
 *
 * e.g. observations file:
 *   0.997688956727337,0.999038085478193,0.999394555451623 etc.
 *   0.972244552126234,0.989593523387658,0.996950560094463 etc.
 *   0.883575729725838,0.919446119013846,0.953594034503209 etc.
 *
 * Full file examples can be found in 'testdata/svr_training'.
 *
 * These files can be created from the ViSQOL Matlab training script with the
 * following function calls:
 *
 *    dlmwrite('training_mat_tcdaudio14_aacvopus15_moslqs.txt', allT.moslqs,
 *        'delimiter', ',', 'precision', 15);
 *
 *    dlmwrite('training_mat_tcdaudio14_aacvopus15_fvnsims.txt',
 *        trainingFvnsimMatrix, 'delimiter', ',', 'precision', 15);
 */
class TrainingDataFileReader {
 public:
  /**
   * Read in a file containing either targets or observations for training the
   * SVR model with.
   *
   * @param data_filepath The filepath to the file containing the training data
   *    to read.
   * @param delimiter The delimiter used to seperate values on a row in the
   *    training data file e.g. a comma.
   *
   * @return A vector of vector, where each child vector corresponds to a row
   *    in the training data file.
   */
  static std::vector<std::vector<double>> Read(const FilePath& data_filepath,
                                               const char delimiter);
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_MATLABMATRIXFILEREADER_H
