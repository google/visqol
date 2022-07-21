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

#ifndef VISQOL_INCLUDE_NEUROGRAMSIMILARITYINDEXMEASURE_H
#define VISQOL_INCLUDE_NEUROGRAMSIMILARITYINDEXMEASURE_H

#include <vector>

#include "amatrix.h"
#include "image_patch_creator.h"
#include "patch_similarity_comparator.h"

namespace Visqol {
/**
 * Provides a neurogram similarity index measure (NSIM) implementation for a
 * patch similarity comparator. NSIM is a distance metric, adapted from the
 * image processing technique called structural similarity (SSIM) and is here
 * used to compare two patches taken from the reference and degraded
 * spectrograms.
 */
class NeurogramSimiliarityIndexMeasure : public PatchSimilarityComparator {
 public:
  // Docs inherited from parent.
  PatchSimilarityResult MeasurePatchSimilarity(
      const ImagePatch& ref_patch, const ImagePatch& deg_patch) const override;

 private:
  /**
   * The intensity range used during NSIM calculations.
   */
  const double intensity_range_ = 1.0;
};
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_NEUROGRAMSIMILARITYINDEXMEASURE_H
