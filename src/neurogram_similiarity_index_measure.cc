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

#include "neurogram_similiarity_index_measure.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "convolution_2d.h"

namespace Visqol {
PatchSimilarityResult NeurogramSimiliarityIndexMeasure::MeasurePatchSimilarity(
    const ImagePatch& ref_patch, const ImagePatch& deg_patch) const {
  const std::vector<double> w = {
      0.0113033910173052, 0.0838251475442633, 0.0113033910173052,
      0.0838251475442633, 0.619485845753726,  0.0838251475442633,
      0.0113033910173052, 0.0838251475442633, 0.0113033910173052};
  AMatrix<double> window(3, 3, std::move(w));

  std::vector<double> k{0.01, 0.03};
  double c1 = pow(k[0] * intensity_range_, 2);
  double c3 = pow(k[1] * intensity_range_, 2) / 2;

  auto mu_r = Convolution2D<double>::Valid2DConvWithBoundary(window, ref_patch);
  auto mu_d = Convolution2D<double>::Valid2DConvWithBoundary(window, deg_patch);
  auto ref_mu_sq = mu_r.PointWiseProduct(mu_r);
  auto deg_mu_sq = mu_d.PointWiseProduct(mu_d);
  auto mu_r_mu_d = mu_r.PointWiseProduct(mu_d);
  auto ref_neuro_sq = ref_patch.PointWiseProduct(ref_patch);
  auto deg_neuro_sq = deg_patch.PointWiseProduct(deg_patch);
  auto conv2_ref_neuro_sq =
      Convolution2D<double>::Valid2DConvWithBoundary(window, ref_neuro_sq);
  auto sigma_r_sq = conv2_ref_neuro_sq - ref_mu_sq;
  auto conv2_deg_neuro_sq =
      Convolution2D<double>::Valid2DConvWithBoundary(window, deg_neuro_sq);
  auto sigma_d_sq = conv2_deg_neuro_sq - deg_mu_sq;
  auto ref_neuro_deg = ref_patch.PointWiseProduct(deg_patch);
  auto conv2_ref_neuro_deg =
      Convolution2D<double>::Valid2DConvWithBoundary(window, ref_neuro_deg);
  auto sigma_r_d = conv2_ref_neuro_deg - mu_r_mu_d;

  auto intensity_numer = mu_r_mu_d * 2.0 + c1;
  auto intensity_denom = ref_mu_sq + deg_mu_sq + c1;
  auto intensity = intensity_numer.PointWiseDivide(intensity_denom);

  auto structure_numer = sigma_r_d + c3;
  auto structure_denom = sigma_r_sq.PointWiseProduct(sigma_d_sq);
  std::transform(structure_denom.begin(), structure_denom.end(),
                 structure_denom.begin(),
                 [&](decltype(*structure_denom.begin())& d) {
                   // Avoid a nan is when stddev is negative.
                   // This occasionally happens with silent patches,
                   // which generate an epison negative value.
                   return (d < 0.) ? c3 : (sqrt(d) + c3);
                 });
  auto structure = structure_numer.PointWiseDivide(structure_denom);
  auto sim_map = intensity.PointWiseProduct(structure);

  // These three matrices correspond to the similarity_result.proto fields
  // such as fvnsim.
  auto freq_band_deg_energy = deg_patch.Mean(kDimension::ROW);
  auto freq_band_means = sim_map.Mean(kDimension::ROW);
  auto freq_band_stddevs = sim_map.StdDev(kDimension::ROW);

  double freq_band_sim_sum = 0;
  std::for_each(
      freq_band_means.begin(), freq_band_means.end(),
      [&](decltype(*freq_band_means.begin())& d) { freq_band_sim_sum += d; });
  auto mean_freq_band_means =
      freq_band_sim_sum / freq_band_means.NumRows();  // A.K.A. NSIM

  PatchSimilarityResult r;
  r.similarity = mean_freq_band_means;
  r.freq_band_deg_energy = std::move(freq_band_deg_energy);
  r.freq_band_means = std::move(freq_band_means);
  r.freq_band_stddevs = std::move(freq_band_stddevs);
  return r;
}
}  // namespace Visqol
