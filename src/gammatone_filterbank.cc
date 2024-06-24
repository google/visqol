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

#include "gammatone_filterbank.h"

#include <utility>
#include <valarray>
#include <cstring>

#include "amatrix.h"
#include "equivalent_rectangular_bandwidth.h"
#include "signal_filter.h"

using namespace Eigen;

namespace Visqol {
GammatoneFilterBank::GammatoneFilterBank(const size_t num_bands,
                                         const double min_freq)
    : num_bands_(num_bands), min_freq_(min_freq)
      {
        fltr_cond_1_ = MatrixXd::Zero(num_bands, kFilterLength);
        fltr_cond_2_ = MatrixXd::Zero(num_bands, kFilterLength);
        fltr_cond_3_ = MatrixXd::Zero(num_bands, kFilterLength);
        fltr_cond_4_ = MatrixXd::Zero(num_bands, kFilterLength);
        a1 = MatrixXd::Zero(num_bands, kFilterLength);
        a2 = MatrixXd::Zero(num_bands, kFilterLength);
        a3 = MatrixXd::Zero(num_bands, kFilterLength);
        a4 = MatrixXd::Zero(num_bands, kFilterLength);
        b = MatrixXd::Zero(num_bands, kFilterLength);
      }

size_t GammatoneFilterBank::GetNumBands() const { return num_bands_; }

double GammatoneFilterBank::GetMinFreq() const { return min_freq_; }

void GammatoneFilterBank::ResetFilterConditions() {
  fltr_cond_1_.fill(0.0);
  fltr_cond_2_.fill(0.0);
  fltr_cond_3_.fill(0.0);
  fltr_cond_4_.fill(0.0);
}

void GammatoneFilterBank::SetFilterCoefficients(
    const AMatrix<double> &filter_coeffs) {
  for (size_t i = 0; i < num_bands_; ++i) {
    Eigen::RowVector3d a1_i = {filter_coeffs(i, ErbFiltersResult::A0) / filter_coeffs(i, ErbFiltersResult::Gain),
                         filter_coeffs(i, ErbFiltersResult::A11) / filter_coeffs(i, ErbFiltersResult::Gain),
                         filter_coeffs(i, ErbFiltersResult::A2) / filter_coeffs(i, ErbFiltersResult::Gain)};
    a1.row(i) = a1_i;

    Eigen::RowVector3d a2_i = {filter_coeffs(i, ErbFiltersResult::A0), filter_coeffs(i, ErbFiltersResult::A12), filter_coeffs(i, ErbFiltersResult::A2)};
    a2.row(i) = a2_i;

    Eigen::RowVector3d a3_i = {filter_coeffs(i, ErbFiltersResult::A0), filter_coeffs(i, ErbFiltersResult::A13), filter_coeffs(i, ErbFiltersResult::A2)};
    a3.row(i) = a3_i;

    Eigen::RowVector3d a4_i = {filter_coeffs(i, ErbFiltersResult::A0), filter_coeffs(i, ErbFiltersResult::A14), filter_coeffs(i, ErbFiltersResult::A2)};
    a4.row(i) = a4_i;

    Eigen::RowVector3d b_i = {filter_coeffs(i, ErbFiltersResult::B0), filter_coeffs(i, ErbFiltersResult::B1), filter_coeffs(i, ErbFiltersResult::B2)};
    b.row(i) = b_i;
  }
}

void BatchFilterSignal(const MatrixXd &numer_coeffs,
                       const MatrixXd &denom_coeffs,
                       const MatrixXd &input_signal,
                       MatrixXd &output_signal,
                       MatrixXd &init_conditions) {
  size_t i, n = denom_coeffs.cols();
  for (size_t m = 0; m < input_signal.rows(); m++) {
    output_signal.col(m) = numer_coeffs.col(0) * input_signal(m, 0) + init_conditions.col(0);
    for (i = 1; i < n; i++) {
      init_conditions.col(i - 1) = numer_coeffs.col(i) * input_signal(m, 0) +
        init_conditions.col(i) - denom_coeffs.col(i).cwiseProduct(output_signal.col(m));
    }
  }
}

void BatchFilterBands(const MatrixXd &numer_coeffs,
                      const MatrixXd &denom_coeffs,
                      const MatrixXd &input_signal,
                      MatrixXd &output_signal,
                      MatrixXd &init_conditions) {
  size_t i, n = denom_coeffs.cols();
  for (size_t m = 0; m < input_signal.cols(); m++) {
    output_signal.col(m) = numer_coeffs.col(0).cwiseProduct(input_signal.col(m)) + init_conditions.col(0);
    for (i = 1; i < n; i++) {
      init_conditions.col(i - 1) = numer_coeffs.col(i).cwiseProduct(input_signal.col(m)) +
        init_conditions.col(i) - denom_coeffs.col(i).cwiseProduct(output_signal.col(m));
    }
  }
}

MatrixXd GammatoneFilterBank::ApplyFilter(const MatrixXd &signal) {

  MatrixXd helper1 = Eigen::MatrixXd::Zero(num_bands_, signal.rows());
  MatrixXd helper2 = Eigen::MatrixXd::Zero(num_bands_, signal.rows());

  // Loop over each filter coefficient now to produce a filtered column.
  // First filter
  Visqol::BatchFilterSignal(a1, b, signal, helper1, fltr_cond_1_);

  // Second filter
  Visqol::BatchFilterBands(a2, b, helper1, helper2, fltr_cond_2_);

  // Third filter
  helper1.fill(0.0);
  Visqol::BatchFilterBands(a3, b, helper2, helper1, fltr_cond_3_);

  // Fourth filter
  helper2.fill(0.0);
  Visqol::BatchFilterBands(a4, b, helper1, helper2, fltr_cond_4_);

  return helper2;
}
}  // namespace Visqol
