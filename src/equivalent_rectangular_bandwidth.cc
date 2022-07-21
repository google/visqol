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

#include "equivalent_rectangular_bandwidth.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <vector>

#include "absl/base/internal/raw_logging.h"
#include "complex_valarray.h"

namespace Visqol {
ErbFiltersResult EquivalentRectangularBandwidth::MakeFilters(
    std::size_t sample_rate, std::size_t num_channels, double low_freq,
    double high_freq) {
  // Outputs the same as Slaney's but I've cleaned things up a bit.
  // The variable names are still quite meaningless.

  if (high_freq > sample_rate / 2.) {
    ABSL_RAW_LOG(
        WARNING,
        "EquivalentRectangularBandwidth::MakeFilters "
        "high_freq >= (sample_rate / 2), for sample_rate=%zu high_freq=%f. "
        "Falling back to (sample_rate / 2)",
        sample_rate, high_freq);
    high_freq = sample_rate / 2.;
  }
  auto cf1 = EquivalentRectangularBandwidth::CalcUniformCenterFreqs(
      low_freq, high_freq, num_channels);
  auto cf = ComplexValArray{cf1};

  double earQ = 9.26449;  // Glasberg and Moore Parameters
  double minBW = 24.7;
  double order = 1.0;

  auto B = ComplexValArray{num_channels};
  auto B1 = ComplexValArray{num_channels};
  for (std::size_t i = 0; i < num_channels; i++) {
    auto erb = pow(pow(cf[i] / earQ, order) + pow(minBW, order), 1.0 / order);
    B[i] = 1.019 * 2 * M_PI * erb;
  }
  double T = 1.0 / sample_rate;

  auto expBT = (B * T).exp();
  for (std::size_t i = 0; i < B.size(); i++) {
    B1[i] = -2.0 * cos(2.0 * cf[i] * M_PI * T) / expBT[i];
  }
  auto B2 = (B * T * -2.0).exp();

  auto b1 = ComplexValArray{(cf * 2.0 * M_PI * T).sin() * T};
  auto bPos = b1 * 2.0 * sqrt(3.0 + pow(+2.0, 1.5));
  auto bNeg = b1 * 2.0 * sqrt(3.0 + -pow(2.0, 1.5));
  auto a = ComplexValArray{((cf * 2.0 * M_PI * T).cos() * 2.0 * T)};
  auto A11 = -(a / expBT + bPos / expBT) / 2.0;
  auto A12 = -(a / expBT - bPos / expBT) / 2.0;
  auto A13 = -(a / expBT + bNeg / expBT) / 2.0;
  auto A14 = -(a / expBT - bNeg / expBT) / 2.0;

  // setup gain variables
  const std::complex<double> i{0.0, 1.0};
  auto p1 = pow(2.0, (3.0 / 2.0));
  auto s1 = sqrt(3.0 - p1);
  auto s2 = sqrt(3.0 + p1);
  auto xExp = (4.0 * i * cf * M_PI * T).exp();
  auto x01 = -2.0 * xExp * T;
  auto x02 = 2.0 * (-(B * T) + 2.0 * i * cf * M_PI * T).exp() * T;
  auto xCos = (2.0 * cf * M_PI * T).cos();
  auto xSin = (2.0 * cf * M_PI * T).sin();

  // calculate gain
  auto x12 = xCos - (s1 * xSin);
  auto x1 = x01 + (x02 * x12);
  auto x22 = xCos + (s1 * xSin);
  auto x2 = x01 + (x02 * x22);
  auto x32 = xCos - (s2 * xSin);
  auto x3 = x01 + (x02 * x32);
  auto x42 = xCos + (s2 * xSin);
  auto x4 = x01 + (x02 * x42);
  auto x5 =
      (-2.0 / (2 * B * T).exp()) - 2 * xExp + (2 * (1 + xExp)) / (B * T).exp();
  auto y = x5 ^ 4.0;
  auto gain = ((x1 * x2 * x3 * x4) / (x5 ^ 4.0)).abs();

  std::vector<double> A0(num_channels, T);
  std::vector<double> A2(num_channels, 0);
  std::vector<double> B0(num_channels, 1);

  // stick it all together so that we can turn it into a matrix later
  std::vector<std::vector<double>> vfCoeffs{A0,
                                            A11.ToDoubleVector(),
                                            A12.ToDoubleVector(),
                                            A13.ToDoubleVector(),
                                            A14.ToDoubleVector(),
                                            A2,
                                            B0,
                                            B1.ToDoubleVector(),
                                            B2.ToDoubleVector(),
                                            gain};

  ErbFiltersResult r;
  r.centerFreqs = cf.ToDoubleVector();
  r.filterCoeffs = vfCoeffs;
  return r;
}

std::vector<double> EquivalentRectangularBandwidth::CalcUniformCenterFreqs(
    double low_freq, double high_freq, std::size_t num_channels) {
  double earQ = 9.26449;  // Glasberg and Moore Parameters
  double minBW = 24.7;

  // All of the follow_freqing expressions are derived in Apple TR #35, "An
  // Efficient Implementation of the Patterson - Holdsworth Cochlear
  // Filter Bank."  See pages 33-34.
  auto a = -(earQ * minBW);
  auto b = -log(high_freq + earQ * minBW);
  auto c = log(low_freq + earQ * minBW);
  auto d = high_freq + earQ * minBW;
  auto e = (b + c) / num_channels;

  std::vector<double> cfs;
  cfs.reserve(num_channels);
  for (std::size_t i = 0; i < num_channels; i++) {
    auto f = exp((i + 1) * e) * d;
    cfs.push_back(a + f);
  }
  return cfs;
}

}  // namespace Visqol
