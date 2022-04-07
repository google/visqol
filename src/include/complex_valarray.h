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

#ifndef VISQOL_INCLUDE_COMPLEXVALARRAY_H
#define VISQOL_INCLUDE_COMPLEXVALARRAY_H

#include <algorithm>
#include <complex>
#include <cstdio>
#include <valarray>
#include <vector>

namespace Visqol {

class ComplexValArray {
 private:
  std::valarray<std::complex<double>> va_;

 public:
  explicit ComplexValArray(size_t size) : va_(size) {}

  explicit ComplexValArray(const std::vector<double>& v) {
    va_.resize(v.size());
    for (size_t i = 0; i < va_.size(); i++) {
      va_[i].real(v[i]);
    }
  }

  explicit ComplexValArray(const std::valarray<double>& v) {
    va_.resize(v.size());
    for (size_t i = 0; i < va_.size(); i++) {
      va_[i].real(v[i]);
    }
  }

  size_t size() const { return va_.size(); }

  ComplexValArray operator+(double d) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] + d;
    }
    return out;
  }

  ComplexValArray operator+(const ComplexValArray& v) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] + v.va_[i];
    }
    return out;
  }

  ComplexValArray operator-() const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = -va_[i];
    }
    return out;
  }

  ComplexValArray operator-(double d) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] - d;
    }
    return out;
  }

  ComplexValArray operator-(const ComplexValArray& v) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] - v.va_[i];
    }
    return out;
  }

  ComplexValArray operator*(double d) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] * d;
    }
    return out;
  }

  ComplexValArray operator*(const std::complex<double>& c) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] * c;
    }
    return out;
  }

  ComplexValArray operator*(const ComplexValArray& v) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] * v.va_[i];
    }
    return out;
  }

  ComplexValArray operator/(double d) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] / d;
    }
    return out;
  }

  ComplexValArray operator/(const ComplexValArray& v) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] / v.va_[i];
    }
    return out;
  }

  ComplexValArray operator/(const std::valarray<double>& v) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = va_[i] / v[i];
    }
    return out;
  }

  ComplexValArray operator^(double d) const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = pow(va_[i], d);
    }
    return out;
  }

  std::complex<double>& operator[](size_t index) { return va_[index]; }

  std::complex<double> at(size_t index) const { return va_[index]; }

  ComplexValArray exp() const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = std::exp(va_[i]);
    }
    return out;
  }

  ComplexValArray sin() const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = std::sin(va_[i]);
    }
    return out;
  }

  ComplexValArray cos() const {
    auto out = ComplexValArray{va_.size()};
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = std::cos(va_[i]);
    }
    return out;
  }

  std::vector<double> abs() const {
    auto out = std::vector<double>(va_.size());
    for (size_t i = 0; i < va_.size(); i++) {
      out[i] = std::abs(va_[i]);
    }
    return out;
  }

  std::vector<double> ToDoubleVector() {
    std::vector<double> out;
    for (size_t i = 0; i < va_.size(); i++) {
      out.push_back(va_[i].real());
    }
    return out;
  }

  void PrintSummary(const char* c) const {
    // printf("%s\n", c);
    size_t outMatSize = va_.size();
    printf("first five\n");
    for (size_t i = 0; i < 5; i++) {
      printf("%s[%2zu] = %9.20f , %9.20f\n", c, i, va_[i].real(),
             va_[i].imag());
    }
    printf("middle \n");
    for (size_t i = (outMatSize / 2) - 4; i < (outMatSize / 2) + 6; i++) {
      printf("%s[%2zu] = %9.20f , %9.20f\n", c, i, va_[i].real(),
             va_[i].imag());
    }
    printf("last five\n");
    for (size_t i = outMatSize - 6; i < outMatSize; i++) {
      printf("%s[%2zu] = %9.20f , %9.20f\n", c, i, va_[i].real(),
             va_[i].imag());
    }
  }
};

ComplexValArray operator*(double x, const ComplexValArray& v);
ComplexValArray operator+(double x, const ComplexValArray& v);
ComplexValArray operator*(std::complex<double> x, const ComplexValArray& v);
ComplexValArray operator/(double d, const ComplexValArray& v);
}  // namespace Visqol

#endif  // VISQOL_INCLUDE_COMPLEXVALARRAY_H
