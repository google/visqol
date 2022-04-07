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

#include "fast_fourier_transform.h"

#include <complex>
#include <string>
#include <valarray>

#include "absl/memory/memory.h"
#include "amatrix.h"
#include "gtest/gtest.h"
#include "test_utility.h"

namespace Visqol {
namespace {

const double kTolerance = 0.00000001;

// This 65 sample matrix was cut from a genuine signal to produce realistic
// input. Using 65 samples to ensure non-power of two inputs are handled
// correctly.
const AMatrix<double> k65Samples{std::valarray<double>{
    0.000150529,  5.89739e-05,  -9.36187e-05, -9.36187e-05, 0.000394677,
    0.000303122,  -9.36187e-05, 0.000303122,  8.94924e-05,  0.000150529,
    -2.06314e-06, -0.000154656, 2.84554e-05,  0.000272603,  -0.000185174,
    5.89739e-05,  0.000364159,  0.000425196,  0.000120011,  5.89739e-05,
    0.000120011,  0.000211566,  0.000150529,  2.84554e-05,  0.000120011,
    0.000150529,  8.94924e-05,  -9.36187e-05, 0.000242085,  0.000486233,
    8.94924e-05,  0.000120011,  8.94924e-05,  8.94924e-05,  0.000150529,
    -0.000154656, -2.06314e-06, 2.84554e-05,  0.000150529,  -3.25817e-05,
    5.89739e-05,  5.89739e-05,  -0.000124137, 5.89739e-05,  2.84554e-05,
    0.000181048,  -0.000124137, -0.000368285, -9.36187e-05, -0.000246211,
    -3.25817e-05, 2.84554e-05,  -6.31002e-05, -9.36187e-05, -2.06314e-06,
    -2.06314e-06, -0.000185174, -0.000124137, 2.84554e-05,  8.94924e-05,
    -0.00027673,  -6.31002e-05, -0.000215693, 0.000425196,  8.94924e-05}};

// This 65 sample complex matrix was produced from the above input, with 0.0
// used as the imaginary part.  Using 65 samples to ensure non-power of two
// inputs are handled correctly.
const AMatrix<std::complex<double>> k65SamplesComplex{
    std::valarray<std::complex<double>>{
        {0.000150529, 0.0},  {5.89739e-05, 0.0},  {-9.36187e-05, 0.0},
        {-9.36187e-05, 0.0}, {0.000394677, 0.0},  {0.000303122, 0.0},
        {-9.36187e-05, 0.0}, {0.000303122, 0.0},  {8.94924e-05, 0.0},
        {0.000150529, 0.0},  {-2.06314e-06, 0.0}, {-0.000154656, 0.0},
        {2.84554e-05, 0.0},  {0.000272603, 0.0},  {-0.000185174, 0.0},
        {5.89739e-05, 0.0},  {0.000364159, 0.0},  {0.000425196, 0.0},
        {0.000120011, 0.0},  {5.89739e-05, 0.0},  {0.000120011, 0.0},
        {0.000211566, 0.0},  {0.000150529, 0.0},  {2.84554e-05, 0.0},
        {0.000120011, 0.0},  {0.000150529, 0.0},  {8.94924e-05, 0.0},
        {-9.36187e-05, 0.0}, {0.000242085, 0.0},  {0.000486233, 0.0},
        {8.94924e-05, 0.0},  {0.000120011, 0.0},  {8.94924e-05, 0.0},
        {8.94924e-05, 0.0},  {0.000150529, 0.0},  {-0.000154656, 0.0},
        {-2.06314e-06, 0.0}, {2.84554e-05, 0.0},  {0.000150529, 0.0},
        {-3.25817e-05, 0.0}, {5.89739e-05, 0.0},  {5.89739e-05, 0.0},
        {-0.000124137, 0.0}, {5.89739e-05, 0.0},  {2.84554e-05, 0.0},
        {0.000181048, 0.0},  {-0.000124137, 0.0}, {-0.000368285, 0.0},
        {-9.36187e-05, 0.0}, {-0.000246211, 0.0}, {-3.25817e-05, 0.0},
        {2.84554e-05, 0.0},  {-6.31002e-05, 0.0}, {-9.36187e-05, 0.0},
        {-2.06314e-06, 0.0}, {-2.06314e-06, 0.0}, {-0.000185174, 0.0},
        {-0.000124137, 0.0}, {2.84554e-05, 0.0},  {8.94924e-05, 0.0},
        {-0.00027673, 0.0},  {-6.31002e-05, 0.0}, {-0.000215693, 0.0},
        {0.000425196, 0.0},  {8.94924e-05, 0.0}}};

// This output matrix was produced using Matlab to process the above input 65
// sample matrix. The fft size was specified as 128.
const AMatrix<std::complex<double>> k65SamplesForwardFFT{
    std::valarray<std::complex<double>>{
        {0.00322292904000000, 0.00000000000000},
        {0.00301682385173052, -0.00234848985608509},
        {-0.00103278592163890, -0.00340299713362453},
        {-0.00144609203801138, 0.000879145244592523},
        {0.00111610613898786, -0.000302181002376930},
        {-0.000318859126372253, -0.000367181194435114},
        {0.000673949619008653, -0.000310723599151972},
        {-3.49644446093239e-05, -0.000563954656030864},
        {0.000339817207812471, -0.000595601799964675},
        {0.000101609221059575, -0.000744744002384681},
        {-0.000586264026928805, -0.00191608882520754},
        {-0.00259248055250948, 0.000471875974326385},
        {0.000227097152831817, 0.00110287047767037},
        {-0.000722511594188183, 0.000313570758095348},
        {0.000407311564417282, 0.000853530053469442},
        {-0.000195617316848781, 0.000154945338489743},
        {0.000233147810677542, 0.000734349802916912},
        {0.000691246491351084, 0.000556673374244994},
        {0.000755913153376688, -0.00108748635434686},
        {-0.00157770746633928, 0.000303794653066800},
        {0.000896182317264214, 0.000835164328459981},
        {-0.000822997471347754, 0.000140031434634188},
        {0.00109713925301471, 0.00153219988568366},
        {0.000868409640532475, -0.000373929386206932},
        {0.000535362381861835, -0.000461656449099621},
        {-0.000994143865712840, 0.000213392912383107},
        {0.00128029686672661, 0.000927358696182279},
        {-0.000558399599341835, -0.000596241613983044},
        {0.000590345970887055, 0.00153746054605763},
        {0.000123623000739260, 0.000159309107927759},
        {0.00177647225128173, 0.00233356699195329},
        {0.00301793611579363, -0.00119668950186613},
        {0.00124919604000000, -0.00161748004000000},
        {-0.000322728346193144, -0.00214236853854895},
        {-0.000527239999720400, 0.000360874054041502},
        {0.000703316552686529, -0.000897998866185932},
        {-0.000525863898635626, -0.000277837377775431},
        {0.000432650139994782, -9.95533541164137e-05},
        {0.000151305582086382, -0.000519865718932982},
        {-6.14596442789589e-05, -7.84463404614790e-06},
        {0.00118869378923226, -0.000194024415891887},
        {0.000172206834052672, -0.00166972697788257},
        {-0.000273651726999761, -0.000867166897935699},
        {-0.00113046085098264, -0.00125829455984567},
        {-0.000958346131576863, 0.000520388328872528},
        {-0.000253487462883782, -0.000415594953354160},
        {-0.000970643681183129, -4.25544922587714e-05},
        {-0.00121987061492059, 0.000551139974883648},
        {0.000189986069322458, 0.00146679520291691},
        {3.91316837376560e-05, -3.82684506226442e-05},
        {0.000364085132964744, 0.00119600012367842},
        {0.000438119142196402, -0.000741712630986012},
        {-0.000619592071135628, 0.00114204782699505},
        {0.00128409408020558, 0.000424538502147734},
        {0.000651940026079785, -0.000330701565255067},
        {-0.000378907596869206, -0.000711769960239768},
        {3.12982109343697e-06, 0.00138106567324306},
        {0.00118819196612810, -0.000672809033262345},
        {-0.000236735235269374, 0.000463610199849060},
        {0.00121540962643760, -0.000589114879992856},
        {-0.000254141478622833, -5.44911772978580e-05},
        {0.00102320576458072, -0.000954731288479258},
        {-0.00112262885721622, -0.000516904492870956},
        {0.000267885080182829, -0.000205545255713845},
        {-0.00110072968000000, 0.00000000000000},
        {0.000267885080182829, 0.000205545255713845},
        {-0.00112262885721622, 0.000516904492870956},
        {0.00102320576458072, 0.000954731288479258},
        {-0.000254141478622833, 5.44911772978580e-05},
        {0.00121540962643760, 0.000589114879992856},
        {-0.000236735235269374, -0.000463610199849060},
        {0.00118819196612810, 0.000672809033262345},
        {3.12982109343697e-06, -0.00138106567324306},
        {-0.000378907596869206, 0.000711769960239768},
        {0.000651940026079785, 0.000330701565255067},
        {0.00128409408020558, -0.000424538502147734},
        {-0.000619592071135628, -0.00114204782699505},
        {0.000438119142196402, 0.000741712630986012},
        {0.000364085132964744, -0.00119600012367842},
        {3.91316837376560e-05, 3.82684506226442e-05},
        {0.000189986069322458, -0.00146679520291691},
        {-0.00121987061492059, -0.000551139974883648},
        {-0.000970643681183129, 4.25544922587714e-05},
        {-0.000253487462883782, 0.000415594953354160},
        {-0.000958346131576863, -0.000520388328872528},
        {-0.00113046085098264, 0.00125829455984567},
        {-0.000273651726999761, 0.000867166897935699},
        {0.000172206834052672, 0.00166972697788257},
        {0.00118869378923226, 0.000194024415891887},
        {-6.14596442789589e-05, 7.84463404614790e-06},
        {0.000151305582086382, 0.000519865718932982},
        {0.000432650139994782, 9.95533541164137e-05},
        {-0.000525863898635626, 0.000277837377775431},
        {0.000703316552686529, 0.000897998866185932},
        {-0.000527239999720400, -0.000360874054041502},
        {-0.000322728346193144, 0.00214236853854895},
        {0.00124919604000000, 0.00161748004000000},
        {0.00301793611579363, 0.00119668950186613},
        {0.00177647225128173, -0.00233356699195329},
        {0.000123623000739260, -0.000159309107927759},
        {0.000590345970887055, -0.00153746054605763},
        {-0.000558399599341835, 0.000596241613983044},
        {0.00128029686672661, -0.000927358696182279},
        {-0.000994143865712840, -0.000213392912383107},
        {0.000535362381861835, 0.000461656449099621},
        {0.000868409640532475, 0.000373929386206932},
        {0.00109713925301471, -0.00153219988568366},
        {-0.000822997471347754, -0.000140031434634188},
        {0.000896182317264214, -0.000835164328459981},
        {-0.00157770746633928, -0.000303794653066800},
        {0.000755913153376688, 0.00108748635434686},
        {0.000691246491351084, -0.000556673374244994},
        {0.000233147810677542, -0.000734349802916912},
        {-0.000195617316848781, -0.000154945338489743},
        {0.000407311564417282, -0.000853530053469442},
        {-0.000722511594188183, -0.000313570758095348},
        {0.000227097152831817, -0.00110287047767037},
        {-0.00259248055250948, -0.000471875974326385},
        {-0.000586264026928805, 0.00191608882520754},
        {0.000101609221059575, 0.000744744002384681},
        {0.000339817207812471, 0.000595601799964675},
        {-3.49644446093239e-05, 0.000563954656030864},
        {0.000673949619008653, 0.000310723599151972},
        {-0.000318859126372253, 0.000367181194435114},
        {0.00111610613898786, 0.000302181002376930},
        {-0.00144609203801138, -0.000879145244592523},
        {-0.00103278592163890, 0.00340299713362453},
        {0.00301682385173052, 0.00234848985608509}}};

// This vector of zeros is included to ensure the fft will handle cases where
// the input is zero.
const AMatrix<double> k65SamplesZero{std::valarray<double>{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

// Test the function Forward1d(FftManager,AMatrix<double>). Ensure
// that the correct matrix is returned for an input matrix of random doubles.
TEST(FastFourierTransformTest, Forward1d_MatrixContentsRandom) {
  auto fft_manager = absl::make_unique<FftManager>(k65Samples.NumElements());
  auto fft_out_matrix =
      FastFourierTransform::Forward1d(fft_manager, k65Samples);

  std::string fail_msg;
  ASSERT_TRUE(CompareComplexMatrix(fft_out_matrix, k65SamplesForwardFFT,
                                   kTolerance, &fail_msg))
      << fail_msg;
}

// Test the function Forward1d(FftManager,AMatrix<double>,size_t).
// Ensure that the correct matrix is returned for an input matrix of doubles
// of value zero. Specify a new size for the fft.
TEST(FastFourierTransformTest, Forward1d_Resize_MatrixContentsRandom) {
  auto fft_manager = absl::make_unique<FftManager>(k65Samples.NumElements());
  auto fft_out_matrix = FastFourierTransform::Forward1d(
      fft_manager, k65Samples, k65Samples.NumElements() + 1);

  std::string fail_msg;
  ASSERT_TRUE(CompareComplexMatrix(fft_out_matrix, k65SamplesForwardFFT,
                                   kTolerance, &fail_msg))
      << fail_msg;
}

// Test the function Inverse1dPffft(FftManager, AMatrix<complex<double>>).
// Ensure that following a forward fft, the inverse fft will reconstruct the
// original input (in complex format).
TEST(FastFourierTransformTest, FftReconstruct) {
  auto fft_manager = absl::make_unique<FftManager>(k65Samples.NumElements());
  auto fft_forward = FastFourierTransform::Forward1d(fft_manager, k65Samples);
  auto fft_inverse = FastFourierTransform::Inverse1d(fft_manager, fft_forward);

  std::string fail_msg;
  ASSERT_TRUE(CompareComplexMatrix(k65SamplesComplex, fft_inverse, kTolerance,
                                   &fail_msg))
      << fail_msg;
}

// Test the function Inverse1dConjSymPffft(FftManager, AMatrix<complex<double>>)
// Ensure that following a forward fft, the inverse fft will reconstruct the
// original input (in real format).
TEST(FastFourierTransformTest, FftReconstructConjsym) {
  auto fft_manager = absl::make_unique<FftManager>(k65Samples.NumElements());
  auto fft_forward = FastFourierTransform::Forward1d(fft_manager, k65Samples);
  auto fft_inverse =
      FastFourierTransform::Inverse1dConjSym(fft_manager, fft_forward);

  std::string fail_msg;
  ASSERT_TRUE(
      CompareDoubleMatrix(k65Samples, fft_inverse, kTolerance, &fail_msg))
      << fail_msg;
}

// Test that the fft (both forward and inverse) will correctly handle the case
// where the input is all zeros.
TEST(FastFourierTransformTest, FftReconstructConjsymZero) {
  auto fft_manager =
      absl::make_unique<FftManager>(k65SamplesZero.NumElements());
  auto fft_forward =
      FastFourierTransform::Forward1d(fft_manager, k65SamplesZero);
  auto fft_inverse =
      FastFourierTransform::Inverse1dConjSym(fft_manager, fft_forward);

  std::string fail_msg;
  ASSERT_TRUE(
      CompareDoubleMatrix(k65SamplesZero, fft_inverse, kTolerance, &fail_msg))
      << fail_msg;
}

}  // namespace
}  // namespace Visqol
