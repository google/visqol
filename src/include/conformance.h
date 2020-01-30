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

#ifndef VISQOL_INCLUDE_CONFORMANCE_H
#define VISQOL_INCLUDE_CONFORMANCE_H

// ViSQOL tracks the conformance of scores from visqol version to version, to
// allow academic comparison.
// This is done with a set of known files.
// If the score changes for these files due some algorithm or parameter
// modification, we must bump the conformance version number.
// Previous matlab versions were known as 238.
// The C++ version introduces fixes and starts at 250.
// As of 300, the first digit is the major version and last two digits are the
// minor version.
#define kVisqolConformanceNumber (300)

// If the scores for these known files changes, tests/conformance_test will fail
// Whenever these constants need to be updated, kVisqolConformanceNumber should
// be incremented.
#define kConformanceSpeechCA01Transcoded (2.472834)

#define kConformanceStraussLp35 (1.9905729378864558)

#define kConformanceSteelyLp7 (2.4019387463496034)

#define kConformanceSopr256aac (4.7265073131023394)

#define kConformanceRavel128opus (4.6393500094950069)

#define kConformanceMoonlight128aac (4.6902760241035715)

#define kConformanceHarpsichord96mp3 (4.2902957875132159)

#define kConformanceGuitar64aac (4.5123244380958774)

#define kConformanceGlock48aac (4.333169489109431)

#define kConformanceContrabassoon24aac (4.0500790740957617)

#define kConformanceCastanetsIdentity (4.7321012530423481)

#endif // VISQOL_INCLUDE_CONFORMANCE_H
