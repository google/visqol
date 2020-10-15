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
// Major version change history:
// Feb 2020: v300 - polynomial fitting for speech mode, SVR for audio mode.
// Oct 2020: v310 - exponential fitting for speech mode, (audio mode unchanged).
#define kVisqolConformanceNumber (310)

// If the scores for these known files changes, tests/conformance_test will fail
// Whenever these constants need to be updated, kVisqolConformanceNumber should
// be incremented.
#define kConformanceSpeechCA01Transcoded (2.3691150188221259)

#define kConformanceStraussLp35 (1.9905729378864558)

#define kConformanceSteelyLp7 (2.4019387463496034)

#define kConformanceSopr256aac (4.7265073131023394)

#define kConformanceRavel128opus (4.6393500094950069)

#define kConformanceMoonlight128aac (4.6902760241035715)

#define kConformanceHarpsichord96mp3 (4.2902957875132159)

#define kConformanceGuitar64aac (4.5123244380958774)

#define kConformanceGuitarShortDegradedPatch (4.4056028414814765)

#define kConformanceGuitarShortReferencePatch (4.6761039922498471)

#define kConformanceDifferentAudios (1.5093694203696959)

#define kConformanceGlock48aac (4.333169489109431)

#define kConformanceContrabassoon24aac (4.0500790740957617)

#define kConformanceCastanetsIdentity (4.7321012530423481)

#define kConformanceBadDegraded (1.5260486187764308)

#define kConformanceGuitarLongDuration (4.5177618344298294)

#define kCA01_01UnscaledPerfectScore (4.1557613014690995)

#define kCA01_01AsAudio (2.0003927800390828)

#endif // VISQOL_INCLUDE_CONFORMANCE_H
