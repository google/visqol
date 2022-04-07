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
// Feb 2022: v333 - retrained lattice and exponential speech model for Hann
//                  window fix and quantile-based NSIM.
// Aug 2021: v320 - lattice model for speech mode, (audio mode unchanged).
// Oct 2020: v310 - exponential fitting for speech mode, (audio mode unchanged).
// Feb 2020: v300 - polynomial fitting for speech mode, SVR for audio mode.

#define kVisqolConformanceNumber (333)

// If the scores for these known files changes, tests/conformance_test will fail
// Whenever these constants need to be updated, kVisqolConformanceNumber should
// be incremented.
#define kConformanceSpeechCA01TranscodedLattice (3.3129234313964844)
#define kConformanceSpeechCA01TranscodedExponential (3.374505555111911)

#define kConformanceStraussLp35 (1.3888791489130758)

#define kConformanceSteelyLp7 (2.2501683734385183)

#define kConformanceSopr256aac (4.68228969737946)

#define kConformanceRavel128opus (4.465141897255348)

#define kConformanceMoonlight128aac (4.684292801646114)

#define kConformanceHarpsichord96mp3 (4.22374532766003)

#define kConformanceGuitar64aac (4.349722308064298)

#define kConformanceGuitarShortDegradedPatch (4.314508583690198)

#define kConformanceGuitarShortReferencePatch (4.550791119387646)

#define kConformanceDifferentAudiosLattice (1.4982070922851562)
#define kConformanceDifferentAudiosExponential (1.269675546824064)

#define kConformanceGlock48aac (4.332452943882108)

#define kConformanceContrabassoon24aac (2.346868205375293)

#define kConformanceCastanetsIdentity (4.732101253042348)

#define kConformanceBadDegradedLattice (1.19293212890625)
#define kConformanceBadDegradedExponential (1.357521678867611)

#define kConformanceCA01PerfectScoreLattice (4.505550384521484)
#define kConformanceUnscaledPerfectScoreExponential (4.015861169223797)

#define kCA01_01AsAudio (1.7658378752958486)

#endif  // VISQOL_INCLUDE_CONFORMANCE_H
