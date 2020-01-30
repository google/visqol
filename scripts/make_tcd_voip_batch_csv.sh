#!/bin/sh
# Copyright 2019 Google LLC, Andrew Hines
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Produces a CSV file for the TCD-VOIP dataset.
# This CSV can be used as batch input for ViSQOL.
# It will have the complete path/filename for  Reference,Degraded
# 
# Instructions:
# 1. Download the TCD-VOIP dataset.  Look online for instructions on how to do this.
# 2. Modify "Test Set" directory to "Test_Set" since things barf on spaces
# 3. Place this script in the root directory of the TCD-VOIP dataset,
#    alongside the "Test_set" directory.
# 4. run the script and pipe the output to myvalues.csv
#
# Ref paths are always in /ref/ subdir of the degraded category
ref_paths="$(find "$(pwd -P)/Test_Set" -name "*.wav"| grep /ref/)"

# Degraded paths don't have the /ref subdir
found_degraded_paths="$(find "Test_Set" -name "*.wav"| grep -v /ref/)"

# Compute the degraded path by replacing the '/ref/R_' prefix with '/C_', for example:
# Test Set/echo/ref/R_14_ECHO_FA.wav -> Test Set/echo/C_14_ECHO_FA.wav
computed_degraded_paths="$(echo ${found_degraded_paths}| sed -e 's/\/ref\/R_/\/C_/')"

echo reference, degraded

for path in ${ref_paths};
do
  echo "${path}","$(echo ${path}| sed -e 's/\/ref\/R_/\/C_/')";
done
