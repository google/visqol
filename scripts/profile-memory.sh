#!/usr/bin/env bash

valgrind --tool=massif -v ./bazel-bin/visqol --reference_file ./testdata/conformance_testdata_subset/guitar48_stereo.wav --degraded_file ./testdata/conformance_testdata_subset/guitar48_stereo_64kbps_aac.wav

for f in ./massif.out.*; do
    ms_print $f > ./"mem-profile.txt"
done

rm ./massif.out.*