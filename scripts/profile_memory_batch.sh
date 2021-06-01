#!/usr/bin/env bash

echo "reference,degraded" > ./tmp_batch.csv
echo "./testdata/conformance_testdata_subset/guitar48_stereo.wav,./testdata/conformance_testdata_subset/guitar48_stereo_64kbps_aac.wav" >> ./tmp_batch.csv
echo "./testdata/conformance_testdata_subset/contrabassoon48_stereo.wav,./testdata/conformance_testdata_subset/contrabassoon48_stereo_24kbps_aac.wav" >> ./tmp_batch.csv
echo "./testdata/conformance_testdata_subset/glock48_stereo.wav,./testdata/conformance_testdata_subset/glock48_stereo_48kbps_aac.wav" >> ./tmp_batch.csv
echo "./testdata/conformance_testdata_subset/harpsichord48_stereo.wav,./testdata/conformance_testdata_subset/harpsichord48_stereo_96kbps_mp3.wav" >> ./tmp_batch.csv
echo "./testdata/conformance_testdata_subset/moonlight48_stereo.wav,./testdata/conformance_testdata_subset/moonlight48_stereo_128kbps_aac.wav" >> ./tmp_batch.csv

valgrind --tool=massif -v ./bazel-bin/visqol --batch_input_csv "./tmp_batch.csv" --results_csv "./tmp_results.csv" --output_debug "./tmp_debug.csv"

for f in ./massif.out.*; do
    ms_print $f > ./"mem-profile.txt"
done

rm ./massif.out.*
rm ./tmp_*.csv