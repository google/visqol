# ViSQOL

ViSQOL (Virtual Speech Quality Objective Listener) is an objective, full-reference metric for perceived audio quality. It uses a spectro-temporal measure of similarity between a reference and a test speech signal to produce a MOS-LQO (Mean Opinion Score - Listening Quality Objective) score. MOS-LQO scores range from 1 (the worst) to 5 (the best).

## Table of Contents
- [Features](#features)
- [Build](#build)
- [Command Line Usage](#command-line-usage)
- [API Usage](#api-usage)
- [Dependencies](#dependencies)
- [License](#license)

## Features
ViSQOL can be run from the command line, or integrated into a project and used through its API. Whether being used from the command line, or used through the API, ViSQOL is capable of running in two modes:
1. #### Audio Mode:
- When running in audio mode, input signals must have a 48k sample rate. Non-48k sample rate inputs can be used, but the comparison will likely be negatively impacted.
- Input signals can be multi-channel, but they will be down-mixed to mono for performing the comparison.
- Audio mode uses support vector regression, with the maximum range at ~4.75.
2. #### Speech Mode:
- When running in speech mode, ViSQOL uses a wideband model. It therefore expects input sample rates of 16k. Higher sample rate input can be used, but frequencies above 8kHz will not be included in the comparison.
- As part of the speech mode processing, a root mean square implementation for voice activity detection is performed on the reference signal to determine what parts of the signal have voice activity and should therefore be included in the comparison. The signal is normalized before performing the voice activity detection.
- Input signals can be multi-channel, but they will be down-mixed to mono for performing the comparison.
- Speech mode is scaled to have a maximum MOS of 5.0 to match previous version behavior.


## Build

#### Linux Build Instructions
1. ##### Install Bazel
- Bazel can be install for Linux from [here](https://docs.bazel.build/versions/master/install-ubuntu.html).
- Tested with Bazel version 0.22.0.
2. ##### Install Boost
- Boost can be installed on Linux with the following command: `sudo apt-get install libboost-all-dev`
- Tested with Boost version 1.65.0.
3. ##### Build ViSQOL
- Change directory to the root of the ViSQOL project (i.e. where the WORKSPACE file is) and run the following command: `bazel build :visqol -c opt`

#### Windows Build Instructions (Experimental, last Tested on Windows 10 x64, 2019 March)

1. ##### Install Visual Studio
- Bazel requires Visual Studio 2015 or later. See further details [here](https://docs.bazel.build/versions/master/windows.html#build-c).
- Install Visual Studio from [here](https://visualstudio.microsoft.com/vs/community/). Ensure that the C++ build tools are installed and the Windows SDK e.g. the 'Desktop development with C++' option in the VS installer.

2. ##### Install Bazel
- Bazel can be install for Windows from [here](https://docs.bazel.build/versions/master/windows.html).
- Tested with Bazel version 0.22.0.

3. ##### Install Boost
- Tested with version 1.60.0.
- Download the Windows version from [here](https://www.boost.org/users/history/version_1_60_0.html).
- Extract the contents of the download to the `C:` directory, and rename the extracted folder to remove any version info. You should now have the following directory structure: `C:\boost`
- Navigate to this directory and build the Boost binaries for x64. For version 1.60.0, these instructions can be found [here](https://www.boost.org/doc/libs/1_60_0/more/getting_started/windows.html#prepare-to-use-a-boost-library-binary).


4. ##### Build ViSQOL:
- Change directory to the root of the ViSQOL project (i.e. where the WORKSPACE file is) and run the following command: `bazel build :visqol -c opt`

## Command Line Usage
#### Note Regarding Usage
- When run from the command line, input signals must be in WAV format.


#### Flags

`--reference_file`

- The 48k sample rate WAV file used as the reference audio.

`--degraded_file`

- The 48k sample rate WAV file that will be compared to the reference audio.

`--batch_input_csv`

- Used to specify a path to a CSV file with the format:

  reference,degraded
  ref1.wav,deg1.wav
  ref2.wav,deg2.wav

- If the `batch_input_csv` flag is used, the `reference_file` and `degraded_file` flags will be ignored.

`--results_csv`

- Used to specify a path that the similarity score results will be output to. This will be a CSV file with the format:

  reference,degraded,moslqo
  ref1.wav,deg1.wav,3.4
  ref2.wav,deg2.wav,4.1

`--verbose`

- The reference file path, degraded file path and the MOS-LQO values will be output to the console after the MOS-LQO has been calculated, along with similarity scores on a per-patch and per-frequency band basis.

`--output_debug`

- Used to specify a file path where output debug information will be written to. This debug info contains the full details of the comparison between the reference and degraded audio signals and is in JSON format. The file does not need to previously exist. Contents will be appended to the file if it does already exist or if ViSQOL is run in batch mode.

`--similarity_to_quality_model`

- The libsvm model to use during comparison. Use this only if you want to explicitly specify the model file location, otherwise the default model will be used.

`--use_speech_mode`
- Use a wideband model (sensitive up to 8kHz) with voice activity detection that normalizes the polynomial NSIM->MOS mapping so that a perfect NSIM score of 1.0 translates to 5.0.

`--use_unscaled_speech_mos_mapping`
- When used in conjunction with --use_speech_mode, this flag will prevent a perfect NSIM score of 1.0 being translated to a MOS score of 5.0. Perfect NSIM scores will instead result in MOS scores of ~4.x.

#### Example Command Line Usage

  To compare two files and output their similarity to the console:

##### Linux:
- `./bazel-bin/visqol --reference_file ref1.wav --degraded_file deg1.wav --verbose`

##### Windows:
- `bazel-bin\visqol.exe --reference_file "ref1.wav" --degraded_file "deg1.wav" --verbose`

---

To compare all reference-degraded file pairs in a CSV file, outputting the
results to another file and also outputting additional "debug" information:

##### Linux:
- `./bazel-bin/visqol --batch_input_csv input.csv --results_csv results.csv
    --output_debug debug.json`

##### Windows:
- `bazel-bin\visqol.exe --batch_input_csv "input.csv" --results_csv "results.csv" --output_debug "debug.json"`

---

To compare two files using scaled speech mode and output their similarity to the console:
##### Linux:
- `./bazel-bin/visqol --reference_file ref1.wav --degraded_file deg1.wav --use_speech_mode --verbose`

##### Windows:
- `bazel-bin\visqol.exe --reference_file "ref1.wav" --degraded_file "deg1.wav" --use_speech_mode --verbose`

---

To compare two files using unscaled speech mode and output their similarity to the console:
##### Linux:
- `./bazel-bin/visqol --reference_file ref1.wav --degraded_file deg1.wav --use_speech_mode --use_unscaled_speech_mos_mapping --verbose`

##### Windows:
- `bazel-bin\visqol.exe --reference_file "ref1.wav" --degraded_file "deg1.wav" --use_speech_mode --use_unscaled_speech_mos_mapping --verbose`

## API Usage
#### ViSQOL Integration
To integrate ViSQOL with your Bazel project:
1. Add ViSQOL to your WORKSPACE file as a local_repository:
	```
	local_repository (
	    name = "visqol",
	    path = "/path/to/visqol",
	)
	```
2. Then in your project's BUILD file, add the ViSQOL library as a dependency to your binary/library dependency list:
	```
    deps = ["@visqol//:visqol_lib"],
	```
3. Note that Bazel does not currently resolve transitive dependencies (see [issue #2391](https://github.com/bazelbuild/bazel/issues/2391)). As a workaround, it is required that you copy the contents of the ViSQOL WORKSPACE file to your own project's WORKSPACE file until this is resolved.

#### Sample Program
```cpp
int main(int argc, char **argv) {

  // Create an instance of the ViSQOL API configuration class.
  Visqol::VisqolConfig config;

  // Set the sample rate of the signals that are to be compared.
  // Both signals must have the same sample rate.
  config.mutable_audio()->set_sample_rate(48000);

  // When running in audio mode, sample rates of 48k is recommended for the input signals.
  // Using non-48k input will very likely negatively affect the comparison result.
  // If, however, API users wish to run with non-48k input, set this to true.
  config.mutable_options()->set_allow_unsupported_sample_rates(false);

  // Optionally, set the location of the model file to use.
  // If not set, the default model file will be used.
  config.mutable_options()->set_svr_model_path("visqol/model/libsvm_nu_svr_model.txt");

  // ViSQOL will run in audio mode comparison by default.
  // If speech mode comparison is desired, set to true.
  config.mutable_options()->set_use_speech_scoring(false);

  // Speech mode will scale the MOS mapping by default. This means that a
  // perfect NSIM score of 1.0 will be mapped to a perfect MOS-LQO of 5.0.
  // Set to true to use unscaled speech mode. This means that a perfect
  // NSIM score will instead be mapped to a MOS-LQO of ~4.x.
  config.mutable_options()->set_use_unscaled_speech_mos_mapping(false);

  // Create an instance of the ViSQOL API. A new instance
  // is required for each signal pair to be compared.
  Visqol::VisqolApi visqol;
  util::Status status = visqol.Create(config);

  // Ensure that the creation succeeded.
  if (!status.ok()) {
    std::cout<<status.ToString()<<std::endl;
    return -1;
  }

  // Perform the comparison.
  google::protobuf::util::StatusOr<Visqol::SimilarityResultMsg> comparison_status_or =
          visqol.Measure(reference_signal, degraded_signal);

  // Ensure that the comparison succeeded.
  if (!comparison_status_or.ok()) {
    std::cout<<comparison_status_or.status().ToString()<<std::endl;
    return -1;
  }

  // Extract the comparison result from the StatusOr.
  Visqol::SimilarityResultMsg similarity_result = comparison_status_or.ValueOrDie();

  // Get the "Mean Opinion Score - Listening Quality Objective" for the degraded
  // signal, following the comparison to the reference signal.
  double moslqo = similarity_result.moslqo();

  // Get the similarity results for each frequency band.
  google::protobuf::RepeatedField<double> fvnsim = similarity_result.fvnsim();

  // Get the center frequency bands that the above FVNSIM results correspond to.
  google::protobuf::RepeatedField<double> cfb = similarity_result.center_freq_bands();

  // Get the mean of the FVNSIM values (the VNSIM).
  double vnsim = similarity_result.vnsim();

  // Get the comparison results for each patch that was compared.
  google::protobuf::RepeatedPtrField<Visqol::SimilarityResultMsg_PatchSimilarityMsg> patch_sims =
          similarity_result.patch_sims();

  for (Visqol::SimilarityResultMsg_PatchSimilarityMsg each_patch : patch_sims) {
    // Get the similarity score for this patch.
    double patch_similarity = each_patch.similarity();

    // Get the similarity results for each frequency band for this patch.
    // The center frequencies that these values correspond to are the
    // same as those that are returned in the parent center_freq_bands().
    google::protobuf::RepeatedField<double> patch_fvnsim = each_patch.freq_band_means();

    // Get the time (in sec) where this patch starts in the reference signal.
    double ref_patch_start_time = each_patch.ref_patch_start_time();

    // Get the time (in sec) where this patch ends in the reference signal.
    double ref_patch_end_time = each_patch.ref_patch_end_time();

    // Get the time (in sec) where this patch starts in the degraded signal.
    double deg_patch_start_time = each_patch.deg_patch_start_time();

    // Get the time (in sec) where this patch ends in the degraded signal.
    double deg_patch_end_time = each_patch.deg_patch_end_time();
  }

  return 0;
}
```
## Dependencies

Armadillo - http://arma.sourceforge.net/

Libsvm - http://www.csie.ntu.edu.tw/~cjlin/libsvm/

PFFFT - https://bitbucket.org/jpommier/pffft

Boost - https://www.boost.org/

## Support Vector Regression Model Training

Using the libsvm codebase, you can train a model specific to your data.
The procedure is as follows:
1. Gather audio file pairs in 48kHz (for audio mode) with subjective test scores.
2. Create 2 CSV files, one that lists the file pairs to be compared according to --batch_input_csv, and one that has the MOS-LQS (mean subjective scores) that correspond to the same rows in the batch csv file under a 'moslqs' column.
3. Modify src/include/sim_results_writer.h to output_fvnsim=true and output_moslqo=false
4. Run ViSQOLAudio in batch mode, using --batch_input_csv and --output_csv
5. Run scripts:make_svm_train_file on myvisqoloutput.csv
6. Run a grid search to find the SVM parameters.  See the docs in scripts/make_svm_train_file.py for help with that.
7. This model can be passed into ViSQOL in audio mode using --similarity_to_quality_model

Currently, SVR is only supported for audio mode.

## License

Use of this source code is governed by a Apache v2.0 license that can be found in the LICENSE file.

## Papers

There have been several papers that describe the design of the ViSQOL algorithm and compare it to other metrics.
These three should serve as an overview:

[ViSQOL: The virtual speech quality objective listener](https://arrow.dit.ie/cgi/viewcontent.cgi?article=1168&context=scschcomcon) (IWAENC 2012)
[Robustness of speech quality metrics to background noise and network degradations: Comparing ViSQOL, PESQ and POLQA](https://arrow.dit.ie/cgi/viewcontent.cgi?article=1160&context=scschcomcon) (ICASSP 2013)
[ViSQOLAudio: An objective audio quality metric for low bitrate codecs](https://asa.scitation.org/doi/full/10.1121/1.4921674) (Journal of the Acoustic Society of America, 2015)

## FAQ

### Why do I get compile error about undeclared inclusion(s) in rule '//:visqol_lib'?
This may have to do with bazel being out of sync.  You may need to run `bazel clean --expunge` and rebuild.

### Why are the MOS predictions on my files so bad?
There are a number of possible explanations, here are the most common ones:
- In audio mode, ViSQOL was trained with a clean reference and degraded files full-band (audio containing frequencies up to 24 kHz) with bit rates as low as 24 kbps.  If the degraded audio is lower than this it may behave poorly.  If you have subjective scores, you might consider training your own model, as can be seen in scripts/make_svm_train_file.py.
- Another explanation is that too much silence is being analyzed.  We recommend 3 to 10 seconds of audio (typically 5 seconds) that has significant activity in the reference audio.

## Acknowledgement

In addition to the contributions visible on the repository history, Colm Sloan and Feargus O'Gorman have significantly contributed to the codebase in the collaboration between Andrew Hines and Google.
