"""Create a script to update the conformance scores.

This assumes that the current version of ViSQOL is reasonable, and uses
it's output on the conformance test set to generate new scores.  Since
the process of doing this manually is tedious, this .py script runs ViSQOL
to create sed commands to update the scores.

Inspect the generated sed commands and run it if it looks reasonable to update
the scores.  After doing this, conformance_test should pass.
"""

import dataclasses
import os
from typing import Sequence

from absl import app

from google3.pyglib import resources

from google3.third_party.visqol.python import visqol_lib_py
from google3.third_party.visqol.src.proto import similarity_result_pb2

# The search window radius is constant through the test cases.
SEARCH_WINDOW_RADIUS = 60
SVR_MODEL_FILE = 'google3/third_party/visqol/model/libsvm_nu_svr_model.txt'


@dataclasses.dataclass
class ConformanceCase:
  reference_file: str
  degraded_file: str
  use_speech_mode: bool
  constant_name: str
  use_lattice_model: bool
  use_unscaled_speech_mos_mapping: bool = False


CASES = [
    ConformanceCase(
        'third_party/visqol/testdata/clean_speech/CA01_01.wav',
        'third_party/visqol/testdata/clean_speech/transcoded_CA01_01.wav', True,
        'kConformanceSpeechCA01TranscodedLattice', True),
    ConformanceCase(
        'third_party/visqol/testdata/clean_speech/CA01_01.wav',
        'third_party/visqol/testdata/clean_speech/transcoded_CA01_01.wav', True,
        'kConformanceSpeechCA01TranscodedExponential', False),
    ConformanceCase('third_party/visqol/testdata/clean_speech/CA01_01.wav',
                    'third_party/visqol/testdata/clean_speech/CA01_01.wav',
                    True, 'kConformanceCA01PerfectScoreLattice', True),
    ConformanceCase('third_party/visqol/testdata/clean_speech/CA01_01.wav',
                    'third_party/visqol/testdata/clean_speech/CA01_01.wav',
                    True, 'kConformanceUnscaledPerfectScoreExponential', False,
                    True),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'strauss48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'strauss48_stereo_lp35.wav', False, 'kConformanceStraussLp35', False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'steely48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'steely48_stereo_lp7.wav', False, 'kConformanceSteelyLp7', False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'sopr48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'sopr48_stereo_256kbps_aac.wav', False, 'kConformanceSopr256aac',
        False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'ravel48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'ravel48_stereo_128kbps_opus.wav', False, 'kConformanceRavel128opus',
        False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'moonlight48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'moonlight48_stereo_128kbps_aac.wav', False,
        'kConformanceMoonlight128aac', False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'harpsichord48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'harpsichord48_stereo_96kbps_mp3.wav', False,
        'kConformanceHarpsichord96mp3', False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'guitar48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'guitar48_stereo_64kbps_aac.wav', False, 'kConformanceGuitar64aac',
        False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'glock48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'glock48_stereo_48kbps_aac.wav', False, 'kConformanceGlock48aac',
        False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'contrabassoon48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'contrabassoon48_stereo_24kbps_aac.wav', False,
        'kConformanceContrabassoon24aac', False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'castanets48_stereo.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'castanets48_stereo.wav', False, 'kConformanceCastanetsIdentity',
        False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'guitar48_stereo.wav',
        'third_party/visqol/testdata/short_duration/5_second/'
        'guitar48_stereo_5_sec.wav', False,
        'kConformanceGuitarShortDegradedPatch', False),
    ConformanceCase(
        'third_party/visqol/testdata/short_duration/5_second/'
        'guitar48_stereo_5_sec.wav',
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'guitar48_stereo.wav', False, 'kConformanceGuitarShortReferencePatch',
        False),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'guitar48_stereo.wav',
        'third_party/visqol/testdata/clean_speech/CA01_01.wav', True,
        'kConformanceDifferentAudiosLattice', True),
    ConformanceCase(
        'third_party/visqol/testdata/conformance_testdata_subset/'
        'guitar48_stereo.wav',
        'third_party/visqol/testdata/clean_speech/CA01_01.wav', True,
        'kConformanceDifferentAudiosExponential', False),
    ConformanceCase('third_party/visqol/testdata/alignment/reference.wav',
                    'third_party/visqol/testdata/alignment/degraded.wav', True,
                    'kConformanceBadDegradedLattice', True),
    ConformanceCase('third_party/visqol/testdata/alignment/reference.wav',
                    'third_party/visqol/testdata/alignment/degraded.wav', True,
                    'kConformanceBadDegradedExponential', False)
]


def RunVisqolWithSettings(
    reference_file: str, degraded_file: str, svr_model_file: str,
    use_speech_mode: bool, use_unscaled_speech_mos_mapping: bool,
    search_window_radius: int,
    use_lattice_model: bool) -> similarity_result_pb2.SimilarityResultMsg:
  """Runs Visqol with settings that match the conformance test."""
  files_dir = resources.GetARootDirWithAllResources()
  model_path = visqol_lib_py.FilePath(os.path.join(files_dir, svr_model_file))

  # The ConformanceCase file relative paths are missing a 'google3' to match
  # the cpp relative paths in conformance_test.cpp, so add it here.
  ref_path = visqol_lib_py.FilePath(
      os.path.join(files_dir, 'google3', reference_file))
  deg_path = visqol_lib_py.FilePath(
      os.path.join(files_dir, 'google3', degraded_file))
  # Initialize a fresh VisqolManager instead of reusing the previous one.
  manager = visqol_lib_py.VisqolManager()
  manager.Init(model_path, use_speech_mode, use_unscaled_speech_mos_mapping,
               search_window_radius, use_lattice_model)
  similarity_result = manager.Run(ref_path, deg_path)
  return similarity_result


def RunCases(cases: Sequence[ConformanceCase]) -> None:
  for case in cases:
    result = RunVisqolWithSettings(case.reference_file, case.degraded_file,
                                   SVR_MODEL_FILE, case.use_speech_mode,
                                   case.use_unscaled_speech_mos_mapping,
                                   SEARCH_WINDOW_RADIUS, case.use_lattice_model)
    score = result.moslqo
    # Print the sed command to update the score for this case.
    print('sed -i \'s/'
          f'#define {case.constant_name} ([0-5][.][0-9]*)/'
          f'#define {case.constant_name} ({score})/\' '
          'third_party/visqol/src/include/conformance.h')


def main(_):
  RunCases(CASES)


if __name__ == '__main__':
  app.run(main)
