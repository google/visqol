"""Tests for visqol.python.visqol_lib_py."""

import os

import numpy as np

# Placeholder for resource import.
import unittest
from visqol.python import visqol_lib_py

MODEL_FILE = 'model/libsvm_nu_svr_model.txt'
REF_FILE = 'testdata/clean_speech/CA01_01.wav'
DEG_FILE = 'testdata/clean_speech/transcoded_CA01_01.wav'

CONFORMANCE_TOLERANCE = 0.0001

class VisqolLibPyTest(unittest.TestCase):

  def test_docstring(self):
    self.assertContainsInOrder(['ViSQOL'], visqol_lib_py.__doc__)

  def test_VisqolManager(self):
    files_dir = os.path.dirname(__file__)

    model_path = visqol_lib_py.FilePath(os.path.join(files_dir, MODEL_FILE))
    ref_path = visqol_lib_py.FilePath(os.path.join(files_dir, REF_FILE))
    deg_path = visqol_lib_py.FilePath(os.path.join(files_dir, DEG_FILE))

    manager = visqol_lib_py.VisqolManager()
    manager.Init(model_path, True, False)

    similarity_result = manager.Run(ref_path, deg_path)

    # The conformance value lives in a c++ header.
    conformance_value = visqol_lib_py.ConformanceSpeechCA01TranscodedValue()
    self.assertGreaterEqual(conformance_value, 1.0)
    self.assertLessEqual(conformance_value, 5.0)

    self.assertGreater(similarity_result.moslqo,
                       conformance_value - CONFORMANCE_TOLERANCE)
    self.assertLess(similarity_result.moslqo,
                    conformance_value + CONFORMANCE_TOLERANCE)

  def test_VisqolApi(self):
    files_dir = os.path.dirname(__file__)

    model_path = os.path.join(files_dir, MODEL_FILE)

    config = visqol_lib_py.MakeVisqolConfig()
    config.audio.sample_rate = 48000
    config.options.svr_model_path = model_path

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    # Create an artificial signal.
    sin_ref = np.sin(np.linspace(0., 1000, 48000 * 2))
    noisy_deg = sin_ref + np.random.normal(size=sin_ref.shape)

    similarity_result = api.Measure(sin_ref, noisy_deg)

    self.assertGreaterEqual(similarity_result.moslqo, 1.0)
    self.assertLess(similarity_result.moslqo, 5.0)


if __name__ == '__main__':
  unittest.main()
