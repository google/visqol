"""Tests for visqol.python.visqol_lib_py."""

import os

import numpy as np

# Placeholder for resource import.
import unittest
from visqol.python import visqol_lib_py
from concurrent import futures
from absl import logging

MODEL_FILE = 'model/libsvm_nu_svr_model.txt'
REF_FILE = 'testdata/clean_speech/CA01_01.wav'
DEG_FILE = 'testdata/clean_speech/transcoded_CA01_01.wav'

CONFORMANCE_TOLERANCE = 0.0001


def _CalculateVisqol(reference_file, degraded_file):

  files_dir = os.path.dirname(__file__)

  model_path = visqol_lib_py.FilePath(os.path.join(files_dir, MODEL_FILE))
  ref_path = visqol_lib_py.FilePath(os.path.join(files_dir, reference_file))
  deg_path = visqol_lib_py.FilePath(os.path.join(files_dir, degraded_file))
  manager = visqol_lib_py.VisqolManager()
  manager.Init(model_path, True, False, 60)
  similarity_result = manager.Run(ref_path, deg_path)
  return similarity_result


def _log_wrapper(method, args, kwargs):
  try:
    return_value = method(*args, **kwargs)
  except:
    logging.exception('Exception in %s', method)
    raise
  return return_value


class VisqolLibPyTest(unittest.TestCase):

  def test_Parallel(self):
    methods_and_args = []
    methods_and_args.append((_CalculateVisqol, [REF_FILE, DEG_FILE]))
    methods_and_args.append((_CalculateVisqol, [REF_FILE, DEG_FILE]))
    results = []
    with futures.ThreadPoolExecutor(
        max_workers=len(methods_and_args)) as executor:
      started_futures = []
      # This is pretty much a map() but we have better control over the chunk
      # size.
      for method, args in methods_and_args:
        future = executor.submit(_log_wrapper, method, args, {})
        started_futures.append((future, method, args))
      last_exception = None
      for f, method, args in started_futures:
        try:
          results.append(f.result())
        except Exception as ex:  # pylint: disable=broad-except
          e = ex  # This assigment works around pytype bug http://b/136279340.
          msg = (
              'Execution of method %s with args %s in a separate thread failed.'
              % (method, args))
          msg += " The last failed method's exception will be re-thrown."
          last_exception = e
          logging.exception(msg)
      if last_exception:
        raise last_exception  # Can only be Exception pylint: disable=raising-bad-type
      return results

  def test_docstring(self):
    self.assertContainsInOrder(['ViSQOL'], visqol_lib_py.__doc__)

  def test_VisqolManager(self):
    similarity_result = _CalculateVisqol(REF_FILE, DEG_FILE)

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
