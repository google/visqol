"""Tests for train_lattice."""

import numpy as np
import pandas as pd

from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized
from google3.third_party.visqol.scripts import train_lattice


class TrainLatticeTest(parameterized.TestCase):

  def generate_fake_data_frame(self, size):
    """Returns data in (features, labels) tuple."""
    data = {
        'fvnsim0': np.random.uniform(0, 1, size=(size)),
        'fvnsim10_0': np.random.uniform(0, 1, size=(size)),
        'fstdnsim0': np.random.uniform(0, 1, size=(size)),
        'fvdegenergy0': np.random.uniform(0, 1, size=(size)),
        'mos': np.random.uniform(1, 5, size=(size)),
    }
    return pd.DataFrame(data)

  @parameterized.named_parameters(
      ('linear', train_lattice.make_linear_estimator),
      ('random_ensemble', train_lattice.make_random_ensemble_estimator),
      ('tau_ensemble', train_lattice.make_tau_ensemble_estimator),
  )
  def test_make_linear_estimator(self, make_estimator_fn):
    config = {
        'use_output_calibration': False,
        'model_dir': None,
        'learning_rate': 0.001,
        'num_fvnsim_bands': 1,
        'num_keypoints': 10,
        'lattice_size': 2,
        'lattice_rank': 3,
        'num_lattices': 3,
        'target_fvnsim_per_lattice': 1,
        'target_features_per_lattice': 3,
    }
    data_frame = self.generate_fake_data_frame(size=14)
    mos = data_frame.pop('mos')
    train_input_fn = train_lattice.make_input_fn(
        data_frame, mos, 1, batch_size=7)
    estimator = make_estimator_fn(
        loss_fn=None,
        feature_analysis_input_fn=train_input_fn,
        feature_columns=train_lattice.make_feature_columns(config),
        feature_configs=train_lattice.make_feature_configs(config),
        config=config,
        config_updates=[])
    self.assertIsNotNone(estimator)

    estimator.train(train_input_fn)

    # Test with a different batch size.
    test_data_frame = self.generate_fake_data_frame(size=15)
    test_mos = test_data_frame.pop('mos')
    test_input_fn = train_lattice.make_input_fn(
        test_data_frame, test_mos, 1, batch_size=5)

    results = estimator.evaluate(test_input_fn)
    self.assertIsNotNone(results)


if __name__ == '__main__':
  googletest.main()
