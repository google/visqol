# Copyright 2021 Google LLC
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

# Lint as: python3
r"""Trains a deep lattice network model that maps NSIM->MOS.

For more info on deep lattice networks see these resources:
- Tensorflow lattice docs: https://www.tensorflow.org/lattice/overview
- "Deep Lattice Networks and Partial Monotonic Functions" (You et al., 2017)
  https://arxiv.org/abs/1709.06680

This model has a few interesting properties:
- Calibrated quantile estimates allow modelling uncertainty. More info in
  "Regularization Strategies for Quantile Regression" (Narayan et al., 2021)
   https://arxiv.org/abs/2102.05135
- Monotonic priors on NSIM (We expect quality to rise as similarity rises)

Example usage:
blaze run -c opt  third_party/visqol/scripts:train_lattice.par -- \
--mos_csv=/cns/pf-d/home/cmaudio/mchinen/visqol/merged_tcdvoip_genspeech_psupp23_subjective_mos.csv
\
--fvnsim_csv=/cns/pf-d/home/cmaudio/mchinen/visqol/merged_tcdvoip_genspeech_psupp23_stdengframe.csv
\
--alsologtostderr \
--model=tau_ensemble \
--num_epochs=140 \
--lattice_size=3 \
--batch_size=100 \
--prefitting_num_epochs=10 \
--learning_rate=.005 \
--num_keypoints=20 \
--use_output_calibration=true \
--num_lattices=40 \
--lattice_rank=16
"""

import csv
import os
import random
import re
from typing import Any, Callable, Dict, List, Sequence, Type, Tuple

from absl import app
from absl import flags
from absl import logging

import attr
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import estimator as tf_estimator
from tensorflow import feature_column as fc
from tensorflow.compat.v1 import estimator as tf_compat_v1_estimator
from tensorflow_lattice import configs
from tensorflow_lattice import estimators

from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string('mos_csv', None,
                    'Path to a csv with degraded, subjective MOS')
flags.DEFINE_string('fvnsim_csv', None,
                    'Path to a csv with reference, degraded, nsim')
flags.DEFINE_integer('num_epochs', 5000, 'Number of training epoch.')
flags.DEFINE_integer('prefitting_num_epochs', 40, 'Prefitting epochs.')

flags.DEFINE_string(
    'model_dir', None, 'Path where the model should be saved.  If `None` use a'
    'temporary directory.')

flags.DEFINE_list(
    'config_updates', '',
    'Comma separated list of updates to model configs in name=value format.'
    'See tfl.configs.apply_updates().')
flags.DEFINE_integer(
    'num_fvnsim_bands', 21, 'Number of NSIM frequency bands '
    'from ViSQOL.  This should match what is in `fvnsim_csv`')
# Hyperparams
flags.DEFINE_float('train_set_ratio', 0.8, 'Ratio of data to use for training.')
flags.DEFINE_float('learning_rate', 0.005, 'Learning rate.')
flags.DEFINE_integer('batch_size', 300, 'Batch size.')
flags.DEFINE_integer('num_lattices', 15, 'Number of lattice for ensembles.')
flags.DEFINE_integer('lattice_rank', 5, 'Maximum number of features in each '
                     'lattice.')
flags.DEFINE_integer('num_keypoints', 5, 'Number of keypoints per lattice '
                     'feature.')
flags.DEFINE_integer(
    'lattice_size', 3, 'Lattice size for each FVNSIM feature. '
    'This is the number of learned interpolation points each feature has. '
    'Other features such as fvstddev or tau use a lattice size '
    'of 2.')
flags.DEFINE_integer(
    'target_fvnsim_per_lattice', 4,
    'For random lattice construction, target approximately '
    'this many fvnsim features in each lattice')
flags.DEFINE_integer(
    'target_features_per_lattice', 11,
    'For random lattice construction, target approximately '
    'this many total features in each lattice (including '
    'fvnsim).')

flags.DEFINE_boolean('use_output_calibration', True, 'Use output calibration.')
flags.DEFINE_enum('model', 'linear',
                  ['linear', 'random_ensemble', 'tau_ensemble'],
                  'The deep lattice netowrk type.')

# This seed can be used at inference to recreate data splitting/shuffling.
# This is used to train on the same training data for alternative mappings.
RANDOM_SEED = 6284968128


@attr.s(auto_attribs=True)
class FeaturesAndMos:
  features: np.ndarray
  mos: float


def load_fvnsim(fvnsim_csv: str,
                num_fvnsim_bands: int) -> Dict[str, np.ndarray]:
  """Returns dictionary with degraded file key and mean nsim value.

  The CSV should have several columns: reference path, degraded path,
  followed by an optional mos column and multiple fvnsim values (one per
  band).

  Args:
    fvnsim_csv: Path to CSV file with NSIM values, format described above.
    num_fvnsim_bands: The number of fvnsim bands to expect in CSV.

  Returns:
    Dictionary with degraded file key and NSIM value.
  """
  nsim_dict = {}

  with gfile.Open(fvnsim_csv, 'r') as csvfile:
    nsim_reader = csv.DictReader(csvfile)
    for row in nsim_reader:
      # Keep the degraded file without the directory info as key.
      # This will match what the mos dictionary has.
      degraded = row['degraded']
      if 'Speech_Databases/Genspeech' in degraded:
        # Genspeech basenames are not unique, but the last four directories
        # combined with them are.
        deg_file = '_'.join(degraded.split('/')[-4:])
      else:
        deg_file = os.path.basename(degraded)

      feature_names = get_feature_names(num_fvnsim_bands)
      features = np.array([row[feature_name] for feature_name in feature_names],
                          dtype=np.float32)
      nsim_dict[deg_file] = features

  return nsim_dict


def load_mos(mos_csv: str) -> Dict[str, float]:
  """Load a csv file with subjective MOS scores.

  Args:
    mos_csv: Path to a csv file that has `Filename` and `MOS` columns.

  Returns:
    Dictionary with filename keys and MOS values
  """
  mos_dict = {}
  with gfile.Open(mos_csv, 'r') as csvfile:
    mos_reader = csv.DictReader(csvfile)
    for row in mos_reader:
      base_name = os.path.basename(row['Filename'])
      mos_dict[base_name] = float(row['MOS'])
  return mos_dict


def get_feature_names(num_fvnsim_bands: int) -> List[str]:
  """Return a list of all feature names, such as fvnsim0."""
  all_names = []
  for idx in range(num_fvnsim_bands):
    all_names.append('fvnsim{}'.format(idx))
    all_names.append('fvnsim10_{}'.format(idx))
    all_names.append('fstdnsim{}'.format(idx))
    all_names.append('fvdegenergy{}'.format(idx))

  return all_names


def merge_fvnsim_and_mos(mos_csv: str, nsim_csv: str,
                         num_fvnsim_bands: int) -> Dict[str, FeaturesAndMos]:
  """Combines NSIM and MOS CSVs into a dictionary.

  Args:
    mos_csv: Path to csv file with MOS data.
    nsim_csv: Path to csv with NSIM data.
    num_fvnsim_bands: The number of fvnsim bands in the nsim CSV.

  Returns:
    Dictionary with degraded filename keys and FeaturesAndMos values.
  """

  # NSIM keys are full paths, wheras MOS keys are filenames
  merged_dict = {}

  mos_dict = load_mos(mos_csv)
  nsim_dict = load_fvnsim(nsim_csv, num_fvnsim_bands)

  for nsim_key, nsim_value in nsim_dict.items():
    # If the csv is invalid nsim values can be NaN, skip those.
    if np.isnan(nsim_value).any():
      logging.warn('skipping nan nsim key %s', nsim_key)
      continue

    if nsim_key not in mos_dict:
      logging.warn('skipping missing mos key %s', nsim_key)
      continue

    mos_value = mos_dict[nsim_key]
    merged_dict[nsim_key] = FeaturesAndMos(features=nsim_value, mos=mos_value)

  return merged_dict


def make_linear_estimator(
    loss_fn: Callable[..., Type[tf.Tensor]],
    feature_analysis_input_fn: Callable[[], Tuple[tf.Tensor, tf.Tensor]],
    feature_columns: Sequence[Any],
    feature_configs: Sequence[configs.FeatureConfig], config: Dict[str, Any],
    config_updates: List[Tuple[str, Any]]) -> Type[tf_estimator.Estimator]:
  """Make a simple linear model without a lattice.

  This is calibrated linear model with output calibration: Inputs are
  calibrated, linearly combined and the output of the linear layer is
  calibrated again using a PWL function.

  Args:
    loss_fn: A loss function to be used with tf.Estimator.
    feature_analysis_input_fn: an tf-style input_fn to calculate feature stats.
    feature_columns: List of tf.feature_columns.
    feature_configs: List of tf.lattice.configs.FeatureConfigs.
    config: Dictionary with hyperparameters as described in flags.
    config_updates: Updates for configuration, typically for tf-runner usage.

  Returns:
    A tf.Estimator object.
  """
  model_config = configs.CalibratedLinearConfig(
      feature_configs=feature_configs,
      use_bias=True,
      output_calibration=config['use_output_calibration'],
      output_min=1.,
      output_max=5.,
      regularizer_configs=[])
  # Update model configuration.
  # See tfl.configs.apply_updates for details.
  configs.apply_updates(model_config, config_updates)
  return estimators.CannedRegressor(
      model_dir=config['model_dir'],
      feature_columns=feature_columns,
      model_config=model_config,
      loss_fn=loss_fn,
      feature_analysis_input_fn=feature_analysis_input_fn,
      optimizer=tf.keras.optimizers.Adam(config['learning_rate']))


def make_random_ensemble_estimator(
    loss_fn: Callable[..., Type[tf.Tensor]],
    feature_analysis_input_fn: Callable[[], Tuple[tf.Tensor, tf.Tensor]],
    feature_columns: Sequence[Any],
    feature_configs: Sequence[configs.FeatureConfig], config: Dict[str, Any],
    config_updates: List[Tuple[str, Any]]) -> Type[tf_estimator.Estimator]:
  """This is a random lattice ensemble model with separate calibration.

  It has multiple lattices that have randomly selected features.  The
  model output is the average output of separately calibrated lattices.

  Args:
    loss_fn: A loss function to be used with tf.Estimator.
    feature_analysis_input_fn: an tf-style input_fn to calculate feature stats.
    feature_columns: Llist of tf.lattice.configs.FeatureConfigs.
    feature_configs: List of tf.feature_columns.
    config: Dictionary with hyperparameters as described in flags.
    config_updates: Updates for configuration, typically for tf-runner usage.

  Returns:
    A tf.Estimator object.
  """
  model_config = configs.CalibratedLatticeEnsembleConfig(
      feature_configs=feature_configs,
      output_min=1.,
      output_max=5.,
      output_calibration=config['use_output_calibration'],
      num_lattices=config['num_lattices'],
      lattice_rank=config['lattice_rank'],
      separate_calibrators=True,
      regularizer_configs=[])
  configs.apply_updates(model_config, config_updates)
  return estimators.CannedRegressor(
      model_dir=config['model_dir'],
      feature_columns=feature_columns,
      model_config=model_config,
      loss_fn=loss_fn,
      feature_analysis_input_fn=feature_analysis_input_fn,
      optimizer=tf.keras.optimizers.Adam(config['learning_rate']))


def make_tau_ensemble_estimator(
    loss_fn: Callable[..., Type[tf.Tensor]],
    feature_analysis_input_fn: Callable[[], Tuple[tf.Tensor, tf.Tensor]],
    feature_columns: Sequence[Any],
    feature_configs: Sequence[configs.FeatureConfig], config: Dict[str, Any],
    config_updates: List[Tuple[str, Any]]) -> Type[tf_estimator.Estimator]:
  """This is tau lattice ensemble model with separate calibration.

  It has multiple lattices that have randomly selected features.  It is very
  similar to the random ensemble model, but uses a variable number of features.
  Another major difference is that the 'tau' quantile feature is forced to be
  present in each lattice to allow for each lattice to be aware of quantile
  monotonicity.

  Args:
    loss_fn: A loss function to be used with tf.Estimator.
    feature_analysis_input_fn: an tf-style input_fn to calculate feature stats.
    feature_columns: Llist of tf.lattice.configs.FeatureConfigs.
    feature_configs: List of tf.feature_columns.
    config: Dictionary with hyperparameters as described in flags.
    config_updates: Updates for configuration, typically for tf-runner usage.

  Returns:
    A tf.Estimator object.
  """

  # Create random ensembles with tau in each lattice.
  def create_random_subensembles_under_limit(
      num_lattices: int, lattice_rank: int,
      param_limit: int) -> List[List[str]]:
    lattices = []
    used_names = set()
    all_names = get_feature_names(config['num_fvnsim_bands'])

    for _ in range(num_lattices):
      names = np.random.choice(all_names, size=lattice_rank - 1)
      # Add features until we surpass the parameter limit.
      params_used = 1
      names_under_limit = 0
      for name in names:
        if name.startswith('fvnsim'):
          # FVNSIM features can use a higher lattice size than 2.
          params_used *= config['lattice_size']
        else:
          # All other features use the minimum lattice size of 2.
          params_used *= 2
        if params_used > param_limit:
          break
        names_under_limit += 1
        used_names.add(name)
      names = names[:names_under_limit]
      lattices.append(list(names) + ['tau'])

    # Make a new lattice with the remainder to get the remaining lattice
    # for normal lattices only one or two params will live here.
    extra_lattice = []
    for name in all_names:
      if name not in used_names:
        extra_lattice.append(name)

    if extra_lattice:
      print('adding extra lattice with {}'.format(str(extra_lattice)))
      lattices.append(extra_lattice + ['tau'])
    return lattices

  # The number of parameters that a lattice uses varies because of the
  # different features having different lattice ranks.
  # In the simplest case, each feature has two points at the extreme values
  # over which linear interpolation is applied, and the total number of
  # parameters per lattice is 2**lattice_rank.
  # However, in practice we find fvnsim can benefit from using more points.
  # So add the desired number of fvnsim which have a lattice_size higher than
  # 2.
  target_non_fvnsim_features = (
      config['target_features_per_lattice'] -
      config['target_fvnsim_per_lattice'])
  parameter_limit = (
      (config['lattice_size']**config['target_fvnsim_per_lattice']) *
      (2**target_non_fvnsim_features))
  all_lattices = create_random_subensembles_under_limit(config['num_lattices'],
                                                        config['lattice_rank'],
                                                        parameter_limit)

  model_config = configs.CalibratedLatticeEnsembleConfig(
      output_min=1.,
      output_max=5.,
      output_calibration=config['use_output_calibration'],
      feature_configs=feature_configs,
      lattices=all_lattices,
      separate_calibrators=True,
      use_linear_combination=True,
      regularizer_configs=[
          configs.RegularizerConfig(name='calib_laplacian', l2=1e-3),
      ])
  configs.apply_updates(model_config, config_updates)
  return estimators.CannedRegressor(
      model_dir=config['model_dir'],
      feature_columns=feature_columns,
      model_config=model_config,
      loss_fn=loss_fn,
      feature_analysis_input_fn=feature_analysis_input_fn,
      optimizer=tf.keras.optimizers.Adam(config['learning_rate']))


def make_feature_columns(config: Dict[str, Any]) -> List[Any]:
  """Create a list of feature columns for tensorflow usage."""
  feature_columns = []
  # Construct the features and columns for the individual frequency bands.
  for idx in range(config['num_fvnsim_bands']):
    feature_columns.append(
        fc.numeric_column('fvnsim{}'.format(idx), default_value=0.7))
    feature_columns.append(
        fc.numeric_column('fvnsim10_{}'.format(idx), default_value=0.5))
    feature_columns.append(fc.numeric_column('fstdnsim{}'.format(idx)))
    feature_columns.append(fc.numeric_column('fvdegenergy{}'.format(idx)))

  # Add a single tau parameter to control the quantile.
  feature_columns += [fc.numeric_column('tau')]
  return feature_columns


def make_feature_dict(config: Dict[str, Any]) -> Dict[str, tf.Tensor]:
  """Create a list of feature columns for tensorflow usage."""
  feature_dict = {}
  # Construct the features and columns for the individual frequency bands.
  for idx in range(config['num_fvnsim_bands']):
    for base_name in ['fvnsim', 'fvnsim10_', 'fstdnsim', 'fvdegenergy']:
      name = f'{base_name}{idx}'
      feature_dict[name] = tf.constant(
          0.5, dtype=tf.float32, shape=(), name=name)

  # Add a single tau parameter to control the quantile.
  feature_dict['tau'] = tf.constant(0.5, dtype=tf.float32, shape=(), name='tau')
  return feature_dict


def make_feature_configs(config: Dict[str, Any]) -> List[configs.FeatureConfig]:
  """Create a list of FeatureConfigs for tf.lattice customization."""

  def make_feature_config(format_str: str,
                          idx: int,
                          lattice_size: int = 2,
                          monotonicity: str = 'none') -> configs.FeatureConfig:
    return configs.FeatureConfig(
        name=format_str.format(idx),
        lattice_size=lattice_size,
        # By default, input keypoints of pwl are quantiles of the feature.
        pwl_calibration_num_keypoints=config['num_keypoints'],
        monotonicity=monotonicity,
        # Add regularization to prevent large swings and overfitting.
        regularizer_configs=[
            configs.RegularizerConfig(name='calib_wrinkle', l2=1.0),
        ],
    )

  feature_configs = []
  # Construct the features and columns for the individual frequency bands.
  for idx in range(config['num_fvnsim_bands']):
    feature_configs.append(
        make_feature_config(
            'fvnsim{}',
            idx,
            monotonicity='increasing',
            lattice_size=config['lattice_size']))
    feature_configs.append(
        make_feature_config(
            'fvnsim10_{}',
            idx,
            monotonicity='increasing',
            lattice_size=config['lattice_size']))
    feature_configs.append(make_feature_config('fstdnsim{}', idx))
    feature_configs.append(make_feature_config('fvdegenergy{}', idx))

  # The tau feature needs more explit keypoints since it always goes
  # from 0 to 1.0.
  feature_configs += [
      configs.FeatureConfig(
          name='tau',
          lattice_size=2,
          monotonicity='increasing',
          pwl_calibration_input_keypoints=np.linspace(
              0., 1., num=config['num_keypoints']),
          regularizer_configs=[
              configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
          ],
      )
  ]
  return feature_configs


def make_input_fn(x: Type[pd.DataFrame], y: Type[pd.DataFrame], epochs: int,
                  batch_size: int) -> Callable[[], Tuple[tf.Tensor, tf.Tensor]]:
  """Create an estimator input_fn given some data."""
  pd_input_fn = tf_compat_v1_estimator.inputs.pandas_input_fn(
      x=x,
      y=y,
      shuffle=True,
      batch_size=batch_size,
      num_epochs=epochs,
      num_threads=1)

  def merged_input_fn():
    # We want the extreme values of 0.0 and 1.0 to be usable, but if we use
    # (0, 1) uniform sampling, they won't be.  So sample a wider interval
    # and clamp to it.
    tau_tensor = tf.random.uniform(shape=(batch_size,), minval=-.1, maxval=1.1)
    tau_tensor = tf.clip_by_value(tau_tensor, 0.0, 1.0)
    features, target = pd_input_fn()
    # Merge tau into the features with the correct batch size.
    features['tau'] = tau_tensor[:tf.shape(target)[0]]
    return features, target

  return merged_input_fn


def train_lattice(train_data: Type[pd.DataFrame], test_data: Type[pd.DataFrame],
                  config: Dict[str, Any]):
  """Train a deep lattice network on provided data.

  This trains a regressor estimator and writes a saved model to disk.

  Args:
    train_data: A pandas DataFrame to be used for training.
    test_data: A pandas DataFrame to be used for testing.
    config: A dictionary with the following parameters:
      'batch_size': batch size used for training.
      'learning_rate': The learning rate used for training.
      'num_epochs': number of times to loop over the data.
      'lattice_size': The number of interpolation points per feature.
      'num_keypoints': The number of quantiles to normalize the input features
        with.
      'model_dir': The location to write the model to. There are some other
        fields as well, see the FLAGS section.
  """
  # Parse configs updates from command line flags.
  config_updates = []
  for update in config['config_updates']:
    config_updates.extend(re.findall(r'(\S*)\s*=\s*(\S*)', update))

  # Target and y are both the MOS labels.
  train_y = train_data.pop('mos')
  train_x = train_data

  test_y = test_data.pop('mos')
  test_x = test_data
  logging.info('Train len %d test len %d', len(train_x), len(test_x))

  # feature_analysis_input_fn is used to collect statistics about the input
  # features, thus requiring only one loop of the dataset.
  #
  # feature_analysis_input_fn is required if you have at least one FeatureConfig
  # with "pwl_calibration_input_keypoints='quantiles'". Note that 'quantiles' is
  # default keypoints configuration so most likely you'll need it.
  feature_analysis_input_fn = make_input_fn(train_x, train_y, 1,
                                            config['batch_size'])

  train_input_fn = make_input_fn(train_x, train_y, config['num_epochs'],
                                 config['batch_size'])
  test_input_fn = make_input_fn(test_x, test_y, 1, config['batch_size'])

  feature_configs = make_feature_configs(config)
  feature_columns = make_feature_columns(config)
  feature_dict = make_feature_dict(config)

  def loss_fn(labels, logits, features):
    """Example-specific pinball loss based on value of tau."""
    tau = features['tau'][:, tf.newaxis]
    diff = labels - logits
    loss = (
        tf.maximum(diff, 0.0) * tf.cast(tau, dtype=tf.float32) +
        tf.minimum(diff, 0.0) * (tf.cast(tau, dtype=tf.float32) - 1.0))
    tf.debugging.assert_shapes([(logits, ('B', 1)), (diff, ('B', 1)),
                                (labels, ('B', 1)), (tau, ('B', 1))])
    return loss

  # Serving input fn is used to create saved models.
  serving_input_fn = (
      tf_estimator.export.build_parsing_serving_input_receiver_fn(
          feature_spec=fc.make_parse_example_spec(feature_columns)))
  # Also export a raw model that doesn't use TF Example protos as inputs but
  # uses raw tensors instead. This is better for TFLite.
  raw_serving_input_fn = (
      tf_estimator.export.build_raw_serving_input_receiver_fn(
          features=feature_dict))

  make_estimator_args = {
      'loss_fn': loss_fn,
      'feature_analysis_input_fn': feature_analysis_input_fn,
      'feature_columns': feature_columns,
      'feature_configs': feature_configs,
      'config': config,
      'config_updates': config_updates,
  }
  if config['model'] == 'linear':
    estimator = make_linear_estimator(**make_estimator_args)
  elif config['model'] == 'random_ensemble':
    estimator = make_random_ensemble_estimator(**make_estimator_args)
  elif config['model'] == 'tau_ensemble':
    estimator = make_tau_ensemble_estimator(**make_estimator_args)
  else:
    raise ValueError('unknown model type {}'.format(config['model']))

  estimator.train(input_fn=train_input_fn)
  if test_x.shape[0] > 0:
    results = estimator.evaluate(input_fn=test_input_fn)
    print('{} results: {}'.format(config['model'], results))

  exported_dir = estimator.export_saved_model(estimator.model_dir,
                                              serving_input_fn)
  print('{} TFExample proto input model exported to {}'.format(
      config['model'], exported_dir))

  raw_exported_dir = estimator.export_saved_model(
      os.path.join(estimator.model_dir, 'raw'), raw_serving_input_fn)
  print('{} Raw tensor input model exported to {}'.format(
      config['model'], raw_exported_dir))


def load_csv_to_data_frame(
    config: Dict[str, Any]) -> Tuple[Type[pd.DataFrame], Type[pd.DataFrame]]:
  """Loads data from csv files.

  Args:
    config: Dictionary with parameters described in FLAGS.

  Returns:
    A tuple of two pd.DataFrames for training and testing.
  """
  merged_dict = merge_fvnsim_and_mos(config['mos_csv'], config['fvnsim_csv'],
                                     config['num_fvnsim_bands'])
  # Use the first index and assume all the values have the same structure.
  # The value is a tuple with the first index as the fvnsims, and the second
  # index as the MOS.
  example = next(iter(merged_dict.items()))[1]

  # For each band there should fvnsim, fvnism10_, fstdnsim, and energy values.
  assert len(example.features) == config['num_fvnsim_bands'] * 4

  # Note: Assumes python 3.7's deterministic dictionaries.
  random.seed(RANDOM_SEED)
  merged_list = list(merged_dict.items())
  train_list = random.sample(merged_list,
                             int(len(merged_list) * config['train_set_ratio']))
  test_list = [x for x in merged_list if x not in train_list]

  def make_pandas_frame(data_list):
    data_dict = {}
    data_dict['mos'] = [v.mos for _, v in data_list]
    feature_names = get_feature_names(config['num_fvnsim_bands'])
    for feature_idx, feature_name in enumerate(feature_names):
      data_dict[feature_name] = [v.features[feature_idx] for _, v in data_list]

    # Sometimes the stddev can have 'NaN' values, which usually means something
    # went wrong in the ViSQOL batch processing.
    nans = [(k, v)
            for k, v in data_list
            if np.isnan(v.features[config['num_fvnsim_bands'] + 1])]
    if nans:
      logging.warn('found nans %s', nans)
    return pd.DataFrame(data_dict)

  return make_pandas_frame(train_list), make_pandas_frame(test_list)


def main(_):
  config = {
      'learning_rate': FLAGS.learning_rate,
      'batch_size': FLAGS.batch_size,
      'num_epochs': FLAGS.num_epochs,
      'prefitting_num_epochs': FLAGS.prefitting_num_epochs,
      'config_updates': FLAGS.config_updates,
      'num_lattices': FLAGS.num_lattices,
      'lattice_rank': FLAGS.lattice_rank,
      'lattice_size': FLAGS.lattice_size,
      'target_features_per_lattice': FLAGS.target_features_per_lattice,
      'target_fvnsim_per_lattice': FLAGS.target_fvnsim_per_lattice,
      'num_keypoints': FLAGS.num_keypoints,
      'use_output_calibration': FLAGS.use_output_calibration,
      'model': FLAGS.model,
      'model_dir': FLAGS.model_dir,
      'num_fvnsim_bands': FLAGS.num_fvnsim_bands,
      'mos_csv': FLAGS.mos_csv,
      'fvnsim_csv': FLAGS.fvnsim_csv,
      'train_set_ratio': FLAGS.train_set_ratio,
  }
  data = load_csv_to_data_frame(config)
  train_lattice(data[0], data[1], config)


if __name__ == '__main__':
  app.run(main)
