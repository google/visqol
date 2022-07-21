r"""Converts a lattice SavedModel trained with train_lattice.py to TFLite."""
import os

from absl import app
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_dir', None,
                    'Path to the SavedModel directory.')
flags.DEFINE_string('output_dir', None,
                    'Directory where the TFLite model will be written to.')
flags.mark_flag_as_required('saved_model_dir')
flags.mark_flag_as_required('output_dir')


def main(_):
  print(f'Converting saved model at {FLAGS.saved_model_dir}')
  converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.saved_model_dir)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
  ]
  tflite_model = converter.convert()

  # Save the model.  Use the saved model directory name as the basename.
  output_path = os.path.join(
      FLAGS.output_dir,
      os.path.basename(FLAGS.saved_model_dir) + '.tflite')

  with open(output_path, 'wb') as f:
    f.write(tflite_model)
  print(f'Wrote TFLite model to {output_path}')


if __name__ == '__main__':
  app.run(main)
