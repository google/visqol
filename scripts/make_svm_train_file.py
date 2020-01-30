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

"""Creates a file from ViSQOL to work with svm-train.

Takes as input ViSQOL's batch output and a MOSLQS csv file and creates
a file that works as input for svm-train in libsvm.
The output just goes to stdout so pipe it to a file once you are happy with it.

Typically a grid search will be needed - the same file will work with
libsvm's grid.py with some minor modification.

The modification is in grid.py's LocalWorker Class, run_one method:
-      if str(line).find('Cross') != -1:
-        return float(line.split()[-1][0:-1])
+      if str(line).find('Cross Validation Squared correlation coefficient') != -1:
+        ret = float(line.split()[-1])

This change allows grid.py to work with the regression flags.
Here is an example invocation that searches a coarse space.  Note that the -n nu parameter must be found independently:
libsvm-3.24/tools/grid.py -v 4 -log2c -5,15,1 -log2g -5,10,1. -s 4 -t 2 -n .568 file_created_by_make_svm_train_file.txt

You can repeat the process with finer and finer searches using -log2{c,g}, and then create the final model by using the parameters:
libsvm-3.24/svm-train -s 4 -t 2 -n .568 -c 5.31474325639 -g 3.17773760038 file_created_by_make_svm_train_file.txt output_model_file_usable_with_visqols_similarity_to_quality_model_flag.txt
"""
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("results_csv_path", None,
                    "Path to ViSQOL results_csv output")
flags.DEFINE_string("moslqs_csv_path", None,
                    "Path to csv containing moslqs as the last element per row, "
                    "aligned to the results_csv")

# Required flag.
flags.mark_flag_as_required("results_csv_path")
flags.mark_flag_as_required("moslqs_csv_path")

def main(argv):
  del argv  # Unused.

  fvnsims_df = pd.read_csv(FLAGS.results_csv_path)
  moslqs = pd.read_csv(FLAGS.moslqs_csv_path)

  # Remove the ViSQOL reference,degraded columns that come before the nsim.
  fvnsims_cols = fvnsims_df[fvnsims_df.columns[2:]]
  merged = moslqs.join(fvnsims_cols)

  for index, row in merged.iterrows():
    label_str = str(row['moslqs'])
    # Gather the fvnsims in 1:fvnsim0, 2:fvnsim1 format
    input_str = ''.join([' {}:{}'.format(i + 1, row['fvnsim'+str(i)]) for i in range(len(row) - 1)])
    print(label_str + input_str)


if __name__ == '__main__':
  app.run(main)
