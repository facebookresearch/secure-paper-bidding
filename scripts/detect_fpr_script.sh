# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python ./src/detect_fpr.py --K 50 --output_dir ./results/ --num_seeds 1  --num_threads 5 --subsample 60 --lam 1e-4 --L 4
python ./src/detect_fpr_results_evaluation.py --K 50 --output_dir ./results/ --num_seeds_X 1 --subsample 60 --lam 1e-4 --L 4 --input_dir ./data/
