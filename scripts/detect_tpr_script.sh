# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python ./src/detect_tpr.py --L_attack 4 --cheat_mode white_box --num_seeds 1 --lam 1e-4 --subsample_max 60 --K 50 --output_dir ./results --seed 0
python ./src/detect_tpr_results_evaluation.py --L_attack 4 --cheat_mode white_box --K 50 --subsample_max 60 --num_seeds_results 1 --num_seeds_X 1 --lam 1e-4 --output_dir ./results/
