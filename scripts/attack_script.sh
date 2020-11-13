# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python ./src/attack.py --L 4 --K 50 --num_sample_paper 400 --output_dir ./results/ --num_seeds 1 --cheat_mode white_box --num_threads 5 --subsample_max 60 --lam 1e-4 --seed 0
python ./src/attack_results_evaluation.py --L 4 --K 50 --output_dir ./results/ --num_seeds_X 1 --cheat_mode white_box --subsample_max 60 --lam 1e-4 --num_seeds_results 1
