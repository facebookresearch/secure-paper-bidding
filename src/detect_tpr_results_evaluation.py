# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from utils import assign_IP, load_X_and_H_inv, load_y, load_preds, solve_ip_adv, get_global_variable
import argparse
from logger import set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='./results/', type=str, help='output dir')
parser.add_argument('--lam', default=1e-4, type=float, help='L2 regularization')
parser.add_argument('--num_seeds_X', default=1, type=int, help='seed for hashing')
parser.add_argument('--num_seeds_results', default=1, type=int, help='seed for attack sampling')
parser.add_argument('--subsample_max', default=60, type=int, help='max subsampled training labels')
parser.add_argument('--K', default=50, type=int, help='top K reviewers per paper considered for assignment')
parser.add_argument('--L_attack', default=0, type=int, help='the attacking colluding size for detect evalaution')
parser.add_argument('--cheat_mode', default="white_box", type=str, help="white_box / black_box /simple_black_box")
args = parser.parse_args()

logger = set_logger("detect tpr results evaluation")
logger.info(args)

num_paper, num_reviewer, _, _, hashed_ratio = get_global_variable()
num_seeds = args.num_seeds_X
K = args.K
lam = args.lam
subsample_max = args.subsample_max
cheat_mode = args.cheat_mode
output_dir = args.output_dir
L_attack = args.L_attack

detect_rate = np.zeros([2, 5])

detect_num_s = []
total_num_s = []
adv_rank_s = []
for seed in range(args.num_seeds_results):
    data = np.load("{}/attack/attack_collusion_{}_top_{}_{}_num_seeds_{}_lam_{}_subsample_max_{}_seed_{}.npz".format(output_dir, L_attack, K, cheat_mode, num_seeds, lam, subsample_max, seed))

    advs_rank = data["advs_rank"]
    advs_collusion = data["advs_collusion"]
    advs_rev = data["advs_rev"]
    advs_original_rank = data["advs_original_rank"]
    del data

    (num_sample_paper, sample_num) = advs_rev.shape
    adv_test_num = 0
    for i in range(num_sample_paper):
        for j in range(sample_num):
            if advs_rev[i, j] == -1:
                continue
            if advs_original_rank[i, j] < K:
                continue
            if advs_rank[i, advs_original_rank[i, j].astype(int)] >= K:
                continue
            adv_test_num += 1
    adv_rank = np.zeros(adv_test_num)
    adv_test_num = 0
    for i in range(num_sample_paper):
        for j in range(sample_num):
            if advs_rev[i, j] == -1:
                continue
            if advs_original_rank[i, j] < K:
                continue
            if advs_rank[i, advs_original_rank[i, j].astype(int)] >= K:
                continue
            adv_rank[adv_test_num] = advs_rank[i, advs_original_rank[i, j].astype(int)]
            adv_test_num += 1

    data = np.load("{}/detect_tpr/detect_tpr_collusion_{}_top_{}_{}_num_seeds_{}_lam_{}_subsample_max_{}_seed_{}.npz".format(output_dir, L_attack, K, cheat_mode, num_seeds, lam, subsample_max, seed))
    total_num_s.append(data["total_num"])
    adv_rank_s.append(adv_rank)
    detect_num_s.append(data["detect_num"])

detect_num_total = np.concatenate(detect_num_s)
total_num_s = sum(total_num_s)
adv_rank_total = np.concatenate([adv_rank for adv_rank in adv_rank_s])

detect_rate[0] = detect_num_total[adv_rank_total <= 4].sum(0) / (adv_rank_total <= 4).sum()
detect_rate[1] = detect_num_total.sum(0) / total_num_s

logger.info("top 5 tpr when M_d = 1, ..., 5: {}".format(detect_rate[0]))
logger.info("top 50 tpr when M_d = 1, ..., 5: {}".format(detect_rate[1]))
