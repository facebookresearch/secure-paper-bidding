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
parser.add_argument('--num_sim', default=10, type=int, help='the number of samples in each bins')
parser.add_argument('--L', default=0, type=int, help='the colluding size for attack evalaution')
parser.add_argument('--cheat_mode', default="white_box", type=str, help="white_box / black_box /simple_black_box")
args = parser.parse_args()

logger = set_logger("attack results evaluation")
logger.info(args)

num_paper, num_reviewer, _, _, hashed_ratio = get_global_variable()
num_sim = args.num_sim
X_seeds = range(args.num_seeds_X)
K = args.K
lam = args.lam
subsample_max = args.subsample_max
cheat_mode = args.cheat_mode
L = args.L

X_csr_s = []
H_inv_s = []
y, y_train = load_y(hashed_ratio, logger, subsample_max=subsample_max)
preds_s = []
for seed in X_seeds:
    X_csr, H_inv = load_X_and_H_inv(hashed_ratio, seed, logger, lam)
    preds = load_preds(X_csr, y_train, H_inv, hashed_ratio, seed, logger, lam, subsample_max=subsample_max)
    X_csr_s.append(X_csr)
    H_inv_s.append(H_inv)
    preds_s.append(preds)
    del X_csr, H_inv, preds
    
ensemble_preds = np.add.reduce(preds_s)
prev_rank = np.argsort(-(ensemble_preds))

interval_num = int(np.log2(num_reviewer + 1 - K) + 2) 
suc_rate = np.zeros([interval_num])
datas = []
for seed in range(args.num_seeds_results):
    data = np.load("{}/attack/attack_collusion_{}_top_{}_{}_num_seeds_1_lam_{}_subsample_max_{}_seed_{}.npz".format(args.output_dir, L, K, cheat_mode, lam, subsample_max, seed))
    datas.append(data)
advs_rank = np.concatenate([data["advs_rank"] for data in datas])
advs_original_rank = np.concatenate([data["advs_original_rank"] for data in datas])
suc_ind = np.concatenate([data["suc_ind"] for data in datas])
num_samples = suc_ind.shape[0]
suc_rate[0] = suc_ind[advs_original_rank < K].sum() / float(num_samples * 10)
for i in range(interval_num - 1):
    suc_rate[i + 1] = suc_ind[ (advs_original_rank >= K + np.power(2, i) - 1) & (advs_original_rank < K + np.power(2, i+1) - 1)].sum() / float(num_samples * min(10, np.power(2, i)))
del data
logger.info("Size of colluding group: {}, success rate at each bins: {}:".format(L, suc_rate[1:]))
