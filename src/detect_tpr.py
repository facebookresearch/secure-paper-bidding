# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from scipy import sparse
import time
import argparse
from logger import set_logger
import os
from utils import assign_IP, load_X_and_H_inv, load_y, load_preds, solve_ip_adv, get_global_variable
import threading
from tqdm.auto import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--K', default=30, type=int, help='the top K reviewer for assignment and detection')
parser.add_argument('--L_attack', default=0, type=int, help='the size of colluding group')
parser.add_argument('--output_dir', default='./results/', type=str, help='output dir')
parser.add_argument('--cheat_mode', default="white_box", type=str, help="white_box / black_box")
parser.add_argument('--num_seeds', default=1, type=int, help='number of seeds for model ensemble')
parser.add_argument('--seed', default=0, type=int, help='see for attack example')
parser.add_argument('--num_threads', default=4, type=int, help='number of threads')
parser.add_argument('--lam', default=1e-4, type=float, help='L2 regularization')
parser.add_argument('--subsample_max', default=60, type=int, help='max subsampled training labels')
args = parser.parse_args()

num_paper, num_reviewer, input_dir, max_pc_quota, hashed_ratio = get_global_variable()
seeds = range(args.num_seeds)
L_attack = args.L_attack
K = args.K
lam = args.lam
subsample_max = args.subsample_max
cheat_mode = args.cheat_mode
c_train = 2.0 / (num_reviewer  * num_paper)

logger = set_logger("detect_tpr", "{}/detect_tpr/log_detect_tpr_collusion_{}_top_{}_{}_lam_{}_subsample_max_{}_seed_{}.txt".format(args.output_dir, L_attack, K, cheat_mode, lam, subsample_max, args.seed))
logger.info(args)

#1. init data
X_csr_s = []
H_inv_s = []
y, y_train = load_y(hashed_ratio, logger, subsample_max=subsample_max)
preds_s = []
for seed in seeds:
    X_csr, H_inv = load_X_and_H_inv(hashed_ratio, seed, logger, lam)
    preds = load_preds(X_csr, y_train, H_inv, hashed_ratio, seed, logger, lam, subsample_max=subsample_max)
    X_csr_s.append(X_csr)
    H_inv_s.append(H_inv)
    preds_s.append(preds)
    del X_csr, H_inv, preds

ensemble_preds = np.add.reduce(preds_s)
prev_rank = np.argsort(-ensemble_preds)

def detect(p, r, preds, y_train, L_dict):
    adv_pairs = np.asarray([r]).astype(np.int)
    X_p_H_inv_s = [np.asarray(X_csr[adv_pairs * num_paper + p].dot(H_inv)) for (X_csr, H_inv) in zip(X_csr_s, H_inv_s)]
    v_batch = c_train * np.add.reduce([X_csr.dot(X_p_H_inv.T).T.reshape(-1, num_paper) for (X_csr, X_p_H_inv) in zip(X_csr_s, X_p_H_inv_s)])
    
    delta_X_train_t_y_s = np.zeros([args.num_seeds, H_inv_s[0].shape[0], len(L_dict.keys())])
    sol = np.zeros(v_batch.shape)
    delta_y = sol - np.tile(y_train.reshape(-1, num_paper), [len(adv_pairs), 1])
    val = np.einsum("ij,ij->i", v_batch, delta_y).reshape(-1, num_reviewer)
    for L in L_dict.keys():
        collusion_rev = np.argsort(val)[:, :L+1]
        collusion_rev[np.sum(adv_pairs == collusion_rev.T, axis=0) == 0, -1] = adv_pairs[np.sum(adv_pairs == collusion_rev.transpose(), axis=0) == 0]
        delta_X_train_t_y_s[:, :,  L_dict[L]] = np.stack([c_train * X_csr.transpose()[:, (np.expand_dims(collusion_rev[0], 1) * num_paper + np.arange(num_paper)).reshape(-1)].dot(delta_y[collusion_rev[0]].reshape(-1)) for X_csr in X_csr_s], axis=0)

    mul_delta_preds_p = np.add.reduce([np.asarray(X_csr[np.arange(num_reviewer) * num_paper + p].dot(H_inv.dot(delta_X_train_t_y))) for (X_csr, H_inv, delta_X_train_t_y) in zip(X_csr_s, H_inv_s, delta_X_train_t_y_s)])
    mul_preds_p = np.tile(np.expand_dims(preds[p], 1), [1, len(adv_pairs)])
    adv_rank = np.argsort(np.argsort(-(mul_delta_preds_p + mul_preds_p).T))[np.arange(len(L_dict.keys())), np.tile(adv_pairs, [len(L_dict.keys())])]
    
    return (adv_rank >= K).astype(int).reshape(len(L_dict.keys()))

#2. init attack samples
data = np.load("{}/attack/attack_collusion_{}_top_{}_{}_num_seeds_{}_lam_{}_subsample_max_{}_seed_{}.npz".format(args.output_dir, L_attack, K, cheat_mode, args.num_seeds, lam, subsample_max, args.seed))
advs_rank = data["advs_rank"]
advs_sol = data["advs_sol"]
advs_collusion = data["advs_collusion"]
advs_rev = data["advs_rev"]
advs_original_rank = data["advs_original_rank"]
sampled_papers = data["sampled_papers"]
del data

(num_sample_paper, sample_num) = advs_rev.shape

L_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        
def thread_eval_detect(i_j_pair):
    (i, j) = i_j_pair
    collusion_rev = advs_collusion[i, j].astype(np.int)
    r = advs_rev[i, j].astype(np.int)
    p = sampled_papers[i].astype(np.int)
    y_adv = y_train.copy()
    y_adv[(np.expand_dims(collusion_rev * num_paper, 1) + np.expand_dims(np.arange(num_paper), 0)).reshape(-1)] = np.expand_dims(advs_sol[i, j].reshape(-1), 1)

    y_delta = (advs_sol[i, j] - y_train.reshape(-1, num_paper)[collusion_rev]).reshape(-1)
    preds_delta = np.add.reduce([X_csr.dot(H_inv.dot(c_train * X_csr.transpose()[:, (np.expand_dims(collusion_rev * num_paper, 1) + np.expand_dims(np.arange(num_paper), 0)).reshape(-1)].dot(y_delta))).reshape(-1, num_paper).T for (X_csr, H_inv) in zip(X_csr_s, H_inv_s)])
    preds_adv = ensemble_preds + preds_delta

    detect_result = detect(p, r, preds_adv, y_adv, L_dict)
    del collusion_rev, y_adv, y_delta, preds_delta, preds_adv
    logger.info("pair ({}, {}) || detect result: {}".format(i, j, detect_result.squeeze()))
    
    return detect_result, i, j, r, p
    
p = Pool(processes=args.num_threads)
logger.info(f"num_threads: {args.num_threads}")
i_j_pair_list = list((i, j) for i in range(num_sample_paper) for j in range(sample_num) if not (advs_rev[i, j] == -1 or  advs_original_rank[i, j] < K or advs_rank[i, advs_original_rank[i, j].astype(int)] >= K))
results = list(tqdm(p.imap(thread_eval_detect, i_j_pair_list), total=len(i_j_pair_list), desc='thread_detect'))
detect_num = np.stack([result[0] for result in results], axis=0)
i_s = np.stack([result[1] for result in results], axis=0)
j_s = np.stack([result[2] for result in results], axis=0)
rs = np.stack([result[3] for result in results], axis=0)
ps = np.stack([result[4] for result in results], axis=0)
logger.info(detect_num.shape)
logger.info(detect_num)
np.savez("{}/detect_tpr/detect_tpr_collusion_{}_top_{}_{}_num_seeds_{}_lam_{}_subsample_max_{}_seed_{}.npz".format(args.output_dir, L_attack, K, cheat_mode, args.num_seeds, lam, subsample_max, args.seed), detect_num=detect_num, total_num=len(i_j_pair_list), i_s=i_s, j_s=j_s, rs=rs, ps=ps)  
