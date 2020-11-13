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
from multiprocessing.pool import ThreadPool
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--K', default=30, type=int, help='the top K reviewer for assignment and detection')
parser.add_argument('--L', default=1, type=int, help='the number of additional collusion people')
parser.add_argument('--output_dir', default='./results/', type=str, help='output dir')
parser.add_argument('--num_seeds', default=1, type=int, help='number of seeds for model ensemble')
parser.add_argument('--num_threads', default=6, type=int, help='number of threads')
parser.add_argument('--lam', default=1e-4, type=float, help='L2 regularization')
parser.add_argument('--subsample_max', default=0, type=int, help='max subsampled training labels')
parser.add_argument('--start_id', default=None, type=int, help='start paper id')
parser.add_argument('--end_id', default=None, type=int, help='end paper id')

args = parser.parse_args()
num_paper, num_reviewer, input_dir, max_pc_quota, hashed_ratio = get_global_variable()

K = args.K
L = args.L
lam = args.lam
subsample_max = args.subsample_max
hashed_ratio = 0.0
seeds = range(args.num_seeds)
c_train = 2.0 / (num_reviewer * num_paper)

start_id = args.start_id
if start_id is None:
    start_id = 0
    end_id = num_paper
else:
    end_id = min(args.end_id, num_paper)

os.makedirs(os.path.join(args.output_dir, 'detect'), exist_ok=True)

output_name = "{}/detect_fpr/defense_fpr_collusion_{}_top_{}_num_seeds_{}_lam_{}_subsample_max_{}_start_id_{}_end_id_{}".format(args.output_dir, L, K, args.num_seeds, lam, subsample_max, args.start_id, args.end_id)
if os.path.exists(output_name + ".npz"):
    print("done")
    os.exit()

logger = set_logger("defense", "{}/detect_fpr/log_defense_fpr_collusion_{}_top_{}_num_seeds_{}_lam_{}_subsample_max_{}_start_id_{}_end_id_{}.txt".format(args.output_dir, L, K, args.num_seeds, lam, subsample_max, args.start_id, args.end_id))
logger.info(args)

logger.info("lam: {}, subsample_max, {}".format(lam, subsample_max))

#1. init data
X_csr_s = []
H_inv_s = []
y, y_train = load_y(hashed_ratio, logger, subsample_max=subsample_max)
logger.info("pos 3: {}, pos 2: {}, pos 1: {}, neg 0: {}".format((y_train == 3).sum(), (y_train == 2).sum(), (y_train == 1).sum(), (y_train == 0).sum()))
preds_s = []
for seed in seeds:
    X_csr, H_inv = load_X_and_H_inv(hashed_ratio, seed, logger, lam)
    preds = load_preds(X_csr, y_train, H_inv, hashed_ratio, seed, logger, lam, subsample_max=subsample_max)
    X_csr_s.append(X_csr)
    H_inv_s.append(H_inv)
    preds_s.append(preds)
    del X_csr, H_inv, preds

logger.info(len(X_csr_s))
    
ensemble_preds = np.add.reduce(preds_s)
prev_rank = np.argsort(-ensemble_preds)

advs_rank = np.zeros([end_id - start_id, K])
advs_collusion = np.zeros([end_id - start_id, K, L + 1])
advs_sol = np.ones([end_id - start_id, K, L + 1, num_paper]) * (-10)
def thread_detect(i):
    adv_pairs = prev_rank[i][:K]
    X_p_H_inv_s = np.stack([np.asarray(X_csr[adv_pairs * num_paper + i].dot(H_inv)) for (X_csr, H_inv) in zip(X_csr_s, H_inv_s)], axis=0)
    if L == 0:
        #v_batch = np.zeros([len(adv_pairs), num_paper])
        v_batch = np.stack([np.add.reduce([c_train * X_csr[adv_pairs[j] * num_paper : (adv_pairs[j] + 1) * num_paper].dot(X_p_H_inv[j]) for (X_csr, X_p_H_inv) in zip(X_csr_s, X_p_H_inv_s)]) for j in range(len(adv_pairs))], axis=0)
    else:
         v_batch = c_train * np.add.reduce([X_csr.dot(X_p_H_inv.T).T.reshape(-1, num_paper) for (X_csr, X_p_H_inv) in zip(X_csr_s, X_p_H_inv_s)])
            
    sol = np.zeros(v_batch.shape)
        
    if L != 0:
        delta_y = sol - np.tile(y_train.reshape(-1, num_paper), [len(adv_pairs), 1])
        val = np.einsum("ij,ij->i", v_batch, delta_y).reshape(-1, num_reviewer)
        collusion_rev = np.argsort(val)[:, :L+1]
        collusion_rev[np.sum(adv_pairs == collusion_rev.T, axis=0) == 0, -1] = adv_pairs[np.sum(adv_pairs == collusion_rev.transpose(), axis=0) == 0]
        advs_collusion[i - start_id, np.arange(K)] = collusion_rev
        advs_sol[i - start_id, np.arange(K)] = sol[(collusion_rev + np.tile(np.expand_dims(np.arange(len(adv_pairs)), 1) * num_reviewer, [1, L + 1])).reshape(-1)].reshape([len(adv_pairs), L+1, num_paper])
    else:
        collusion_rev = np.expand_dims(adv_pairs, 1)
        delta_y = sol - y_train.reshape(-1, num_paper)[adv_pairs]
        advs_collusion[i - start_id, np.arange(K)] = collusion_rev
        advs_sol[i - start_id, np.arange(K), 0] = sol
    
    del sol
    del X_p_H_inv_s
    
    delta_X_train_t_y_s = np.zeros([args.num_seeds, H_inv_s[0].shape[0], len(adv_pairs)])
    for j in range(len(adv_pairs)):
        if L != 0:
            delta_X_train_t_y_s[:, :, j] = c_train * np.stack([X_csr.transpose()[:, (np.expand_dims(collusion_rev[j] * num_paper, 1) + np.expand_dims(np.arange(num_paper), 0)).reshape(-1)].dot(delta_y[j * num_reviewer + collusion_rev[j]].reshape(-1)) for X_csr in X_csr_s] , axis=0)
        else:
            delta_X_train_t_y_s[:, :, j] = c_train * np.stack([X_csr.transpose()[:, (np.expand_dims(collusion_rev[j] * num_paper, 1) + np.expand_dims(np.arange(num_paper), 0)).reshape(-1)].dot(delta_y[j * (L + 1) : (j + 1) * (L + 1)].reshape(-1)) for X_csr in X_csr_s] , axis=0)
    mul_delta_preds_p = np.add.reduce([np.asarray(X_csr[np.arange(num_reviewer) * num_paper + i].dot(H_inv.dot(delta_X_train_t_y))) for (X_csr, H_inv, delta_X_train_t_y) in zip(X_csr_s, H_inv_s, delta_X_train_t_y_s)])
    mul_preds_p = np.tile(np.expand_dims(ensemble_preds[i], 1), [1, len(adv_pairs)])
    
    adv_rank = np.argsort(np.argsort(-(mul_delta_preds_p + mul_preds_p).T))[np.arange(len(adv_pairs)), adv_pairs]
    advs_rank[i - start_id, np.arange(K)] = adv_rank
    logger.info(adv_rank)
    del delta_y, collusion_rev, delta_X_train_t_y_s, mul_delta_preds_p, mul_preds_p, adv_rank

    logger.info("finish paper {}: detect number: {} / {}; top 5 detect number: {} / {};  top 10 detect number: {} / {}; top 20 detect number: {} / {}".format(i, np.sum(advs_rank[i - start_id] >= K),  K, np.sum(advs_rank[i - start_id, :5] >= K), 5, np.sum(advs_rank[i - start_id, :10] >= K), 10, np.sum(advs_rank[i - start_id, :20] >= K), 20))
    
p = ThreadPool(args.num_threads)
_ = list(tqdm(p.imap(thread_detect, range(start_id, end_id)), total=end_id - start_id, desc='thread_detect'))

logger.info(f"start saving {output_name}")
np.savez(output_name, advs_rank=advs_rank, advs_sol=advs_sol, advs_collusion=advs_collusion)
logger.info(f"end saving {output_name}")
