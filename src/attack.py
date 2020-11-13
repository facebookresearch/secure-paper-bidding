# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse
from logger import set_logger
from ortools.graph import pywrapgraph
from collections import Counter
from utils import assign_IP, load_X_and_H_inv, load_y, load_preds, solve_ip_adv, get_global_variable
from multiprocessing.pool import ThreadPool
from tqdm.auto import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--K', default=30, type=int, help='the top K reviewer for assignment and detection')
parser.add_argument('--L', default=1, type=int, help='the number of additional collusion people')
parser.add_argument('--num_sample_paper', default=400, type=int, help="the number of papers to attack")
parser.add_argument('--cheat_mode', default="white_box", type=str, help="white_box / black_box /simple_black_box")
parser.add_argument('--output_dir', default='./results/', type=str, help='output dir')
parser.add_argument('--num_sim', default=10, type=int, help='the number of samples in each bins')
parser.add_argument('--num_seeds', default=1, type=int, help='number of seeds for model ensemble')
parser.add_argument('--seed', default=0, type=int, help='seed for generate samples')
parser.add_argument('--num_threads', default=6, type=int, help='number of threads')
parser.add_argument('--lam', default=1e-4, type=float, help='L2 regularization')
parser.add_argument('--subsample_max', default=0, type=int, help='max subsampled training labels')
args = parser.parse_args()

num_paper, num_reviewer, input_dir, max_pc_quota, hashed_ratio = get_global_variable()
seeds = range(args.num_seeds)
c_train = 2.0 / (num_reviewer * num_paper)
num_sim = args.num_sim
K = args.K
L = args.L
lam = args.lam
subsample_max = args.subsample_max
num_sample_paper = args.num_sample_paper
cheat_mode = args.cheat_mode

os.makedirs(os.path.join(args.output_dir, 'attack'), exist_ok=True)

output_name = "{}/attack/attack_collusion_{}_top_{}_{}_num_seeds_{}_lam_{}_subsample_max_{}_seed_{}".format(args.output_dir, L, K, cheat_mode, args.num_seeds, lam, subsample_max, args.seed)

logger = set_logger("attack", "{}/attack/log_attack_collusion_{}_top_{}_{}_num_seeds_{}_lam_{}_subsample_max_{}_seed_{}.txt".format(args.output_dir, L, K, cheat_mode, args.num_seeds, lam, subsample_max, args.seed))
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
prev_rank = np.argsort(-(ensemble_preds))

if cheat_mode.endswith("black_box"):
    r_pointer, p_pointer = 368, 368 + 368 + 930 
        
    r_feature = X_csr_s[0][num_paper * np.arange(num_reviewer), :r_pointer]
    inner_r_feature = np.asarray(r_feature.dot(r_feature.todense().transpose()))
    
    p_feature = X_csr_s[0][:num_paper, r_pointer : p_pointer]
    inner_p_feature = np.asarray(p_feature.dot(p_feature.todense().transpose()))

#2. init saving cache
if args.cheat_mode == "simple_black_box":
    sample_num = np.concatenate(list((np.power(2, j) - 1 + np.random.permutation(min(np.power(2, j), num_reviewer - np.power(2, j) + 1)))[:num_sim] for j in range(int(np.log2(num_reviewer + 1) + 1)))).shape[0]
else:
    sample_num = np.concatenate([np.random.permutation(K)[:num_sim]] + list((K + np.power(2, j) - 1 + np.random.permutation(min(np.power(2, j), num_reviewer - np.power(2, j) - K + 1)))[:num_sim] for j in range(int(np.log2(num_reviewer + 1 - K) + 1)))).shape[0]

advs_rank = -np.ones([num_sample_paper, num_reviewer])
advs_collusion = -np.ones([num_sample_paper, sample_num, L + 1])
advs_sol = np.ones([num_sample_paper, sample_num, L + 1, num_paper]) * (-10)
advs_original_rank = -np.ones([num_sample_paper, sample_num])
advs_rev = -np.ones([num_sample_paper, sample_num])
suc_ind = np.zeros([num_sample_paper, sample_num])

#4. start simulation
def thread_attack(idx): #id for sample, id for paper
    sample_idx, i = idx
    logger.info("{} / {}".format(sample_idx, i))
    #4a. sample original rank, remove the reviewers who are not in training set
    if args.cheat_mode == "simple_black_box":
        sample_orig_rank = np.concatenate(list((np.power(2, j) - 1 + np.random.permutation(min(np.power(2, j), num_reviewer - np.power(2, j) + 1)))[:num_sim] for j in range(int(np.log2(num_reviewer + 1) + 1))))
    else:
        sample_orig_rank = np.concatenate([np.random.permutation(K)[:num_sim]] + list((K + np.power(2, j) - 1 + np.random.permutation(min(np.power(2, j), num_reviewer - np.power(2, j) - K + 1)))[:num_sim] for j in range(int(np.log2(num_reviewer + 1 - K) + 1))))
    adv_pairs = prev_rank[i, sample_orig_rank]
    sample_orig_rank = sample_orig_rank
    advs_original_rank[sample_idx] = sample_orig_rank
    advs_rev[sample_idx] = adv_pairs
    
    #4b. find the collusion group and corresponding bidding vector (Compute collusion_rev and sol)
    X_p_H_inv_s = np.zeros([args.num_seeds, len(adv_pairs), H_inv_s[0].shape[0]])
    for ind_seed in range(args.num_seeds):
        X_p_H_inv_s[ind_seed] = np.asarray(X_csr_s[ind_seed][adv_pairs * num_paper + i].dot(H_inv_s[ind_seed]))   
    if cheat_mode == "white_box":
        if L == 0:
            v_batch = np.zeros([len(adv_pairs), num_paper])
            for j in range(len(adv_pairs)):
                v_batch[j] = np.add.reduce([c_train * X_csr[adv_pairs[j] * num_paper : (adv_pairs[j] + 1) * num_paper].dot(X_p_H_inv[j]) for (X_csr, X_p_H_inv) in zip(X_csr_s, X_p_H_inv_s)])
        else:
            v_batch = c_train * np.add.reduce([X_csr.dot(X_p_H_inv.T).T.reshape(-1, num_paper) for (X_csr, X_p_H_inv) in zip(X_csr_s, X_p_H_inv_s)])
            
        sol = solve_ip_adv(v_batch, opt_max=True, subsample_max=subsample_max)
        if L != 0:
            delta_y = sol - np.tile(y_train.reshape(-1, num_paper), [len(adv_pairs), 1])
            val = np.einsum("ij,ij->i", v_batch, delta_y).reshape(-1, num_reviewer)
            collusion_rev = np.argsort(-val)[:, :L+1]
            collusion_rev[np.sum(adv_pairs == collusion_rev.T, axis=0) == 0, -1] = adv_pairs[np.sum(adv_pairs == collusion_rev.transpose(), axis=0) == 0]
            advs_sol[sample_idx, advs_rev[sample_idx] != -1] = sol[np.concatenate([collusion_rev[j] + j * num_reviewer for j in range(len(adv_pairs))], )].reshape(len(adv_pairs), L + 1, num_paper)
            del val
        else:
            collusion_rev = np.expand_dims(adv_pairs, 1)
            delta_y = sol - y_train.reshape(-1, num_paper)[adv_pairs]
            advs_sol[sample_idx, advs_rev[sample_idx] != -1] = np.expand_dims(sol, 1)
        del v_batch
    elif cheat_mode == "black_box":
        collusion_rev = np.argsort(-inner_r_feature[adv_pairs])[:, :L+1]
        collusion_rev[np.sum(adv_pairs == collusion_rev.T, axis=0) == 0, -1] = adv_pairs[np.sum(adv_pairs == collusion_rev.transpose(), axis=0) == 0]
        sol = np.zeros([sample_num, L + 1, num_paper])
        for j in range(len(adv_pairs)):
            inner_feature = X_csr_s[0][(np.expand_dims(collusion_rev[j] * num_paper, 1) + np.expand_dims(np.arange(num_paper), 0)).reshape(-1)].dot(np.asarray(X_csr_s[0][adv_pairs[j] * num_paper + i].todense()).squeeze())
            inner_feature = inner_feature.reshape([L + 1, num_paper])
            rank_p = np.argsort(-inner_feature)
            sol[j, np.repeat(np.arange(L+1), subsample_max), rank_p[:, :subsample_max].reshape(-1)] = 3
        advs_sol[sample_idx] = sol
        delta_y = sol.reshape([sample_num * (L + 1), num_paper]) - y_train.reshape(-1, num_paper)[collusion_rev.reshape(-1)]
    elif cheat_mode == "simple_black_box":
        collusion_rev = np.expand_dims(adv_pairs, axis=1)
        sol = np.zeros([num_paper])
        sol[i] = 3 
        advs_sol[sample_idx, advs_rev[sample_idx] != -1] = sol
        delta_y = sol - y_train.reshape(-1, num_paper)[collusion_rev.reshape(-1)]
    else:
        logger.info("invalid cheat mode")
        exit(0)
    advs_collusion[sample_idx, advs_rev[sample_idx] != -1] = collusion_rev
    
    #4c. manipulate cheating
    delta_X_train_t_y_s = np.zeros([args.num_seeds, H_inv_s[0].shape[0], len(adv_pairs)])
    for j in range(len(adv_pairs)):
        if cheat_mode == "white_box":
            if L != 0:
                delta_X_train_t_y_s[:, :, j] = c_train * np.stack([X_csr.transpose()[:, (np.expand_dims(collusion_rev[j] * num_paper, 1) + np.expand_dims(np.arange(num_paper), 0)).reshape(-1)].dot(delta_y[j * num_reviewer + collusion_rev[j]].reshape(-1)) for X_csr in X_csr_s] , axis=0)
            else:
                delta_X_train_t_y_s[:, :, j] = c_train * np.stack([X_csr.transpose()[:, (np.expand_dims(collusion_rev[j], 1) * num_paper + np.arange(num_paper)).reshape(-1)].dot(delta_y[j].reshape(-1)) for X_csr in X_csr_s] , axis=0)
        elif cheat_mode.endswith("black_box"):
            delta_X_train_t_y_s[:, :, j] = c_train * np.stack([X_csr.transpose()[:, (np.expand_dims(collusion_rev[j] * num_paper, 1) + np.expand_dims(np.arange(num_paper), 0)).reshape(-1)].dot(delta_y[j * (L + 1) : (j + 1) * (L + 1)].reshape(-1)) for X_csr in X_csr_s] , axis=0)
    mul_delta_preds_p = np.add.reduce([np.asarray(X_csr[np.arange(num_reviewer) * num_paper + i].dot(H_inv.dot(delta_X_train_t_y))) for (X_csr, H_inv, delta_X_train_t_y) in zip(X_csr_s, H_inv_s, delta_X_train_t_y_s)])
    
    mul_preds_p = np.tile(np.expand_dims(ensemble_preds[i], 1), [1, len(adv_pairs)])
    adv_rank = np.argsort(np.argsort(-(mul_delta_preds_p + mul_preds_p).T))[np.arange(len(adv_pairs)), adv_pairs]
    advs_rank[sample_idx, sample_orig_rank] = adv_rank
    
    #4d. evaluate cheating
    suc_orig_rank = []
    suc_adv_rank = []
    for j in range(len(adv_pairs)):
        if adv_rank[j] >= K:
            continue
        preds_adv = ensemble_preds + np.add.reduce([X_csr.dot(H_inv.dot(delta_X_train_t_y[:, j])).reshape(-1, num_paper).T for (X_csr, H_inv, delta_X_train_t_y) in zip(X_csr_s, H_inv_s, delta_X_train_t_y_s)])
        assignment = assign_IP(preds_adv, K=K, advs_rank=None)
        if assignment[i, adv_pairs[j]] == 1:
            suc_ind[sample_idx, j] = 1
            suc_orig_rank.append(sample_orig_rank[j])
            suc_adv_rank.append(adv_rank[j])
        del preds_adv
    logger.info("paper {}, total successfull cheating number: {}, successfull cheating orig rank: {}, successfull cheating adv rank: {}".format(sample_idx, (suc_ind > 0).sum(), suc_orig_rank, suc_adv_rank))
    #4f. clean vars
    del X_p_H_inv_s
    del sol, delta_y, collusion_rev, delta_X_train_t_y_s, mul_delta_preds_p, mul_preds_p, adv_rank
    del suc_orig_rank
    
logger.info("start!")
p = ThreadPool(args.num_threads)
logger.info(f"num_threads: {args.num_threads}")
np.random.seed(args.seed)
sampled_papers = np.random.permutation(num_paper)[:args.num_sample_paper]
_ = list(tqdm(p.imap(thread_attack, zip(np.arange(args.num_sample_paper), sampled_papers)), total=args.num_sample_paper, desc='thread_attack'))

logger.info(f"start saving {output_name}")
np.savez(output_name, advs_rank=advs_rank, suc_ind=suc_ind, advs_sol=advs_sol, advs_collusion=advs_collusion, advs_rev=advs_rev, advs_original_rank=advs_original_rank, sampled_papers=sampled_papers)
logger.info(f"end saving {output_name}")
