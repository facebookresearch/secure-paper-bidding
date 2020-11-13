# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from ortools.graph import pywrapgraph
import numpy as np
import os
from scipy import sparse

def get_global_variable():
    #num_paper, num_reviewer, input_dir (data & its cache directory), max_pc_quota, hashed_ratio
    return 2446, 2483, './data/', 6, 0.01

def assign_IP(scores, K=None, advs_rank=None):
    num_paper, num_reviewer, input_dir, max_pc_quota, hashed_ratio = get_global_variable()
    a = 1e-6
    (num_paper, num_reviewer) = scores.shape
    scores = scores.copy()
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    if K is None:
        for i in range(num_paper):
            for j in range(num_reviewer):
                min_cost_flow.AddArcWithCapacityAndUnitCost(i, j + num_paper, 1, -int(scores[i, j] / a))
    else:
        for i in range(num_paper):
            top_K_rev = np.argsort(-scores[i])[:K]
            for rank, j in enumerate(top_K_rev):
                if advs_rank is not None:
                    if advs_rank[i, rank] >= K:
                        continue
                min_cost_flow.AddArcWithCapacityAndUnitCost(i, j.item() + num_paper, 1, -int(scores[i, j] / a))
    for j in range(num_reviewer):
        min_cost_flow.AddArcWithCapacityAndUnitCost(j + num_paper, num_paper + num_reviewer, max_pc_quota, 0)
    for i in range(num_paper):
        min_cost_flow.SetNodeSupply(i, 3)
    min_cost_flow.SetNodeSupply(num_paper + num_reviewer, - 3 * num_paper)
    min_cost_flow.Solve()
    assignment = np.zeros([num_paper, num_reviewer])
    for i in range(0, min_cost_flow.NumArcs()):
        if min_cost_flow.Head(i) == num_paper + num_reviewer:
            continue
        if min_cost_flow.Flow(i) > 0:
            assignment[min_cost_flow.Tail(i), min_cost_flow.Head(i) - num_paper] = 1
    return assignment

def load_y(hashed_ratio, logger, subsample_max, seed=0):
    num_paper, num_reviewer, input_dir, max_pc_quota, hashed_ratio = get_global_variable()
    
    y = np.load(input_dir + '/labels_{}_seed_{}.npy'.format(hashed_ratio, seed))
    y_train = y.copy()
    y = y.reshape(-1, num_paper).transpose()

    y_train_file = '{}/y_train_{}_subample_max_{}.npy'.format(input_dir, hashed_ratio, subsample_max)
    if os.path.exists(y_train_file):
        y_train = np.load(y_train_file)
    else:
        logger.info("computing y_train")
        if subsample_max > 0:
            np.random.seed(0)
            y_train = np.reshape(y_train, (int(y_train.shape[0] / num_paper), num_paper))
            for i in range(y_train.shape[0]):
                if (y_train[i] > 0).sum() > 0:
                    indices = np.argwhere(y_train[i] > 0)[:, 0]
                    np.random.shuffle(indices)
                    if indices.shape[0] > subsample_max:
                        y_train[i, indices[subsample_max:]] = 0
                if (y_train[i] < 0).sum() > 0:
                    indices = np.argwhere(y_train[i] < 0)[:, 0]
                    np.random.shuffle(indices)
                    if indices.shape[0] > subsample_max:
                        y_train[i, indices[subsample_max:]] = 0
            y_train = np.reshape(y_train, (num_reviewer * num_paper, -1))
        np.save(y_train_file, y_train)
        
    return y, y_train

def load_X_and_H_inv(hashed_ratio, seed, logger, lam):
    num_paper, num_reviewer, input_dir, max_pc_quota, hashed_ratio = get_global_variable()
    
    X_csr_file = '{}/hashed_features_{}_seed_{}.npz'.format(input_dir, hashed_ratio, seed)
    X_csr = sparse.load_npz(X_csr_file)
        
    hessian_inv_file = '{}/hessian_inv_{}_seed_{}_lam_{}.npy'.format(input_dir, hashed_ratio, seed, lam)
    if os.path.exists(hessian_inv_file):
        H_inv = np.load(hessian_inv_file)
    else:
        logger.info("computing hessian inverse")
        hessian_file = '{}/hessian_{}_seed_{}.npy'.format(input_dir, hashed_ratio, seed)
        if os.path.exists(hessian_file):
            H = np.load(hessian_file)
        else:
            logger.info("computing hessian")
            H = np.asarray(X_csr.transpose().dot(X_csr).todense())
            np.save(hessian_file, H)
        H_inv = np.linalg.inv(2 * H / X_csr.shape[0] + lam * np.eye(X_csr.shape[1])) 
        np.save(hessian_inv_file, H_inv)
        del H
        
    return X_csr, H_inv

def load_preds(X_csr, y_train, H_inv, hashed_ratio, seed, logger, lam, subsample_max):
    num_paper, num_reviewer, input_dir, max_pc_quota, hashed_ratio = get_global_variable()
    preds_file = input_dir + '/preds_hashed_features_{}_seed_{}_lam_{}_subsample_max_{}.npy'.format(hashed_ratio, seed, lam, subsample_max)
    if os.path.exists(preds_file):
        preds = np.load(preds_file)
    else:
        logger.info("compute preds!")
        preds = X_csr.dot(H_inv.dot(X_csr.transpose().dot(y_train))).reshape(num_reviewer, num_paper).transpose() * 2 / X_csr.shape[0]
        np.save(preds_file, preds)
    prev_rank = np.argsort(-preds)
    return preds
    
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def solve_ip_adv(v, opt_max=True, subsample_max=60):
    num_paper, num_reviewer, input_dir, max_pc_quota, hashed_ratio = get_global_variable()
    if len(v.shape) == 1:
        v = np.expand_dims(v, 1)
    n, d = v.shape
    if not opt_max:
        v = -v
    v = np.hstack([v, np.zeros([n, subsample_max])])
    sort_v = np.argsort(v)
    w = np.zeros([n, d + subsample_max])
    w[np.expand_dims(np.arange(n), 1), sort_v[:, -subsample_max:]] = 3
    return w[:, :d]
