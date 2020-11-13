# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from scipy import sparse
import torch
import argparse
from logger import set_logger
import os
from utils import assign_IP, get_global_variable

parser = argparse.ArgumentParser()
parser.add_argument('--hashed_ratio', default=0.01, type=float, help='the ratio for hashing')
parser.add_argument('--output_dir', default='./results/', type=str, help='output dir')
parser.add_argument('--input_dir', default='./data/', type=str, help='input dir')
parser.add_argument('--lam', default=1e-4, type=float, help='L2 regularization')
parser.add_argument('--seed', default=0, type=int, help='seed for hashing')
parser.add_argument('--subsample_max', default=60, type=int, help='max subsampled training labels')
parser.add_argument('--K', default=50, type=int, help='top K reviewers per paper considered for assignment')

args = parser.parse_args()
logger = set_logger("assignment")
logger.info(args)

def precision_evaluate(w, X, y):
    preds = X.dot(w)
    preds = np.reshape(preds, (num_reviewer, int(X.shape[0] / num_reviewer)))
    y = np.reshape(y, (num_reviewer, int(X.shape[0] / num_reviewer)))
    indices = np.argsort(preds, 1)
    prec_at_k = np.zeros((preds.shape[0], 10))
    for i in range(preds.shape[0]):
        out = y[i, indices[i, -1:-11:-1]] == 1
        prec_at_k[i] = np.cumsum(out).astype(float) / np.arange(1, 11)
    logger.info("top10papers: " + repr(prec_at_k.mean(0)))
    prec_top_paper = prec_at_k.mean(0)
    
    preds = preds.transpose()
    y = y.transpose()
    indices = np.argsort(preds, 1)
    prec_at_k = np.zeros((preds.shape[0], 10))
    for i in range(preds.shape[0]):
        out = y[i, indices[i, -1:-11:-1]] == 1
        prec_at_k[i] = np.cumsum(out).astype(float) / np.arange(1, 11)
    logger.info("top10reviewers: " + repr(prec_at_k.mean(0)))
    prec_top_reviewer = prec_at_k.mean(0)
    return prec_top_paper, prec_top_reviewer

def assignment_evaluate(assignment, y, tpms):
    frac_of_pos = (assignment * (y > 0)).sum() / (assignment > 0).sum()
    average_scores = (assignment * y).sum() / (assignment > 0).sum()
    average_tpms = (assignment * tpms).sum() / (assignment > 0).sum()
    average_max_tpms = (assignment * tpms).max(1).sum() / assignment.shape[0]
    logger.info("Frac. of pos.: {}, Avg. bids score: {}, Avg. TPMS: {}, Avg. max. TPMS: {}".format(frac_of_pos, average_scores, average_tpms, average_max_tpms))
          
#1. load data
logger.info("loading data")
num_paper, num_reviewer, _, _, _ = get_global_variable()
y = np.load(args.input_dir + 'labels_{}_seed_{}.npy'.format(args.hashed_ratio, args.seed))
X = sparse.load_npz(args.input_dir + 'hashed_features_{}_seed_{}.npz'.format(args.hashed_ratio, args.seed))
    
#2. subsample y
y_train = y.copy()
logger.info("subsample y")
logger.info("positive in train: {}, negative in train: {}".format(len(np.nonzero(y_train > 0)[0]), len(np.nonzero(y_train <= 0)[0])))
if args.subsample_max > 0:
    np.random.seed(0)
    y_train = np.reshape(y_train, (int(y_train.shape[0] / num_paper), num_paper))
    for i in range(y_train.shape[0]):
        if (y_train[i] > 0).sum() > 0:
            indices = np.argwhere(y_train[i] > 0)[:, 0]
            np.random.shuffle(indices)
            if indices.shape[0] > args.subsample_max:
                y_train[i, indices[args.subsample_max:]] = 0
        if (y_train[i] < 0).sum() > 0:
            indices = np.argwhere(y_train[i] < 0)[:, 0]
            np.random.shuffle(indices)
            if indices.shape[0] > args.subsample_max:
                y_train[i, indices[args.subsample_max:]] = 0
    y_train = np.reshape(y_train, (X.shape[0],))

#3. compute hessian and its inverse
hessian_file = '{}/hessian_{}_seed_{}.npy'.format(args.input_dir, args.hashed_ratio, args.seed)
if os.path.exists(hessian_file):
    H = np.load(hessian_file)
else:
    logger.info("computing hessian")
    H = np.asarray(X.transpose().dot(X).todense())
    np.save(hessian_file, H)

d = X.shape[1]
logger.info("# dim: " + str(d))
hessian_inv_file = '{}/hessian_inv_{}_seed_{}_lam_{}.npy'.format(args.input_dir, args.hashed_ratio, args.seed, args.lam)
if os.path.exists(hessian_inv_file):
    H_inv = np.load(hessian_inv_file)
else:
    logger.info("computing hessian inv")
    H_inv = np.linalg.inv(2 * H / X.shape[0] + args.lam * np.eye(X.shape[1]))
    np.save(hessian_inv_file, H_inv)

#4. compute weight vector
w = H_inv.dot(2 * X.transpose().dot(y_train) / X.shape[0])

#5. evaluate precision
prec_results = np.zeros([2, 10])
logger.info("precision: ")
prec_results[0], prec_results[1] = precision_evaluate(w, X, np.sign(y_train))

#6. evaluate assignment
logger.info("compute assignment with K={}: ".format(args.K))
preds = X.dot(w).reshape([num_reviewer, num_paper]).transpose()
y = y.reshape([num_reviewer, num_paper]).transpose()
tpms = torch.load("{}/raw_data/tensor_data.pl".format(args.input_dir))["tpms"].numpy().reshape([num_reviewer, num_paper]).transpose()
assignment = assign_IP(preds, args.K)
assignment_evaluate(assignment, y, tpms)

logger.info("saving parameters...")
np.savez(args.output_dir + '/assignment/learned_weights_hashed_ratio_{}_subsample_{}_seed_{}_lam_{}_K_{}.npy'.format(args.hashed_ratio, args.subsample_max, args.seed, args.lam, args.K), w=w, prec_results=prec_results, assignment=assignment)
