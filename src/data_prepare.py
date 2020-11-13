# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from scipy import sparse
from scipy import optimize
import torch
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='the seed for hashing')
parser.add_argument('--hashed_ratio', default=0.01, type=float, help='the ratio for hashing')
parser.add_argument('--input_dir', default="./data/raw_data/", type=str, help='the directory of raw tensor')
parser.add_argument('--output_dir', default="./data/", type=str, help='the directory for processed data')

args = parser.parse_args()
hashed_ratio = args.hashed_ratio
seed = args.seed

def dict_to_sparse(dict_sp_t):
    return torch.sparse.FloatTensor(dict_sp_t["indices"], dict_sp_t["values"], dict_sp_t["size"])
def torch_sparse_to_numpy(t_sparse):
    return sparse.coo_matrix((t_sparse._values().numpy(), (t_sparse._indices().numpy()[0], t_sparse._indices().numpy()[1])), t_sparse.size())
def sparse_tensor(np_ar):
    tc_ar = torch.from_numpy(np_ar)
    tensor_size = tc_ar.size()
    indices = tc_ar.nonzero()
    return torch.sparse.FloatTensor(indices.t(), tc_ar[indices.split(1, dim=1)].squeeze(1), tensor_size)

def repeat_sparse(sp_t, repeat_num, repeat_order):
    indices = sp_t._indices()
    size = sp_t.size()
    values = sp_t._values()
    if repeat_order == 0:
        new_indices_0 = torch.cat(list(indices[0] + i * size[0] for i in range(repeat_num)))
        new_indices_1 = torch.cat(list(indices[1] for i in range(repeat_num)))
    else:
        new_indices_0 = torch.cat(list(indices[0] * repeat_num + i for i in range(repeat_num)))
        new_indices_1 = torch.cat(list(indices[1] for i in range(repeat_num)))
    new_size = torch.Size([size[0] * repeat_num, size[1]])
    new_indices = torch.cat([new_indices_0.unsqueeze(0), new_indices_1.unsqueeze(0)], dim=0)
    new_values = values.repeat(repeat_num)
    return torch.sparse.FloatTensor(new_indices, new_values, new_size)

def spouter(A,B):
    N,L = A.shape
    N,K = B.shape
    drows = zip(*(np.split(x.data,x.indptr[1:-1]) for x in (A,B)))
    data = [np.outer(a,b).ravel() for a,b in drows]
    del drows
    data = np.concatenate(data)
    irows = zip(*(np.split(x.indices,x.indptr[1:-1]) for x in (A,B)))
    indices = [np.ravel_multi_index(np.ix_(a,b),(L,K)).ravel() for a,b in irows]
    del irows
    indptr = np.fromiter(itertools.chain((0,),map(len,indices)),int).cumsum()
    indices = np.concatenate(indices)
    return (data, indices, indptr)

def quadratic_features_sparse(sp_t1, sp_t2):
    sp_t1_scipy = sparse.csr_matrix((sp_t1._values().numpy(), (sp_t1._indices()[0].numpy(), sp_t1._indices()[1].numpy())), shape=(sp_t1.size()[0], sp_t1.size()[1]))
    sp_t2_scipy = sparse.csr_matrix((sp_t2._values().numpy(), (sp_t2._indices()[0].numpy(), sp_t2._indices()[1].numpy())), shape=(sp_t2.size()[0], sp_t2.size()[1]))
    N,L = sp_t1_scipy.shape
    N,K = sp_t2_scipy.shape
    del sp_t1
    del sp_t2
    (data, indices, indptr) = spouter(sp_t1_scipy, sp_t2_scipy)
    csr = sparse.csr_matrix((data, indices, indptr), (N, L*K))
    coo = sparse.coo_matrix(csr)
    indices = torch.cat([torch.from_numpy(coo.row).unsqueeze(0), torch.from_numpy(coo.col).unsqueeze(0)]).long()
    values = torch.from_numpy(csr.data)
    size = torch.Size(csr.shape)
    return torch.sparse.FloatTensor(indices, values, size)

def hash_torch_sparse_data(t_sparse, ratio=0.01, seed=0):
    indices, values, size = t_sparse._indices().numpy(), t_sparse._values().numpy(), list(t_sparse.size())
    np.random.seed(seed)
    hashed_size = [size[0], int(ratio * size[1])]
    if ratio == 1:
        hash_map = np.arange(size[1]).astype(np.int)
    else:
        hash_map = np.random.randint(0, hashed_size[1], (size[1],))
    indices[1] = hash_map[indices[1]]
    sparse_tensor = sparse.coo_matrix((values, (indices[0], indices[1])), hashed_size)
    sparse_tensor.sum_duplicates()
    return sparse_tensor

num_reviewer, num_paper = 2483, 2446

#load_raw_tensor
print("load raw tensor")
tensor_data = torch.load("{}/tensor_data.pl".format(args.input_dir))
r_subject = dict_to_sparse(tensor_data["r_subject"])
p_subject = dict_to_sparse(tensor_data["p_subject"])
p_title = dict_to_sparse(tensor_data["p_title"])
labels = tensor_data["label"]
tpms = tensor_data["tpms"]

#intersect subject area between reviewer and paper
print("generate intersect subject area between reviewer and paper")
r_subject_dense = r_subject.to_dense().numpy()
p_subject_dense = p_subject.to_dense().numpy()
int_subject_r_p = np.asarray(list(p_subject_dense[j] * r_subject_dense[i] for i in range(len(r_subject_dense)) for j in range(len(p_subject_dense))))
int_subject_r_p = sparse_tensor(int_subject_r_p)

#quantized tpms vector
print("generate quantized tpms vector")
tpms = tpms.reshape(-1).numpy()
tpms_round = (tpms * 10).astype(int)
tpms_vec = np.zeros([num_reviewer * num_paper, 12])
tpms_vec[:, -1] = tpms
for i in range(num_reviewer * num_paper):
    tpms_vec[i][tpms_round[i]] = tpms[i] 
    
#quadratic feature between quantized tpms vector and int_subject_r_p
print("generate quadratic feature between quantized tpms vector and int_subject_r_p")
q_tpms_subject_int = quadratic_features_sparse(int_subject_r_p, sparse_tensor(tpms_vec))
q_tpms_subject_int_hash = hash_torch_sparse_data(q_tpms_subject_int, ratio=hashed_ratio, seed=seed + 0)

#quadratic feature between p_title and int_subject_r_p
print("generate quadratic feature between p_title and int_subject_r_p")
q_p_title_subject_int = quadratic_features_sparse(int_subject_r_p, repeat_sparse(p_title, num_reviewer, 1))
q_p_title_subject_int_hash = hash_torch_sparse_data(q_p_title_subject_int, ratio=hashed_ratio, seed=seed + 1)

#quadratic feature between p_subject and r_subject
print("generate quadratic feature between p_subject and r_subject")
q_p_subject_r_subject = quadratic_features_sparse(repeat_sparse(r_subject, num_paper, 1), repeat_sparse(p_subject, num_reviewer, 0))
q_p_subject_r_subject_hash = hash_torch_sparse_data(q_p_subject_r_subject, ratio=hashed_ratio, seed=seed + 2)

#quadratic feature between p_title and r_subject
print("generate quadratic feature between p_title and r_subject")
q_p_title_r_subject = quadratic_features_sparse(repeat_sparse(r_subject, num_paper, 1), repeat_sparse(p_title, num_reviewer, 0))
q_p_title_r_subject_hash = hash_torch_sparse_data(q_p_title_r_subject, ratio=hashed_ratio, seed=seed + 3)

#repeat r_subject, p_subject and p_title
print("repeat r_subject, p_subject and p_title")
r_subject_rep =torch_sparse_to_numpy(repeat_sparse(r_subject, num_paper, 1))
p_subject_rep = torch_sparse_to_numpy(repeat_sparse(p_subject, num_reviewer, 0))
p_title_rep = torch_sparse_to_numpy(repeat_sparse(p_title, num_reviewer, 0))
tpms_vec = sparse.coo_matrix(tpms_vec)

#cat all features
print("cat all features")
X = sparse.hstack([r_subject_rep, p_subject_rep, p_title_rep, torch_sparse_to_numpy(int_subject_r_p), tpms_vec, q_tpms_subject_int_hash, q_p_title_subject_int_hash, q_p_subject_r_subject_hash, q_p_title_r_subject_hash, np.ones([r_subject_rep.shape[0], 1])])
X_csr = X.tocsr()

#start to save
print("start to save")
sparse.save_npz("{}/hashed_features_{}_seed_{}.npz".format(args.output_dir, hashed_ratio, seed), X_csr)
np.save("{}/labels_{}_seed_{}.npy".format(args.output_dir, hashed_ratio, seed), labels.reshape(-1))


