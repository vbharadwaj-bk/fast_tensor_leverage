import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
import sys
import os
import json
from importlib import reload
import pickle
import cppimport.import_hook
from cpp_ext.als_module import Tensor, LowRankTensor, ALS

# Now let's test the efficient sampler... whatever happened there...
#reload(cpp_ext.efficient_krp_sampler)
from cpp_ext.efficient_krp_sampler import CP_ALS

data = None
with open('data/lstsq_problems.pickle', 'rb') as handle:
    data = pickle.load(handle)
    
problem = data[-1]

sigma_lhs = problem['sigma_lhs']
sigma_rhs = problem['sigma_rhs']

lhs = problem['lhs']
rhs = problem['rhs']

I = lhs[0].shape[0]
height = I * I

r = lhs[0].shape[1]
lhs = np.einsum('r,ir,jr->ijr', sigma_lhs, lhs[0], lhs[1])
lhs = lhs.reshape((height, r))
rhs = np.einsum('r,ir,jr,kr->ijk', sigma_rhs, rhs[0], rhs[1], rhs[2]).reshape(height, I)

res = la.lstsq(lhs, rhs, rcond=None)
print(la.norm(lhs @ problem['true_soln'].T - rhs))

J = 10000

# =========================================================
# Now, we will try to solve the system using preconditioning

lhs = problem['lhs']
rhs = problem['rhs']

lhs = np.einsum('ir,jr->ijr', lhs[0], lhs[1])
lhs = lhs.reshape((height, r))
lhs = lhs @ np.diag(sigma_lhs)
rhs = np.einsum('r,ir,jr,kr->ijk', sigma_rhs, rhs[0], rhs[1], rhs[2]).reshape(height, I)

leverage_scores = np.einsum('ij,jk,ik->i', lhs, la.pinv(lhs.T @ lhs), lhs)
leverage_scores /= np.sum(leverage_scores)
J = 10000

samples = np.random.choice(len(leverage_scores), size=(J), p=leverage_scores)

lhs_ds = lhs[samples]
rhs_ds = rhs[samples]
weights = np.sqrt(leverage_scores[samples] * J)
weights = 1.0 / weights

lhs_ds = np.diag(weights) @ lhs_ds
rhs_ds = np.diag(weights) @ rhs_ds

res = la.lstsq(lhs_ds, rhs_ds, rcond=None)
res = res[0].T @ np.diag(sigma_lhs ** -1)

print(la.norm(lhs @ np.diag(sigma_lhs) @ res.T - rhs))

# Hmmm... looks like there is a problem with the sampler...
# =========================================================
lhs = problem['lhs']
lhs = np.einsum('ir,jr->ijr', lhs[0], lhs[1])
lhs = lhs.reshape((height, r))

N = 3
sampler = CP_ALS(J, r, problem['lhs'])
samples = np.zeros((N, J), dtype=np.uint64)
h = np.zeros((J, r), dtype=np.double)

sampler.KRPDrawSamples(problem['j'], samples, h)

linear_idxs = samples[0] * I + samples[1]

plt.plot(np.bincount(linear_idxs.astype(np.int64)) / J)
plt.plot(leverage_scores)

lhs_ds = h # lhs[linear_idxs]
rhs_ds = rhs[linear_idxs]
weights = np.sqrt(leverage_scores[linear_idxs] * J)
weights = 1.0 / weights

lhs_ds_unweighted = lhs_ds.copy()

lhs_ds = np.diag(weights) @ lhs_ds
rhs_ds = rhs_ds

g_pinv = np.zeros((r, r), dtype=np.double)
sampler.get_G_pinv(g_pinv)

def symmetrize(buf):
    return np.triu(buf, 1) + np.triu(buf, 1).T + np.diag(np.diag(buf))

def gram(x):
    return x.T @ x

g_pinv = symmetrize(g_pinv)
g_pinv_groundtruth = la.pinv(lhs_ds.T @ lhs_ds) 

lhs_ds = np.diag(weights) @ lhs_ds

print(g_pinv_groundtruth)
print(g_pinv)
g_test = la.pinv(gram(problem['lhs'][0]) * gram(problem['lhs'][1]))
print(g_test)

g_pinv_test = la.pinv(gram(problem['lhs'][0]) * gram(problem['lhs'][1]))
#print(g_pinv_test)

res = (rhs_ds.T @ lhs_ds @ g_pinv_groundtruth).T
#res = la.lstsq(lhs_ds, rhs_ds, rcond=None)[0]
res = res.T @ np.diag(sigma_lhs ** -1)

print(la.norm(lhs @ np.diag(sigma_lhs) @ res.T - rhs))

lhs_ten = LowRankTensor(r, problem['lhs'])
rhs_ten = LowRankTensor(r, J, 10000, problem['rhs'])

als = ALS(lhs_ten, rhs_ten)
als.initialize_ds_als(J)
als.execute_ds_als_update(problem['j'], False, False, lhs_ds, samples)

linear_idxs = samples[0] * I + samples[1]
lhs_ds = lhs[linear_idxs]
rhs_ds = rhs[linear_idxs]

weights = np.sqrt(leverage_scores[linear_idxs] * J)
weights = 1.0 / weights

lhs_ds = np.diag(weights) @ lhs_ds
rhs_ds = np.diag(weights) @ rhs_ds

res = la.lstsq(lhs_ds, rhs_ds, rcond=None)[0].T
res = res @ np.diag(sigma_lhs ** -1)

#res = problem['lhs'][problem['j']]
print("Final Test Solution")
print(la.norm(lhs @ np.diag(sigma_lhs) @ res.T - rhs))