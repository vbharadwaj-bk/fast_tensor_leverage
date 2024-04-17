import numpy as np
import numpy.linalg as la

def divide_and_roundup(n, m):
    return (n + m - 1) // m

def symmetrize(buf):
    return np.triu(buf, 1) + np.triu(buf, 1).T + np.diag(np.diag(buf))

def chain_had_prod(matrices):
    res = np.ones(matrices[0].shape)
    for mat in matrices:
        res *= mat
    return res

def inner_prod(U, V, sigma_U, sigma_V):
    elwise_prod = chain_had_prod([U[i].T @ V[i] for i in range(len(U))])
    elwise_prod *= np.outer(sigma_U, sigma_V)
    return np.sum(elwise_prod)

def compute_diff_norm(U, V, sigma_U, sigma_V):
    val = inner_prod(U, U, sigma_U, sigma_U) + inner_prod(V, V, sigma_V, sigma_V) - 2 * inner_prod(U, V, sigma_U, sigma_V)
    val = max(val, 0.0)
    return np.sqrt(val)
        
