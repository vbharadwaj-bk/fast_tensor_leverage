import numpy as np
import numpy.linalg as la
import tensorly as tl
from tensorly.base import matricize
from tensorly.tt_tensor import validate_tt_rank, TTTensor
from synthetic_data import *


def draw_samples_tns_tt(G, n, J2):
    N = len(G)
    C1 = [None] * N
    C1_unf = [None] * N
    C2 = [None] * N
    C2_vec = [None] * N

    # perm_vec = list(range(n, n + 1)) + list(range(n + 1, N + 1)) + list(range(1, n))
    idx_ordering = []
    for m in range(N):
        idx_ordering.append(list(range(m + 1, N)) + list(range(m)))

    for j in range(N):
        shape = G[j].shape
        if j != n:
            # G[j].transpose(G[j], perm_vec)
            G_2 = matricize(G[j], [1], [0, 2])
            temp = np.transpose(
                np.reshape(tl.tenalg.khatri_rao([G_2.T, G_2.T]), (shape[2], shape[0], shape[2], shape[0], shape[1])),
                [1, 3, 4, 0, 2])
            C1[j] = temp.reshape(shape[0] ** 2, shape[1], shape[2] ** 2)
            C1_unf[j] = matricize(C1[j], [1], [0, 2])

            temp = G_2.T @ G_2
            temp = np.reshape(temp, [shape[2], shape[0], shape[2], shape[0]])
            temp = np.transpose(temp, [1, 3, 0, 2])
            C2[j] = np.reshape(temp, [shape[0] ** 2, shape[2] ** 2])
            C2_vec[j] = C2[j].flatten('F').T

    AmTAm = C2[idx_ordering[n][0]]
    for j in range(1, N-1):
        AmTAm = AmTAm @ C2[idx_ordering[n][j]]
    shape = G[n].shape
    AmTAm = np.transpose(AmTAm.reshape([shape[2], shape[2], shape[0], shape[0]]), [2, 0, 3, 1])
    AmTAm = AmTAm.reshape([shape[0] * shape[2], shape[0] * shape[2]])
    phi = la.pinv(AmTAm)
    temp = np.reshape(phi, [shape[0], shape[2], shape[0], shape[2]])
    temp = temp.transpose([0, 2, 1, 3])
    C2[n] = temp.reshape(shape[0] ** 2, shape[2] ** 2)

    samples = np.zeros((J2, N), dtype=int)
    sqrt_probs = np.ones(J2)

    if n == 0:
        first_idx = 1
    else:
        first_idx = 0
    first_idx_flag = True

    # Main loop for drawing all samples
    for samp in range(J2):
        # Compute P(i_m | (i_j)_{j < m, ~= n}) for each m (~=n) and draw i_m
        for m in range(first_idx, N):
            if m != n:
                # Compute conditional probability vector
                idx = idx_ordering[m]
                if idx[0] >= m or idx[0] == n:
                    M = C2[idx[0]]
                else:
                    sz = C1[idx[0]].shape[:3]
                    M = np.reshape(C1[idx[0]][:, samples[samp, idx[0]], :], (sz[0], sz[2]), order='F')
                for j in range(1, len(idx)):
                    if idx[j] >= m or idx[j] == n:
                        M = np.matmul(M, C2[idx[j]])
                    else:
                        sz = C1[idx[j]].shape[:3]
                        M = np.matmul(M, np.reshape(C1[idx[j]][:, samples[samp, idx[j]], :], (sz[0], sz[2]), order='F'))
                common_terms = M.T

                common_terms_vec = common_terms.flatten(order='F')
                prob_m = np.matmul(C1_unf[m], common_terms_vec)
                prob_m = prob_m / np.sum(prob_m)

                if first_idx_flag:
                    # Draw from the vector
                    prob_m = np.maximum(prob_m, 0)
                    if np.isnan(prob_m).sum() > 0:
                        raise ValueError("Probability vector contains NaN")

                    samples[:, m] = np.random.choice(len(prob_m), J2, replace=True, p=prob_m)

                    # Update probability vector
                    sqrt_probs = np.sqrt(prob_m[samples[:, m]])

                    first_idx_flag = False
                else:
                    # Draw from the vector
                    prob_m = np.maximum(prob_m, 0)
                    if np.isnan(prob_m).sum() > 0:
                        raise ValueError("Probability vector contains NaN")
                    samples[samp, m] = np.random.choice(len(prob_m), p=prob_m)

                    # Update probability vector
                    sqrt_probs[samp] = sqrt_probs[samp] * np.sqrt(prob_m[int(samples[samp, m])])

    return samples, sqrt_probs






