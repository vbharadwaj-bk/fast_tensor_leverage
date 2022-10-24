import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from krp_sampler_opt0 import EfficientKRPSampler as SamplerOpt0

def krp(mats):
    if len(mats) == 1:
        return mats[0]
    else:
        running_mat = np.einsum('ik,jk->ijk', mats[0], mats[1]).reshape((mats[0].shape[0] * mats[1].shape[0], mats[0].shape[1]))
        
        for i in range(2, len(mats)):
            running_mat = np.einsum('ik,jk->ijk', running_mat, mats[i]).reshape((running_mat.shape[0] * mats[i].shape[0], mats[0].shape[1]))

        return running_mat

def test_sampler(sampler_class):
    N = 4
    I = 8
    R = 5
    F = 3
    U = [np.random.rand(I, R) for i in range(N)]
    sampler = sampler_class(U, [F] * N)

    j = 3
    J = 100000

    samples = sampler.KRPDrawSamples_scalar(j, J)
    hist = np.bincount(samples)
    krp_materialized = krp(U[:-1])

    krp_q = la.qr(krp_materialized)[0]

    krp_norms = la.norm(krp_q, axis=1) ** 2
    plt.plot(krp_norms / np.sum(krp_norms), label="Ground Truth PDF")
    plt.plot(hist / np.sum(hist), label="PDF of Our Sampler")
    plt.legend()

