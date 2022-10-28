import numpy as np
import numpy.linalg as la
from scipy.special import rel_entr 
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

def test_on_explicit_pmf(tree, masses, sample_count):
    '''
    Test the partition tree sampling on a provided explicit PMF.
    Computes m(v) ahead of time on all nodes, then draws the 
    specified number of samples.
    '''
    m_vals = np.zeros(tree.node_count)
    for i in reversed(range(tree.node_count)):
        if tree.is_leaf(i):
            start, end = tree.S(i)
            m_vals[i] = np.sum(masses[start:end])
        else:
            m_vals[i] = m_vals[tree.L(i)] + m_vals[tree.R(i)]

    m = lambda c : m_vals[c]
    q = lambda c : masses[tree.S(c)[0] : tree.S(c)[1]].copy()

    result = np.zeros(tree.n, dtype=np.int32)
    for i in range(sample_count):
        sample = tree.PTSample(m, q)
        result[sample] += 1

    return result / sample_count

def test_tree(tree, sample_count):
    '''
    Test the partition tree with several distributions
    '''
    def run_pmf_test(pmf):
        tree_samples = test_on_explicit_pmf(pmf, sample_count) 
        pmf_normalized = pmf / np.sum(pmf)
        numpy_samples = np.random.multinomial(sample_count, pmf_normalized) / sample_count
        return pmf_normalized, tree_samples, numpy_samples 

    uniform = np.ones(tree.n)
    exponential_decay = np.ones(tree.n)
    for i in range(1, tree.n):
        exponential_decay[i] = exponential_decay[i-1] / 2
    
    return [run_pmf_test(uniform), run_pmf_test(exponential_decay)]

def test_sampler(sampler_class):
    N = 4
    I = 8
    R = 5
    F = 3
    U = [np.random.rand(I, R) for i in range(N)]
    sampler = sampler_class(U, [F] * N)

    j = 3
    J = 120000

    samples = np.array(sampler.KRPDrawSamples_scalar(j, J), dtype=np.uint64)
    hist = np.bincount(samples.astype(np.int64))
    krp_materialized = krp(U[:-1])

    krp_q = la.qr(krp_materialized)[0]

    krp_norms = la.norm(krp_q, axis=1) ** 2
    #plt.plot(krp_norms / np.sum(krp_norms), label="Ground Truth PDF")
    #plt.plot(hist / np.sum(hist), label="PDF of Our Sampler")
    #plt.legend()

    dist_err = rel_entr(hist / np.sum(hist), krp_norms / np.sum(krp_norms))
    print(f"Relative Entropy: {np.sum(dist_err)}")

if __name__=='__main__':
    from krp_sampler_opt3 import EfficientKRPSampler
    test_sampler(EfficientKRPSampler)