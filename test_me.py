import numpy as np
import cppimport.import_hook
from cpp_ext.als_module import Sampler

if __name__=='__main__':
    # Take J=10,000 samples from the KRP of N=4 matrices,
    # each with dimensions I x R, where I = 1000 and R = 8.
    I, N, J, R = 1000, 4, 10000, 8
    matrices = [np.random.normal(size=(I, R)) for i in range(N)]

    sampler = Sampler(matrices, J, R, "efficient")
    samples = np.zeros((N, J), dtype=np.uint64)
    sampler.KRPDrawSamples(N+1, samples)
