SPARSE_AUTO_DENSIFY=0
import sparse
import numpy as np

import tensorly as tl
tl.set_backend('numpy.sparse')

#from tensorly.contrib.sparse.decomposition import parafac as parafac 
from tensorly.decomposition import parafac as parafac 
from tensorly.contrib.sparse.cp_tensor import cp_to_tensor
#import tensorly.contrib.sparse as stl

def test_sparse_parafac():
    """Test for sparse parafac"""
    # Make sure the algorithm stays sparse. This will run out of memory on
    # most machines if the algorithm densifies.
    random_state = 1234
    rank = 3
    factors = [
        sparse.random((28620, rank), random_state=random_state),
        sparse.random((1403, rank), random_state=random_state),
        sparse.random((1403, rank), random_state=random_state),
    ]
    weights = np.ones(rank)
    tensor = cp_to_tensor((weights, factors))
    print(tensor)
    _ = parafac(
        tensor, rank=rank, init="random", n_iter_max=1, random_state=random_state
    )

if __name__=='__main__':
    test_sparse_parafac()