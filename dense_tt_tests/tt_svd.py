import numpy as np
import scipy
import numpy.linalg as la
# from sklearn.utils.extmath import randomized_svd
import tensorly as tl
from tensorly.tt_tensor import validate_tt_rank, TTTensor

# from sklearn.utils.extmath import randomized_svd

""" This code is slightly modified and borrowed from svd and tt scripts from Tensorly package """


def svd_checks(matrix, n_eigenvecs=None):
    """Runs common checks to all of the SVD methods."""

    # Check that matrix is... a matrix!
    if np.ndim(matrix) != 2:
        raise ValueError(f"matrix be a matrix. matrix.ndim is {np.ndim(matrix)} != 2")

    dim_1, dim_2 = tl.shape(matrix)
    min_dim, max_dim = min(dim_1, dim_2), max(dim_1, dim_2)

    if n_eigenvecs is None:
        n_eigenvecs = max_dim

    if n_eigenvecs > max_dim:
        warnings.warn(
            f"Trying to compute SVD with n_eigenvecs={n_eigenvecs}, which is larger "
            f"than max(matrix.shape)={max_dim}. Setting n_eigenvecs to {max_dim}."
        )
        n_eigenvecs = max_dim

    return n_eigenvecs, min_dim, max_dim


def truncated_svd(matrix, n_eigenvecs=None, **kwargs):
    """Computes a truncated SVD on `matrix` using the backends's standard SVD"""

    n_eigenvecs, min_dim, _ = svd_checks(matrix, n_eigenvecs=n_eigenvecs)
    full_matrices = True if n_eigenvecs > min_dim else False

    U, S, V = tl.svd(matrix, full_matrices=full_matrices)
    return U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]


def randomized_range_finder(A, n_dims, n_iter=1, random_state=None):
    """Computes an orthonormal matrix (Q) whose range approximates the range of A,  i.e., Q Q^H A ≈ A
    Notes
    -----
    This function is implemented based on Algorith 4.4 in `Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions`
    - Halko et al (2009)
    """
    rng = tl.check_random_state(random_state)
    dim_1, dim_2 = tl.shape(A)
    Q = tl.tensor(rng.normal(size=(dim_2, n_dims)), **tl.context(A))
    Q, _ = tl.qr(tl.dot(A, Q))

    # Perform power iterations when spectrum decays slowly
    A_H = tl.conj(tl.transpose(A))
    for i in range(n_iter):
        Q, _ = tl.qr(tl.dot(A_H, Q))
        Q, _ = tl.qr(tl.dot(A, Q))

    return Q


def randomized_svd(
        matrix,
        n_eigenvecs=None,
        n_oversamples=0,
        n_iter=1,
        random_state=None,
        **kwargs,
):
    """Computes a truncated randomized SVD.

    Notes
    -----
    This function is implemented based on Algorith 5.1 in `Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions`
    - Halko et al (2009)
    """
    n_eigenvecs, min_dim, max_dim = svd_checks(matrix, n_eigenvecs=n_eigenvecs)

    dim_1, dim_2 = tl.shape(matrix)
    n_dims = min(n_eigenvecs + n_oversamples, max_dim)

    # transpose matrix to keep the reduced matrix shape minimal
    matrix_T = tl.transpose(matrix)
    Q = randomized_range_finder(
        matrix_T, n_dims=n_dims, n_iter=n_iter, random_state=random_state
    )
    Q_H = tl.conj(tl.transpose(Q))
    matrix_reduced = tl.transpose(tl.dot(Q_H, matrix_T))
    U, S, V = truncated_svd(matrix_reduced, n_eigenvecs=n_eigenvecs)
    V = tl.dot(V, tl.transpose(Q))

    return U, S, V


def tensor_train_svd(input_tensor, rank, svd=None, verbose=False):
    rank = validate_tt_rank(np.shape(input_tensor), rank=rank)
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)

    unfolding = input_tensor
    factors = [None] * n_dim

    # Getting the TT factors up to n_dim - 1
    for k in range(n_dim - 1):
        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k] * tensor_size[k])
        unfolding = (tl.reshape(unfolding, (n_row, -1)))

        # SVD of unfolding matrix
        (n_row, n_column) = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])

        if svd == 'truncated_svd':
            U, S, V = truncated_svd(unfolding, n_eigenvecs=current_rank)
        else:
            U, S, V = randomized_svd(unfolding, n_eigenvecs=current_rank)
        rank[k + 1] = current_rank

        # Get kth TT factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k + 1]))
        if verbose is True:
            print(
                "TT factor " + str(k) + " computed with shape " + str(factors[k].shape)
            )

        # Get new unfolding matrix for the remaining factors
        unfolding = tl.reshape(S, (-1, 1)) * V

    # Getting the last factor
    (prev_rank, last_dim) = unfolding.shape
    factors[-1] = tl.reshape(unfolding, (prev_rank, last_dim, 1))

    if verbose is True:
        print(
            "TT factor "
            + str(n_dim - 1)
            + " computed with shape "
            + str(factors[n_dim - 1].shape)
        )

    return TTTensor(factors)
