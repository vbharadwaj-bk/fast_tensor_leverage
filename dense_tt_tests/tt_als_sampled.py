import numpy as np
import numpy.linalg as la
import tensorly as tl
from tensorly.base import matricize
from tensorly.tt_tensor import validate_tt_rank, TTTensor
from synthetic_data import *

def leverage_score_dist(matrix):
    """
    References
    ----------
    .. [1] P. Drineas, M. W. Mahoney, "RandNLA: randomized numerical linear algebra",
           Commun. ACM 59(6), pp. 80-90, 2016. DOI: 10.1145/2842602
    """
    U, S, _ = tl.svd(matrix, full_matrices=False)
    mat_dtype = tl.context(matrix)["dtype"]
    rank_cutoff = tl.max(S) * max(matrix.shape) * tl.eps(mat_dtype)
    num_rank = (
        int(tl.max(tl.where(S > rank_cutoff)[0])) + 1)  # int(...) needed for mxnet
    lev_score_dist = tl.sum(U[:, :num_rank] ** 2, axis=1) / tl.tensor(num_rank, dtype=mat_dtype)

    if tl.context(lev_score_dist)["dtype"] != tl.float64:
        lev_score_dist = tl.tensor(lev_score_dist, dtype=tl.float64)
        lev_score_dist /= tl.sum(lev_score_dist)

    return lev_score_dist

def tensor_train_als_sampled(tensor, rank, n_samples, nsweeps, init_method=None, tt_decomp=None):
    """
    References
    ----------
    .. [1] O. A. Malik, S. Becker, "A Sampling-Based Method for Tensor Ring
           Decomposition", Proceedings of the 38th International Conference on Machine
           Learning (ICML), PMLR 139:7400-7411, 2021.
    """
    shape = list(tensor.shape)
    rank = validate_tt_rank(shape, rank=rank)
    n_dim = len(shape)
    rng = np.random.default_rng()
    if isinstance(n_samples, int):
        n_samples = [n_samples] * n_dim

    # Create index orderings for computation of sketched design matrix
    idx_ordering = [
        [n for n in range(dim + 1, n_dim)] + [n for n in range(dim)]
        for dim in range(n_dim)
    ]
    # Randomly initialize decomposition cores
    if init_method == "gaussian" and tt_decomp is None:
        tt_decomp = [rng.normal(size=(rank[i], shape[i], rank[i + 1])) for i in range(n_dim)]
        for i in range(n_dim):
            tt_decomp[i] /= la.norm(tt_decomp[i])

    elif init_method is None and tt_decomp is not None:
        for i in range(n_dim):
            tt_decomp[i] /= la.norm(tt_decomp[i])

    # Compute initial sampling distributions
    sampling_probs = [None]
    for dim in range(1, n_dim):
        lev_score_dist = leverage_score_dist(matricize(tt_decomp[dim], [1], [0, 2]))
        sampling_probs.append(lev_score_dist)
    print(len(sampling_probs))
    print(sampling_probs[0].shape)
    print(shape[0])

    print(sampling_probs[1].shape)
    print(shape[1])

    print(sampling_probs[2].shape)
    print(shape[2])

    print(sampling_probs[3].shape)
    print(shape[3])

    # Main loop
    for iter in range(nsweeps):
        print(f"{iter} is running ... ")
        for dim in range(n_dim):
            print(len(sampling_probs))
            for n in range(n_dim):
                print(range(shape[n]))
                if n != dim:
                    print(sampling_probs[n].shape)
            # Randomly draw row indices
            print("Length of sampling_probs:", len(sampling_probs))
            samples = [
                rng.choice(
                    range(shape[n]),
                    size=(n_samples[dim]),
                    p=tl.to_numpy(sampling_probs[n]),
                )
                for n in range(n_dim)
                if n != dim
            ]
            # print(type(samples[0]))
            print(samples[0].shape)
            # Combine repeated samples
            samples_unq, samples_cnt = np.unique(samples, axis=1, return_counts=True)
            # print(samples_unq)
            samples_unq = samples_unq.tolist()
            samples_unq.insert(dim, slice(None, None, None))
            samples_unq = tuple(samples_unq)
            print(samples_unq)
            # print(samples_unq[1])
            # print(samples_unq)
            samples_cnt = tl.tensor(samples_cnt, **tl.context(tensor))

            # Compute row rescaling factors (see discussion in Sec 4.1 in paper by
            # Larsen & Kolda (2022), DOI: 10.1137/21M1441754)
            rescaling = tl.sqrt(samples_cnt / n_samples[dim])

            for n in range(n_dim):
                if n != dim:
                    # Converting samples_unq[n] to a tl.tensor is necessary for indexing
                    # to work with jax, which doesn't allow indexing with lists; see
                    # https://github.com/google/jax/issues/4564. The dtype needs to be
                    # explicitly set to an int type, otherwise tl.tensor does the
                    # conversion to floating type which causes issues with the pytorch
                    # backend.
                    rescaling /= tl.sqrt(
                        sampling_probs[n][tl.tensor(samples_unq[n], dtype=tl.int64)]
                    )
                    # print(sampling_probs[n].shape)

            # Sample core tensors
            print(idx_ordering[dim])
            print(samples_unq[3])
            sampled_cores = [tt_decomp[i][:, samples_unq[i], :] for i in idx_ordering[dim]]

            # Construct sketched design matrix
            sampled_subchain_tensor = sampled_cores[0]
            for i in range(1, len(sampled_cores)):
                sampled_subchain_tensor = tl.tenalg.tensordot(
                    sampled_subchain_tensor,
                    sampled_cores[i],
                    modes=(2, 0),
                    batched_modes=(1, 1),
                )
            sampled_design_mat = matricize(sampled_subchain_tensor, [1], [2, 0])

            sampled_design_mat = tl.einsum("i,ij->ij", rescaling, sampled_design_mat)


            # Construct sampled right-hand side
            sampled_tensor_unf = tensor[samples_unq]
            print(sampled_tensor_unf.shape)
            print(rescaling.shape)
            if dim == 0:
                sampled_tensor_unf = tl.transpose(sampled_tensor_unf)
            sampled_tensor_unf = tl.einsum("i,ij->ij", rescaling, sampled_tensor_unf)
            print(sampled_tensor_unf.shape)

            # Solve sampled least squares problem directly
            sol = tl.lstsq(sampled_design_mat, sampled_tensor_unf)[0]

            # Update core
            tt_decomp[dim] = tl.transpose(
                tl.reshape(sol, (rank[dim], rank[dim + 1], shape[dim])),
                [0, 2, 1],
            )

            # Compute sampling distribution for updated core
            sampling_probs[dim] = leverage_score_dist(tl.transpose(sol))
            size = (n_samples[dim])
    return tt_decomp

if __name__ == '__main__':
    rank = 3
    nsweeps = 1
    # X, ground_truth = generate_tt_full_tensors(true_rank, dims, std_noise)
    # s = tns_tt_test(X.data, rank, J, nsweeps, init_method="gaussian", tt_decomp=None)
    cores = [np.random.rand(1, 32, 5), np.random.rand(5, 32, 5), np.random.rand(5, 3, 5), np.random.rand(5, 7200, 1)]
    tensor = tt_to_tensor(cores)
    r = 10
    J = 3
    s = tensor_train_als_sampled(tensor, rank, J, nsweeps, init_method="gaussian", tt_decomp=None)