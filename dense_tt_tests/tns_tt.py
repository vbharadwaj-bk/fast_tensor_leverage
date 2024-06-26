from draw_samples_tns_tt import *


def tns_tt(tensor, ranks, n_samples, nsweeps, init_method=None, tt_decomp=None):
    shape = tensor.shape
    rank = validate_tt_rank(shape, rank=ranks)
    n_dim = len(shape)
    rng = np.random.default_rng()
    if isinstance(n_samples, int):
        n_samples = [n_samples] * n_dim

    # core_samples = [None] * n_dim
    # idx_ordering = []
    # for m in range(n_dim):
    #     idx_ordering.append(list(range(m + 1, n_dim)) + list(range(m)))
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

    for iter in range(nsweeps):
        print(f"{iter} is running ... ")
        for n in range(n_dim):
            J2 = n_samples[n]
            samples = draw_samples_tns_tt(tt_decomp, n, J2)[0].T
            print(samples[0].shape)
            sqrt_p = draw_samples_tns_tt(tt_decomp, n, J2)[1]

            samples_unq = np.unique(samples, axis=1)
            samples_unq = samples_unq.tolist()
            samples_unq.insert(n, slice(None, None, None))
            samples_unq = tuple(samples_unq)
            print(samples_unq)

            idx = idx_ordering[n]
            print(idx)

            core_samples = [tt_decomp[i][:, samples_unq[i], :] for i in idx]
            #
            # for m in idx_ordering[n]:
            #     if m != n:
            #         core_samples[m] = tt_decomp[m][:, samples_unq[m],:]

            rescaling = np.ones(J2) / sqrt_p * np.sqrt(J2)
            sampled_subchain_tensor = core_samples[0]
            for i in range(1, len(core_samples)):
                sampled_subchain_tensor = tl.tenalg.tensordot(
                    sampled_subchain_tensor,
                    core_samples[i],
                    modes=(2, 0),
                    batched_modes=(1, 1),
                )
            sampled_design_mat = matricize(sampled_subchain_tensor, [1], [2, 0])
            sampled_design_mat = tl.einsum("i,ij->ij", rescaling, sampled_design_mat)

            # samples_unq, samples_cnt = np.unique(samples[:], return_counts=True)
            # print(samples_unq)
            # samples_unq = samples_unq.tolist()

            # Construct sampled right-hand side
            print(samples_unq)
            sampled_tensor_unf = tensor[samples_unq]
            print(sampled_tensor_unf.shape)
            if n == 0:
                sampled_tensor_unf = tl.transpose(sampled_tensor_unf)
            sampled_tensor_unf = tl.einsum("i,ij->ij", rescaling, sampled_tensor_unf)

            # Solve sampled least squares problem directly
            sol = tl.lstsq(sampled_design_mat, sampled_tensor_unf)[0]

            # Update core
            tt_decomp[dim] = tl.transpose(
                tl.reshape(sol, (rank[n], rank[n + 1], shape[n])),
                [0, 2, 1],
            )

        return tt_decomp
#
if __name__ == '__main__':
    n_trials = 1
    N = 4
    dims = [5] * N
    true_rank = 3
    rank = 3
    std_noise = 1e-6
    J = [2,2,2,2]
    nsweeps = 1
    X, ground_truth = generate_tt_full_tensors(true_rank, dims, std_noise)
    cores = tns_tt(X.data, rank, J, nsweeps, init_method='gaussian', tt_decomp=None)
    approx = tt_to_tensor(cores)
    fitness = fit(ground_truth.data, approx)
    print(fitness)



