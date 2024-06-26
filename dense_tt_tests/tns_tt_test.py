from draw_samples_test import *
from synthetic_data import *


def tns_tt_test(tensor, rank, n_samples, nsweeps, init_method=None, tt_decomp=None):
    shape = tensor.shape
    ranks = validate_tt_rank(shape, rank=rank)
    N = len(shape)
    rng = np.random.default_rng()
    if isinstance(n_samples, int):
        n_samples = [n_samples] * N

    if init_method == "gaussian" and tt_decomp is None:
        tt_decomp = [rng.normal(size=(ranks[i], shape[i], ranks[i + 1])) for i in range(N)]

        for i in range(N):
            tt_decomp[i] /= la.norm(tt_decomp[i])

    elif init_method is None and tt_decomp is not None:
        for i in range(N):
            tt_decomp[i] /= la.norm(tt_decomp[i])

    # permute_indices = [
    #     [n for n in range(dim + 1, N)] + [n for n in range(dim)]
    #     for dim in range(N)
    # ]

    permute_indices = []
    for m in range(N):
        permute_indices.append([n for n in range(m + 1, N)] + [n for n in range(m)])

    for iter in range(nsweeps):
        print(f"{iter} is running ... ")
        for n in range(N):
            sampling_probs = draw_samples_test(tt_decomp, n_samples[n], n)
            samples = []

            for j in range(N):
                k = j
                if j >= n:
                    k -= 1
                    # print("k is",k)
                if j != n:
                    samples.append(rng.choice(range(shape[j]), size=(n_samples[n]), p=tl.to_numpy(sampling_probs[k])))
            # print("Done!")

            samples_unq, samples_cnt = np.unique(samples, axis=1, return_counts=True)
            samples_unq = samples_unq.tolist()
            samples_unq.insert(n, slice(None, None, None))
            samples_unq = tuple(samples_unq)
            samples_cnt = tl.tensor(samples_cnt, **tl.context(tensor))

            # Compute row rescaling factors (see discussion in Sec 4.1 in paper by
            # Larsen & Kolda (2022), DOI: 10.1137/21M1441754)
            rescaling = tl.sqrt(samples_cnt / n_samples[n])

            sampled_cores = [tt_decomp[i][:, samples_unq[i], :] for i in permute_indices[n]]

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

            sampled_tensor_unf = tensor[samples_unq]

            if n == 0:
                sampled_tensor_unf = tl.transpose(sampled_tensor_unf)
            sampled_tensor_unf = tl.einsum("i,ij->ij", rescaling, sampled_tensor_unf)

            # Solve sampled least squares problem directly
            sol = tl.lstsq(sampled_design_mat, sampled_tensor_unf)[0]

            # Update core
            tt_decomp[n] = tl.transpose(
                tl.reshape(sol, (ranks[n], ranks[n + 1], shape[n])),
                [0, 2, 1],
            )

    return tt_decomp




if __name__ == '__main__':
    # N = 4
    # dims = [5] * N
    # true_rank = 3
    # rank = 3
    # std_noise = 1e-6
    # J = 3
    nsweeps = 1
    # X, ground_truth = generate_tt_full_tensors(true_rank, dims, std_noise)
    # s = tns_tt_test(X.data, rank, J, nsweeps, init_method="gaussian", tt_decomp=None)
    cores = [np.random.rand(1, 32, 5), np.random.rand(5, 18, 5), np.random.rand(5, 3, 5), np.random.rand(5, 7200, 1)]
    tensor = tt_to_tensor(cores)
    # print(tensor.shape)
    r = 10
    J = 10
    s = tns_tt_test(tensor, r, J, nsweeps, init_method="gaussian", tt_decomp=None)
