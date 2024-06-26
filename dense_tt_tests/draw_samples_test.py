import numpy as np
import numpy.linalg as la
import tensorly as tl
from tensorly.base import matricize
from tensorly.tt_tensor import validate_tt_rank, TTTensor


def classical_mode_n_unfolding(tensor, n):
    shape = tensor.shape
    N = len(shape)
    permute_indices = [n] + [i for i in range(N) if i != n]
    tensor_mat = np.transpose(tensor, permute_indices)
    tensor_mat = tensor_mat.reshape(tensor.shape[n], -1)
    return tensor_mat


def mode_n_unfolding(tensor, n):
    shape = tensor.shape
    N = len(shape)
    permute_indices = [n] + [i for i in reversed(range(N)) if i != n]
    tensor_mat = np.transpose(tensor, permute_indices)
    tensor_mat = tensor_mat.reshape(tensor.shape[n], -1)
    return tensor_mat


def softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def draw_samples_test(cores, J, n):
    N = len(cores)

    permute_indices = []
    for m in range(N):
        permute_indices.append([n for n in range(m + 1, N)] + [n for n in range(m)])

    # permute_indices = [
    #     [n for n in range(dim+1, N)] + [n for n in range(dim)]
    #     for dim in range(N)
    # ]
    # print("permute is here",permute_indices)

    first_term = [None] * N
    second_term = [None] * N
    first_terms_unfoldings = [None] * N
    M = None
    # M_2 = np.ones((cores[1].shape[0] ** 2, cores[1].shape[2] ** 2))

    # common_terms = np.zeros((cores[1].shape[1], (cores[1].shape[2]**2) * (cores[1].shape[0]**2)))
    prob_vec = np.zeros((J), dtype=np.float32)
    for j in range(N):
        shape = cores[j].shape
        if j != n:
            G_2 = mode_n_unfolding(cores[j], 1)
            temp = tl.tenalg.khatri_rao([G_2.T, G_2.T])
            temp = temp.reshape(shape[2], shape[0], shape[2], shape[0], shape[1])
            temp = temp.transpose([1, 3, 4, 0, 2])
            first_term[j] = temp.reshape(shape[0] ** 2, shape[1], shape[2] ** 2)
            first_terms_unfoldings[j] = matricize(first_term[j], [1], [0, 2])
            # print("first term is here", first_terms_unfoldings[j].shape)

            temp = G_2.T @ G_2
            temp = temp.reshape([shape[2], shape[0], shape[2], shape[0]])
            temp = temp.transpose([1, 3, 0, 2])
            second_term[j] = temp.reshape([shape[0] ** 2, shape[2] ** 2])
            # C2_vec[j] = second_term[j].flatten('F').T
    # print(len(first_term))

    A_not_mTA_not_m = second_term[permute_indices[n][0]]
    for j in range(1, N-1):
        A_not_mTA_not_m = A_not_mTA_not_m @ second_term[permute_indices[n][j]]
    shape = np.shape(cores[n])
    A_not_mTA_not_m = np.transpose(A_not_mTA_not_m.reshape([shape[2], shape[2], shape[0], shape[0]]), [2, 0, 3, 1])
    A_not_mTA_not_m = A_not_mTA_not_m.reshape([shape[0] * shape[2], shape[0] * shape[2]])
    phi = la.pinv(A_not_mTA_not_m)
    temp = np.reshape(phi, [shape[0], shape[2], shape[0], shape[2]])
    temp = temp.transpose([0, 2, 1, 3])
    second_term[n] = temp.reshape(shape[0] ** 2, shape[2] ** 2)

    samples = np.zeros((J, N), dtype=np.float32)
    sampling_prob = []

    if n == 0:
        first_idx = 1
    else:
        first_idx = 0
    first_idx_flag = True


    for j in range(first_idx, N):
        if j != n:
            if j <= permute_indices[j][0] or permute_indices[j][0] == n:
                # print("indices are here",permute_indices[j][0])
                M = second_term[permute_indices[j][0]]
            else:
                M = first_term[permute_indices[j][0]][:, permute_indices[j][0], :]
            for i in range(1, len(permute_indices[j])):
                if j <= permute_indices[j][i] or permute_indices[j][i] == n:
                    M = M @ second_term[permute_indices[j][i]]
                else:
                    M = M @ first_term[permute_indices[j][i]][:, permute_indices[j][0], :]
            # print(first_terms_unfoldings[j].shape)
            # print(M.shape)
            # print(first_terms_unfoldings[j].shape)
            prob_vec = first_terms_unfoldings[j] @ M.ravel()
            # print("prob_shape is",prob_vec.shape)
            # print(prob_vec.shape)
            prob_vec = softmax(prob_vec)
            sampling_prob.append((prob_vec))

                # print(prob_vec.shape)
                # prob_vec = softmax(prob_vec)

                    # if j <= i:
                    #     M_1[:, permute_indices[i][j], :] = M_1[:, permute_indices[i][j], :] * first_term[j][:, permute_indices[i][j], :]
                    #     print(M_1.shape)
                    # else:
                    #     M_2 = M_2 * second_term[j]
            #     # M_1.append(M_1)
            # if j == n:
            #     s = (1 / cores[1].shape[2]) * s
            # count -= 1
        # prob_vec = classical_mode_n_unfolding(M, 1) @ (M.ravel())

        # prob_vec = classical_mode_n_unfolding(M, 1) @ (s.ravel() * M_2.ravel())
        # print("prob_vec shape is", prob_vec.shape)
        # prob_vec = softmax(prob_vec)


        # sampling_prob.append(prob_vec)

    return sampling_prob



if __name__ == '__main__':
    cores = [np.random.rand(1, 32, 5), np.random.rand(5, 18, 5), np.random.rand(5, 3, 5), np.random.rand(5, 7200, 1)]
    # n = 2
    J = 10
    s = []
    for n in range(len(cores)):
        s = draw_samples_test(cores, J, n)
        print(len(s))
    # print(s[0].shape)







