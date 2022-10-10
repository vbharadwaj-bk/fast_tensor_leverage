from re import I
import numpy as np
import numpy.linalg as la
from tensornetwork import ncon

def prepend_dash(str_el):
    return f'-{str_el}'

def matricize_isolate_last_mode(arr):
    shape = arr.shape
    return arr.reshape((np.prod(shape[:-1]), shape[-1])) 

class TensorTrain:
    def __init__(self, mode_sizes, ranks):
        self.dim = len(mode_sizes)
        self.mode_sizes = mode_sizes 
        self.ranks = ranks
        assert (len(self.ranks) + 1 == self.dim) 
        self.padded_ranks = [1] + self.ranks + [1]

        self.cores = [2 * (np.random.rand(self.padded_ranks[i], self.mode_sizes[i], self.padded_ranks[i + 1]) - 0.5)
                      for i in range(len(self.padded_ranks) - 1)]

    def get_mode_labels(self):
        return [[f'h{i}', f'g{i}', f'h{i+1}'] for i in range(self.dim)] 

    def unfold_core_left(self, i):
        return self.cores[i].view(self.padded_ranks[i], self.mode_sizes[i] * self.padded_ranks[i + 1])

    def unfold_core_right(self, i):
        return self.cores[i].view(self.padded_ranks[i] * self.mode_sizes[i], self.padded_ranks[i + 1])

    def refold_core(self, i, tensor):
        return tensor.view(self.padded_ranks[i], self.mode_sizes[i], self.padded_ranks[i + 1])

    def evaluate(self, indices):
        slices = [self.cores[i][:, indices[i], :] for i in range(len(indices))]
        return np.chain_matmul(slices)

    def materialize(self):
        labels = self.get_mode_labels()
        for lst in labels:
            lst[1] = prepend_dash(lst[1]) 

        labels[0][0] = prepend_dash(labels[0][0]) 
        labels[-1][-1] = prepend_dash(labels[-1][-1]) 

        return ncon(self.cores, labels).squeeze()

    def materialize_left_contraction(self, i):
        if i == 0:
            return None

        left_cores = self.cores[:i]
        labels = self.get_mode_labels()[:i]

        labels[0][0] = prepend_dash(labels[0][0]) 
        labels[-1][-1] = prepend_dash(labels[-1][-1])
        output_label_list = [labels[0][0]]
        for lst in labels:
            lst[1] = prepend_dash(lst[1])
            output_label_list.append(lst[1])
        output_label_list.append(labels[-1][-1])

        res = ncon(left_cores, labels, out_order=output_label_list).squeeze() 
        return matricize_isolate_last_mode(res)

    def materialize_right_contraction(self, i):
        if i == self.dim - 1:
            return None

        right_cores = self.cores[(i+1):]
        labels = self.get_mode_labels()[(i+1):]

        labels[0][0] = prepend_dash(labels[0][0]) 
        labels[-1][-1] = prepend_dash(labels[-1][-1])
        output_label_list = [labels[-1][-1]]
        for lst in labels:
            lst[1] = prepend_dash(lst[1])
            output_label_list.append(lst[1])

        output_label_list.append(labels[0][0])

        res = ncon(right_cores, labels, out_order=output_label_list).squeeze()
        return matricize_isolate_last_mode(res)

    def optimize_core(self, ground_truth, idx, sweeping_right):
        left_contraction = self.materialize_left_contraction(idx)
        right_contraction = self.materialize_right_contraction(idx)

        if left_contraction is None:
            lhs = right_contraction
        elif right_contraction is None:
            lhs = left_contraction
        else:
            lhs = np.kron(left_contraction, right_contraction)
            
        axes = list(range(self.dim))
        axes.remove(idx)
        axes.append(idx)

        rhs = matricize_isolate_last_mode(np.transpose(ground_truth, axes=axes))
        lstsq_res, _, _, _ = la.lstsq(lhs, rhs, rcond=None)

        left_rank = self.cores[idx].shape[0]
        modesize = self.cores[idx].shape[1]
        right_rank = self.cores[idx].shape[2]

        self.cores[idx] = np.swapaxes(lstsq_res.reshape((left_rank, right_rank, modesize)), 1, 2)

    def compute_residual(self, ground_truth):
        return la.norm(self.materialize() - ground_truth)

    def als_optimize(self, ground_truth, num_iters):
        for iter in range(num_iters):
            for idx in range(self.dim): # Sweep to the right
                self.optimize_core(ground_truth, idx, sweeping_right=True)
                print(f"Residual iteration {iter}, right sweep index {idx}, {self.compute_residual(ground_truth)}")

            for idx in reversed(range(self.dim)): # Sweep to the left
                self.optimize_core(ground_truth, idx, sweeping_right=False)
                print(f"Residual iteration {iter}, left sweep index {idx}, {self.compute_residual(ground_truth)}")

if __name__=='__main__':
    ground_truth = TensorTrain([20, 21, 22, 23], [5, 6, 7])
    tt_to_optimize = TensorTrain([20, 21, 22, 23], [5, 6, 7])

    tt_to_optimize.als_optimize(ground_truth.materialize(), num_iters=3)
