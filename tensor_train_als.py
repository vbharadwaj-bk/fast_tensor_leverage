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
        #gram = lhs.T @ lhs
        #diffnorm = la.norm(gram - np.eye(lhs.shape[1]))
        #print(f"Norm of Diff. Between Gram Matrix and Identity: {diffnorm}")

        lstsq_res, _, _, _ = la.lstsq(lhs, rhs, rcond=None)

        (left_rank, modesize, right_rank) = self.cores[idx].shape

        self.cores[idx] = np.swapaxes(lstsq_res.reshape((left_rank, right_rank, modesize)), 1, 2)

    def compute_residual(self, ground_truth):
        return la.norm(self.materialize() - ground_truth)

    def orthogonalize_core(self, idx, sweeping_right, absorb_r_into_adjacent=True):
        if (idx == 0 and not sweeping_right) or (idx == self.dim-1 and sweeping_right):
            assert(False)

        (left_rank, modesize, right_rank) = self.cores[idx].shape

        labels = self.get_mode_labels()
        if sweeping_right:
            core_reshaped = self.cores[idx].reshape((left_rank * modesize, right_rank))
            q, r = la.qr(core_reshaped)
            self.cores[idx] = q.reshape((left_rank, modesize, right_rank))

            if absorb_r_into_adjacent:
                adj_core_labels = labels[idx + 1]
                adj_core_labels[1] = prepend_dash(adj_core_labels[1]) 
                adj_core_labels[2] = prepend_dash(adj_core_labels[2])
                r_labels = ['-n', adj_core_labels[0]]
                out_order= ['-n', adj_core_labels[1], adj_core_labels[2]]
                res = ncon([r, self.cores[idx + 1]], [r_labels, adj_core_labels], out_order=out_order)
                self.cores[idx + 1] = res
        else:
            core_reshaped = self.cores[idx].reshape((left_rank, modesize * right_rank))
            q, r = la.qr(core_reshaped.T)
            self.cores[idx] = q.T.reshape((left_rank, modesize, right_rank))

            if absorb_r_into_adjacent:
                adj_core_labels = labels[idx - 1]
                adj_core_labels[0] = prepend_dash(adj_core_labels[0]) 
                adj_core_labels[1] = prepend_dash(adj_core_labels[1])
                r_labels = [adj_core_labels[2], '-n']
                out_order= [adj_core_labels[0], adj_core_labels[1], '-n']
                res = ncon([self.cores[idx - 1], r.T], [adj_core_labels, r_labels], out_order=out_order)
                self.cores[idx - 1] = res

    def als_optimize(self, ground_truth, num_iters):
        print(f"Residual before initial orthog. sweep to right: {self.compute_residual(ground_truth)}")
        for idx in reversed(range(1, self.dim)): # Initial sweep to right to orthogonalize cores 
            self.orthogonalize_core(idx, sweeping_right=False) 

        # Here, we will check a certain fact about the tensor-train decomposition
        core = self.cores[1]
        sum_of_grams = np.zeros((core.shape[0], core.shape[0]))
        for i in range(core.shape[1]):
            sum_of_grams += core[:, i, :] @ core[:, i, :].T

        print(sum_of_grams)
        exit(1)

        print(f"Residual after initial orthog. sweep to right: {self.compute_residual(ground_truth)}")

        for iter in range(num_iters):
            for idx in range(self.dim - 1): # Sweep to the right
                self.optimize_core(ground_truth, idx, sweeping_right=True)
                self.orthogonalize_core(idx, sweeping_right=True, absorb_r_into_adjacent=False)             

            for idx in reversed(range(1, self.dim)): # Sweep to the left
                absorb = idx == 1
                self.optimize_core(ground_truth, idx, sweeping_right=False)
                self.orthogonalize_core(idx, sweeping_right=False, absorb_r_into_adjacent=absorb)

            print(f"Iteration {iter} residual magnitude: {self.compute_residual(ground_truth)}")

if __name__=='__main__':
    ground_truth = TensorTrain([20, 21, 22, 23], [5, 6, 7])
    tt_to_optimize = TensorTrain([20, 21, 22, 23], [5, 6, 7])

    tt_to_optimize.als_optimize(ground_truth.materialize(), num_iters=3)
