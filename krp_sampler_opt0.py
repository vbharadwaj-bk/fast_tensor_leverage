import numpy as np
import matplotlib.pyplot as plt

# This is a slow prototype intended to demonstrate correctness.
# It is intended to align as closely as possible to the pseudocode
# in the paper

def divide_and_roundup(n, m):
    return (n + m - 1) // m

def log2_round_down(m):
    assert(m > 0)
    log2_res = 0
    lowest_power_2 = 1

    while lowest_power_2 * 2 < m:
        log2_res += 1
        lowest_power_2 *= 2

    return log2_res, lowest_power_2 

class PartitionTree:
    def __init__(self, n, F):
        '''
        Full, complete binary tree represented through an array of length 2 * (# leaves) - 1.
        Each node is indexed by an integer in [0, nodecount). The root node is 0. 
        '''
        self.n = n
        self.F = F
        self.leaf_count = divide_and_roundup(n, F) 
        self.node_count = 2 * self.leaf_count - 1

        # Lowest level of a complete binary tree that is completely filled 
        self.lfill_level, self.lfill_count = log2_round_down(self.leaf_count)

        # All nodes up to and including the lowest filled layer 
        self.nodes_up_to_lfill = self.lfill_count * 2 - 1
        self.nodes_before_lfill = self.lfill_count - 1 
        self.nodes_at_partial_level_div2 = (self.node_count - self.nodes_up_to_lfill) // 2
        self.complete_level_offset = self.nodes_before_lfill - self.nodes_at_partial_level_div2

    def L(self, v):
        return 2 * v + 1

    def R(self, v):
        return 2 * v + 2

    def is_leaf(self, v):
        return self.L(v) >= self.node_count

    def get_leaf_index(self, v):
        '''
        Gets the index of a leaf in [0, leaf_count). Each leaf is responsible
        for the interval [F * leaf_index, min(F * (leaf_index + 1), node_count))
        '''
        if v >= self.nodes_up_to_lfill:
            return v - self.nodes_up_to_lfill
        else:
            return v - self.complete_level_offset 

    def get_leaf_range(self, v):
        leaf_index = self.get_leaf_index(v)
        start_idx = leaf_index * self.F
        end_idx = min((leaf_index + 1) * self.F, self.n)
        return (start_idx, end_idx)

    def test_node_ranges(self):
        for i in range(self.node_count):
            if self.is_leaf(i):
                print(f"{i} {self.get_leaf_range(i)}")

    def PTSample(self, m, q):
        c = 0
        mc = m(c)
        while not self.is_leaf(c):  
            ml = m(self.L(c))
            Rc = np.random.binomial(1, ml / mc)
            if Rc == 1:
                c = self.L(c)
                mc = ml
            else:
                c = self.R(c)
                mc -= ml

        start, end = self.get_leaf_range(c)
        qprobs = q(c)
        qprobs /= np.sum(qprobs)  # Could also divide by mc 
        Rc = np.random.multinomial(1, qprobs)
        return start + np.nonzero(Rc==1)[0][0]

    def test_on_explicit_pmf(self, masses, sample_count):
        '''
        Test the partition tree sampling on a provided explicit PMF.
        Computes m(v) ahead of time on all nodes, then draws the 
        specified number of samples.
        '''
        m_vals = np.zeros(self.node_count)
        for i in reversed(range(self.node_count)):
            if self.is_leaf(i):
                start, end = self.get_leaf_range(i)
                m_vals[i] = np.sum(masses[start:end])
            else:
                m_vals[i] = m_vals[self.L(i)] + m_vals[self.R(i)]

        m = lambda c : m_vals[c]
        q = lambda c : masses[self.get_leaf_range(c)[0] : self.get_leaf_range(c)[1]].copy()

        result = np.zeros(self.n, dtype=np.int32)
        for i in range(sample_count):
            sample = self.PTSample(m, q)
            result[sample] += 1

        return result / sample_count

def test_tree(tree, sample_count):
    '''
    Test the partition tree with several distributions
    '''
    def run_pmf_test(pmf):
        tree_samples = tree.test_on_explicit_pmf(pmf, sample_count) 
        pmf_normalized = pmf / np.sum(pmf)
        numpy_samples = np.random.multinomial(sample_count, pmf_normalized) / sample_count
        return pmf_normalized, tree_samples, numpy_samples 

    uniform = np.ones(tree.n)
    exponential_decay = np.ones(tree.n)
    for i in range(1, tree.n):
        exponential_decay[i] = exponential_decay[i-1] / 2
    
    return [run_pmf_test(uniform), run_pmf_test(exponential_decay)]

if __name__=='__main__':
    tree = PartitionTree(30, 1)
    test_tree(tree)
