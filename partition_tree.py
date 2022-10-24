import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

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

def chain_had_prod(matrices):
    res = np.ones(matrices[0].shape)
    for mat in matrices:
        res *= mat
    return res

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

        self.rng = np.random.default_rng()

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

    def S(self, v):
        leaf_index = self.get_leaf_index(v)
        start_idx = leaf_index * self.F
        end_idx = min((leaf_index + 1) * self.F, self.n)
        return (start_idx, end_idx)

    def test_node_ranges(self):
        for i in range(self.node_count):
            if self.is_leaf(i):
                print(f"{i} {self.S(i)}")

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

        start, end = self.S(c)
        qprobs = q(c)
        Rc = np.random.multinomial(1, qprobs / np.sum(qprobs)) # Could also divide by mc 
        return start + np.nonzero(Rc==1)[0][0]

    def PTSampleUpgraded(self, m, q):
        c = 0
        mtotal = m(c)
        low, high = 0.0, 1.0
        draw = self.rng.uniform() 
        while not self.is_leaf(c):  
            ml = m(self.L(c))
            cutoff = low + ml / mtotal
            if draw <= cutoff:
                c = self.L(c)
                high = cutoff
            else:
                c = self.R(c)
                low = cutoff

        assert(low <= draw and draw <= high)

        draw_fraction = min(max((draw - low) / (high - low), 0), 1.0)
        qprobs = q(c)
        normalized = qprobs / np.sum(qprobs)
        prefix_sums = np.cumsum(normalized) 
        Rc = np.searchsorted(prefix_sums, draw_fraction) 
        start, _ = self.S(c)

        #Rc = np.random.multinomial(1, qprobs / np.sum(qprobs)) # Could also divide by mc 
        #return start + np.nonzero(Rc==1)[0][0]
        return start + Rc 


    def test_on_explicit_pmf(self, masses, sample_count):
        '''
        Test the partition tree sampling on a provided explicit PMF.
        Computes m(v) ahead of time on all nodes, then draws the 
        specified number of samples.
        '''
        m_vals = np.zeros(self.node_count)
        for i in reversed(range(self.node_count)):
            if self.is_leaf(i):
                start, end = self.S(i)
                m_vals[i] = np.sum(masses[start:end])
            else:
                m_vals[i] = m_vals[self.L(i)] + m_vals[self.R(i)]

        m = lambda c : m_vals[c]
        q = lambda c : masses[self.S(c)[0] : self.S(c)[1]].copy()

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