import numpy as np

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
        Each node is indexed by an integer in [0, nodecount). The root node is 0: 
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
        Gets the index of a leaf in the range [0, leaf_count) 
        '''
        if v >= self.nodes_up_to_lfill:
            return v - self.nodes_up_to_lfill
        else:
            return v - self.complete_level_offset 

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

    def print_leaf_indices(self):
        for i in range(self.node_count):
            if self.is_leaf(i):
                print(f"Leaf {i} {self.get_leaf_index(i)}")

if __name__=='__main__':
    tree = PartitionTree(29, 1)
    tree.print_leaf_indices()
