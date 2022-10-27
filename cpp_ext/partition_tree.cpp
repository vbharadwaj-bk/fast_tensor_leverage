//cppimport
#include <iostream>
#include "common.h"

using namespace std;

class PartitionTree {
public:
    uint32_t n, F;
    uint32_t leaf_count, node_count;

    uint32_t lfill_level, lfill_count;
    uint32_t nodes_upto_lfill, nodes_before_lfill;
    uint32_t complete_level_offset;
    PartitionTree(uint32_t n, uint32_t F) {
        this->n = n;
        this->F = F;

        leaf_count = divide_and_roundup(n, F);
        node_count = 2 * leaf_count - 1;

        log2_round_down(leaf_count, lfill_level, lfill_count);
        nodes_upto_lfill = lfill_count * 2 - 1;
        nodes_before_lfill = lfill_count - 1;

        uint32_t nodes_at_partial_level_div2 = (node_count - nodes_upto_lfill) / 2;
        complete_level_offset = nodes_before_lfill - nodes_at_partial_level_div2;
    }
};

// cfg['libraries'] = ['cuckoofilter']

PYBIND11_MODULE(partition_tree, m) {
  py::class_<PartitionTree>(m, "PartitionTree")
    .def(py::init<uint32_t, uint32_t>());
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-O3','-march=native']
cfg['extra_link_args'] = []
cfg['dependencies'] = ['common.h'] 
%>
*/
