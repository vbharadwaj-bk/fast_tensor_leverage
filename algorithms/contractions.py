import tensornetwork as tn
import numpy as np

def vec(X):
    prod_shape = np.prod(X.shape)
    return X.reshape(prod_shape)

def contract_nodes(nodes_in, 
                   contractor, 
                   output_order=None, 
                   out_row_modes=None, 
                   out_col_modes=None,
                   vectorize=False,
                   debug=False
                   ):

    matricize = out_row_modes is not None
    assert(not (vectorize and matricize))

    if matricize:
        output_order = out_row_modes + out_col_modes
    elif output_order is None:
        output_order = []

    nodes = tn.replicate_nodes(nodes_in)
    edges = {} # Dictionary of axis name to nodes sharing the edge

    for node in nodes:
        for name in node.axis_names:
            if name not in edges:
                edges[name] = [] 

            edges[name].append(node)

    if debug: 
        for name in edges:
            print(f'{name}, {len(edges[name])}')

    dangling_edges = {}
    for name in edges:
        if len(edges[name]) >= 3:
            node_names = ', '.join([node.name for node in edges[name]])
            raise Exception(f"Error, nodes {node_names} all share edge {name}")
        elif(len(edges[name]) == 1):
            dangling_edges[name] = edges[name][0].get_edge(name)
        else:
            tn.connect(edges[name][0][name], edges[name][1][name])

    for name in dangling_edges:
        if name not in output_order:
            exception_text = f"Error, list of dangling edges is {list(dangling_edges.keys())}. Edge {name} is dangling and not in output order.\n"
            exception_text += f"Output order: {output_order}"
            raise Exception(exception_text)

    output_edges = []
    for name in output_order:
        if name not in dangling_edges:
            exception_text = f"Error, edge {name} is not dangling.\n"

            if len(edges[name]) == 2:
                exception_text += f"Edge {name} spans nodes {edges[name][0]} and {edges[name][1]}."
            else:
                exception_text += f"Edge {name} not found."

            raise Exception(exception_text)

        else:
            output_edges.append(dangling_edges[name])

    if len(output_edges) == 0:
        output_edges = None
    result = contractor(nodes, output_edge_order=output_edges)

    if vectorize:
        return vec(result.tensor)        
    elif matricize:
        shape = result.shape
        row_count = np.prod(shape[:len(out_row_modes)])
        col_count = np.prod(shape[len(out_row_modes):])

        return result.tensor.reshape(row_count, col_count)
    else:
        if len(output_order) > 0:
            result.add_axis_names(output_order)
        return result
 