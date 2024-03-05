import time

import networkx as nx
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import yen


def _refine_negative_cycles(matrix, directed=True, unweighted=False):
    EPS = 1e-6
    if directed:
        nx_graph_type = nx.DiGraph
    else:
        nx_graph_type = nx.Graph
    if unweighted:
        nx_wieghts = None
    else:
        nx_wieghts = "weight"
    graph = csr_matrix(matrix)
    nx_graph = nx.from_scipy_sparse_array(graph, create_using=nx_graph_type)
    while nx.negative_edge_cycle(nx_graph, weight=nx_wieghts):
        negative_cycle = nx.find_negative_cycle(nx_graph, 0, weight=nx_wieghts)
        path_weight = nx.path_weight(nx_graph, negative_cycle, weight=nx_wieghts)
        nx_graph[negative_cycle[0]][negative_cycle[1]]["weight"] += (
            abs(path_weight) + EPS
        )
    return nx.adjacency_matrix(nx_graph, weight=nx_wieghts).todense()


def _create_random_adjacency_matrix(
    n, min_weight=0, max_weight=1, symmetric=False, edge_prob: float = 0.5
):
    # Create a random adjacency matrix
    if symmetric:
        edge_prob = edge_prob / 2
    bool_matrix = np.random.choice([0, 1], p=[1 - edge_prob, edge_prob], size=(n, n))
    matrix = np.zeros((n, n))
    matrix[bool_matrix == 1] = (
        np.random.random(size=np.count_nonzero(bool_matrix)) * (max_weight - min_weight)
        + min_weight
    )
    if min_weight < 0:
        first_row = 0
        print(
            f"matrix before: \n{matrix}. \nShifting all rows but {first_row} by {abs(min_weight)}\n"
        )
        matrix[[i for i in range(n) if i != first_row], :] += abs(min_weight)
        matrix = matrix * bool_matrix
        # print(f"matrix after: {matrix}")
    if symmetric:
        matrix = (matrix + matrix.T) / 2

    # Set the diagonal to zero (no self-loops)
    np.fill_diagonal(matrix, 0)

    return matrix


def _compare_methods(graph, source, sink, K, symmetric=False, unweighted=False):
    directed = not symmetric

    # Run your yen function and time it
    print("running SCIPY yen")
    start = time.time()
    dist_array, predecessors = yen(
        graph,
        source,
        sink,
        K,
        return_predecessors=True,
        directed=directed,
        unweighted=unweighted,
    )
    end = time.time()
    yen_time = end - start
    num_paths_found = dist_array.size
    if num_paths_found == 0:
        print("SCIPY found no paths")
    else:
        print(f"SCIPY found {num_paths_found} paths")

    if directed:
        nx_graph_type = nx.DiGraph
    else:
        nx_graph_type = nx.Graph

    # Convert the graph to NetworkX format
    is_negative = False
    min_value = 0
    if any(graph.data) < 0:
        # nx.shortest_simple_paths does not support negative weights
        # So we shift the weights to be non-negative, and then shift back to compare
        is_negative = True
        min_value = min(graph.data)
        graph.data += abs(min_value)
    G = nx.from_scipy_sparse_array(graph, create_using=nx_graph_type)
    print(f"yen took {yen_time} seconds")

    print("running NETWORKX shortest_simple_path")
    if unweighted:
        nx_wieghts = None
    else:
        nx_wieghts = "weight"
    start = time.time()
    try:
        nx_paths = list(nx.shortest_simple_paths(G, source, sink, nx_wieghts))[:K]
    except nx.NetworkXNoPath:
        assert (
            num_paths_found == 0
        ), f"NETWORKX found no path, but SCIPY {num_paths_found} paths"
        return
    end = time.time()
    nx_time = end - start

    # Print the results
    print(f"shortest_simple_paths took {nx_time} seconds")

    # Compare the results
    yen_paths = [
        get_scipy_path(predecessors[k], source, sink) for k in range(num_paths_found)
    ]
    for i, (yen_path, nx_path) in enumerate(zip(yen_paths, nx_paths)):
        print(f"Path {i + 1} distance: {dist_array[i]}")
        print(f"yen: {yen_path}")
        print(f"shortest_simple_paths: {nx_path}")
        print()
        if nx_wieghts:
            nx_distance = nx.path_weight(G, nx_path, nx_wieghts)
            if is_negative:
                nx_distance -= abs(min_value) * (len(nx_path) - 1)
        else:
            # unweighted graph
            nx_distance = len(nx_path) - 1
        assert (
            abs(dist_array[i] - nx_distance) < 1e-6
        ), f"Path {i+1} has different weights: SCIPY: {dist_array[i]} vs NETWORKX: {nx_distance}"
        # if yen_path != nx_path:

    assert len(yen_paths) == len(
        nx_paths
    ), f"Paths are not the same length: SCIPY: {len(yen_paths)} vs NETWORKX: {len(nx_paths)}"


def get_scipy_path(predecessors, source, target):
    path = []
    i = target
    while i != source and i >= 0:
        path.append(i)
        i = predecessors[i]
    if i < 0:
        return []
    path.append(source)
    return path[::-1]


if __name__ == "__main__":
    """
    Compare scipy.sparse.csgraph.yen with networkx.shortest_simple_paths
    Choose number of nodes in the graph, the sparsity of the graph,
    the minimal and maximal weight of the edges.
    The adjacency matrix of the graph is chosen randomly.
    To avoid circular paths, Only the first row is allowed to have negative weights.
    Even then, some refinement is needed to avoid negative cycles.
    Notes:
        Must use networkx version ~=3.2.1
    """

    num_nodes = 12
    sparsity = 0.3
    min_weight = -100
    max_weight = 100
    num_paths_to_search = 15
    print("Checking graph with", num_nodes, "nodes")
    # for symmetric in (False,):
    for symmetric in (True, False):
        for unweighted in (True, False):
            for i in range(50):
                matrix = _create_random_adjacency_matrix(
                    num_nodes,
                    min_weight,
                    max_weight,
                    symmetric=symmetric,
                    edge_prob=sparsity,
                )
                matrix = _refine_negative_cycles(
                    matrix, directed=not symmetric, unweighted=unweighted
                )
                print(matrix)
                # Compare the methods
                graph = csr_matrix(matrix)
                _compare_methods(
                    graph,
                    0,
                    num_nodes - 1,
                    K=num_paths_to_search,
                    symmetric=symmetric,
                    unweighted=unweighted,
                )
