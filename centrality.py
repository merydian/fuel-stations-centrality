import igraph as ig
# import centrality_core
import numpy as np
import logging
import heapq

logger = logging.getLogger(__name__)

import networkx as nx
import numpy as np
from numba import jit, prange, njit

def nodes_highest_avg_knn_distance_ig(graph: ig.Graph, knn: int, n: int, node_subset=None):
    """
    Returns `n` nodes with the highest average distance to their `knn` nearest neighbors
    in an igraph graph, optionally restricted to a subset of nodes.

    Parameters:
        graph (igraph.Graph): The input igraph graph.
        knn (int): Number of nearest neighbors to consider.
        n (int): Number of nodes to return.
        node_subset (list, optional): Node IDs to restrict calculations. Defaults to all nodes.

    Returns:
        list: Node IDs with the highest average distance to their knn neighbors in node_subset.
    """
    if node_subset is None:
        node_subset = list(range(graph.vcount()))

    results = []
    for node in node_subset:
        # distances from this node to all others
        lengths = graph.shortest_paths(node, node_subset, weights="weight")[0]

        # filter out self (0 distance) and unreachable (inf)
        distances = [d for i, d in zip(node_subset, lengths) if i != node and d != float("inf")]

        if len(distances) < knn:
            continue  # skip nodes that don't have enough reachable neighbors

        nearest = heapq.nsmallest(knn, distances)
        avg_distance = sum(nearest) / knn
        results.append((avg_distance, node))

    # pick top-n by largest avg_distance
    results.sort(reverse=True)
    return [node for _, node in results[:n]]

def graph_straightness(g: ig.Graph, weight: str = None):
    """
    Compute global straightness centrality for a graph.
    Uses Numba JIT compilation for significant performance improvements.

    Parameters
    ----------
    g : ig.Graph
        Road network graph (undirected or directed). Each node must
        have attributes "x" and "y".
    weight : str or None
        Edge attribute to use as distance (default = None for unweighted).

    Returns
    -------
    float
        Global straightness centrality (graph-level detour index).
    """
    n = g.vcount()

    if n == 0:
        return 0.0

    m = g.ecount()
    logger.info(f"Computing global graph straightness for {n} nodes and {m} edges using 'Numba JIT'")

    # Extract node coordinates as numpy arrays for Numba optimization
    coords_x = np.array([v["x"] for v in g.vs], dtype=np.float64)
    coords_y = np.array([v["y"] for v in g.vs], dtype=np.float64)

    # All-pairs shortest paths as numpy array
    logger.debug("Computing shortest paths...")
    if weight and weight in g.es.attributes():
        shortest_paths = np.array(g.distances(weights=weight), dtype=np.float64)
    else:
        shortest_paths = np.array(g.distances(), dtype=np.float64)

    logger.debug("Computing global straightness...")

    return _compute_graph_straightness_core(
        coords_x, coords_y, shortest_paths, n
    )

@njit(parallel=True, fastmath=True)
def _compute_graph_straightness_core(coords_x, coords_y, shortest_paths, n):
    num = 0.0
    den = 0

    for i in prange(n):  # parallel loop
        xi, yi = coords_x[i], coords_y[i]
        for j in range(n):
            if i == j:
                continue

            d_g = shortest_paths[i, j]
            if d_g == np.inf:
                continue

            xj, yj = coords_x[j], coords_y[j]
            d_e = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

            if d_e > 0:
                num += d_e / d_g
                den += 1

    return num / den if den > 0 else 0.0
