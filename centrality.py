import igraph as ig
import centrality_core
import numpy as np
import logging

logger = logging.getLogger(__name__)

import networkx as nx
import numpy as np

def nodes_highest_avg_knn_distance_nx(graph: nx.Graph, knn: int, n: int, node_subset=None):
    """
    Returns `n` nodes with the highest average distance to their `knn` nearest neighbors
    in a NetworkX graph, optionally restricted to a subset of nodes.

    Parameters:
        graph (nx.Graph): The input NetworkX graph.
        knn (int): Number of nearest neighbors to consider.
        n (int): Number of nodes to return.
        node_subset (list, optional): Node IDs to restrict calculations. Defaults to all nodes.

    Returns:
        list: Node IDs with the highest average distance to their knn neighbors.
    """
    if node_subset is None:
        node_subset = graph.nodes()

    avg_distances = []

    for node in node_subset:
        # Compute shortest path lengths from node to all other nodes
        lengths = nx.single_source_dijkstra_path_length(graph, node)
        # Remove self-distance
        lengths.pop(node, None)
        # Take knn smallest distances
        knn_dists = sorted(lengths.values())[:knn]
        avg_distances.append((node, np.mean(knn_dists)))

    # Sort by average distance descending and take top n
    avg_distances.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in avg_distances[:n]]
    return top_nodes

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

    logger.info(f"Computing global graph straightness for {n} nodes using 'Numba JIT'")

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

    n = coords_x.shape[0]
    flat_shortest_paths = shortest_paths.ravel()  # converts (n,n) → (n*n,)

    return centrality_core.graph_centrality(
        coords_x, coords_y, flat_shortest_paths, n
    )
