import igraph as ig
import osmnx as ox
import numpy as np
import logging
from numba import jit, prange, njit

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
    print("knn-dists", avg_distances)
    top_nodes = [node for node, _ in avg_distances[:n]]
    return top_nodes



def download_graph(place):
    logger.info(f"Downloading street network for: {place}")
    try:
        # Download the street network for the given place and convert to igraph
        G_nx = ox.graph_from_place(place, network_type="drive")
        logger.info(
            f"Downloaded NetworkX graph with {len(G_nx.nodes)} nodes and {len(G_nx.edges)} edges"
        )

        # Convert NetworkX to igraph
        logger.info("Converting NetworkX graph to igraph...")
        G = ig.Graph.from_networkx(G_nx)
        logger.info(
            f"Conversion completed. igraph has {G.vcount()} vertices and {G.ecount()} edges"
        )

        return G
    except Exception as e:
        logger.error(f"Failed to download graph for {place}: {e}")
        raise


@jit(nopython=True, cache=True)
def _compute_straightness_core(coords_x, coords_y, shortest_paths, n):
    """
    Numba-optimized core computation for straightness centrality.

    Parameters
    ----------
    coords_x : numpy.ndarray
        X coordinates of all nodes
    coords_y : numpy.ndarray
        Y coordinates of all nodes
    shortest_paths : numpy.ndarray
        2D array of shortest path distances between all node pairs
    n : int
        Number of nodes

    Returns
    -------
    numpy.ndarray
        Straightness centrality values for all nodes
    """
    straightness = np.zeros(n, dtype=np.float64)

    for i in range(n):
        si = 0.0
        valid = 0
        xi, yi = coords_x[i], coords_y[i]

        for j in range(n):
            if i == j:
                continue

            d_g = shortest_paths[i, j]
            if d_g == np.inf:
                continue  # disconnected

            xj, yj = coords_x[j], coords_y[j]
            d_e = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

            if d_e > 0:
                si += d_e / d_g
                valid += 1

        straightness[i] = si / valid if valid > 0 else 0.0

    return straightness


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


def straightness_centrality(g: ig.Graph, weight: str = None):
    """
    Compute straightness centrality for all nodes in a graph.
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
    straightness : list of floats
        Straightness centrality for each node.
    """
    n = g.vcount()

    if n == 0:
        return []

    logger.info(
        f"Computing straightness centrality for {g.vcount():,} nodes and {g.ecount():,} edges using Numba JIT"
    )

    # Extract node coordinates as numpy arrays for Numba optimization
    coords_x = np.array([v["x"] for v in g.vs], dtype=np.float64)
    coords_y = np.array([v["y"] for v in g.vs], dtype=np.float64)

    # Precompute all shortest path lengths as numpy array
    logger.debug("Computing shortest paths...")
    if weight and weight in g.es.attributes():
        shortest_paths = np.array(g.distances(weights=weight), dtype=np.float64)
    else:
        shortest_paths = np.array(g.distances(), dtype=np.float64)

    logger.debug("Computing straightness centrality...")

    # Use optimized Numba implementation
    straightness = _compute_straightness_core(coords_x, coords_y, shortest_paths, n)
    return straightness.tolist()


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

    return _compute_graph_straightness_core(coords_x, coords_y, shortest_paths, n)
