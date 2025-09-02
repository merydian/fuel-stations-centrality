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
    logger.info(f"Computing k-NN distances for k={knn}, returning top {n} nodes")
    
    if node_subset is None:
        node_subset = list(range(graph.vcount()))
        logger.debug(f"Using all {len(node_subset)} nodes for k-NN calculation")
    else:
        logger.debug(f"Using subset of {len(node_subset)} nodes for k-NN calculation")

    if len(node_subset) <= knn:
        logger.warning(f"Node subset size ({len(node_subset)}) is not larger than k ({knn})")

    # Compute all distances at once (much faster)
    logger.debug("Computing all pairwise distances...")
    all_distances = graph.distances(source=node_subset, target=node_subset, weights="weight")
    logger.debug("Distance computation completed")
    
    results = []
    nodes_processed = 0
    nodes_with_insufficient_neighbors = 0
    
    for i, node in enumerate(node_subset):
        # Get distances from this node to all others
        distances = [all_distances[i][j] for j in range(len(node_subset)) 
                    if i != j and all_distances[i][j] != float("inf")]
        
        if len(distances) < knn:
            nodes_with_insufficient_neighbors += 1
            continue
            
        nearest = heapq.nsmallest(knn, distances)
        avg_distance = sum(nearest) / knn
        results.append((avg_distance, node))
        nodes_processed += 1

    logger.debug(f"Processed {nodes_processed} nodes successfully")
    if nodes_with_insufficient_neighbors > 0:
        logger.debug(f"Skipped {nodes_with_insufficient_neighbors} nodes with insufficient neighbors")

    results.sort(reverse=True)
    final_results = [node for _, node in results[:n]]
    
    logger.info(f"✓ k-NN computation complete. Returning {len(final_results)} nodes with highest average k-NN distances")
    
    return final_results

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
    m = g.ecount()

    logger.info(f"Computing global graph straightness centrality")
    logger.debug(f"Graph properties: {n} nodes, {m} edges")
    logger.debug(f"Using weight attribute: {weight if weight else 'unweighted'}")

    if n == 0:
        logger.warning("Empty graph provided, returning straightness = 0.0")
        return 0.0

    # Extract node coordinates as numpy arrays for Numba optimization
    logger.debug("Extracting node coordinates for Numba optimization")
    coords_x = np.array([v["x"] for v in g.vs], dtype=np.float64)
    coords_y = np.array([v["y"] for v in g.vs], dtype=np.float64)
    logger.debug(f"Extracted coordinates for {len(coords_x)} nodes")

    # All-pairs shortest paths as numpy array
    logger.debug("Computing all-pairs shortest paths...")
    if weight and weight in g.es.attributes():
        logger.debug(f"Using edge weights from attribute: {weight}")
        shortest_paths = np.array(g.distances(weights=weight), dtype=np.float64)
    else:
        logger.debug("Computing unweighted shortest paths")
        shortest_paths = np.array(g.distances(), dtype=np.float64)
    
    logger.debug("Shortest paths computation completed")

    logger.debug("Starting Numba-optimized straightness calculation...")
    straightness = _compute_graph_straightness_core(
        coords_x, coords_y, shortest_paths, n
    )
    
    logger.info(f"✓ Global graph straightness computed: {straightness:.6f}")
    
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
