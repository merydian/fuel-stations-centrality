import igraph as ig
import osmnx as ox
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_knn_distance(G, weight=None, k=3):
    """
    Calculate the average distance to the k nearest neighbor stations for each node.

    Parameters
    ----------
    G : igraph.Graph
        The graph containing stations as nodes
    weight : str, optional
        Edge attribute to use as weight for shortest path calculation
    k : int, optional
        Number of nearest neighbors to consider (default=3)

    Returns
    -------
    dict
        Dictionary mapping node index to average k-NN distance
    """
    logger.info(f"Computing {k}-nearest neighbor distances for {G.vcount()} nodes")

    n = G.vcount()
    knn_distances = {}

    # Get all shortest path distances
    logger.debug("Calculating shortest path distances matrix for k-NN...")
    distances = G.distances()

    for i in range(n):
        # Get distances from node i to all other nodes
        node_distances = []
        for j in range(n):
            if i != j and distances[i][j] != float("inf"):
                node_distances.append(distances[i][j])

        # Sort distances and take k nearest neighbors
        node_distances.sort()
        k_nearest = node_distances[: min(k, len(node_distances))]

        # Calculate average distance to k nearest neighbors
        if k_nearest:
            avg_knn_dist = np.mean(k_nearest)
        else:
            avg_knn_dist = 0.0  # No reachable neighbors

        knn_distances[i] = avg_knn_dist

        if i % max(1, n // 10) == 0:  # Log progress every 10%
            logger.debug(f"Processed k-NN for {i}/{n} nodes ({100 * i / n:.1f}%)")

    # Add as vertex attribute
    G.vs["knn_dist"] = [knn_distances.get(i, 0) for i in range(n)]

    logger.info(
        f"k-NN distance computation completed - "
        f"Avg {k}-NN distance: {np.mean(list(knn_distances.values())):.2f}"
    )

    return knn_distances


def farness_centrality(G, weight=None, n=None):
    logger.info(
        f"Computing farness centrality for graph with {G.vcount()} nodes and {G.ecount()} edges"
    )

    # Compute farness and normalized farness for each node
    farness = {}
    norm_farness = {}
    n = G.vcount()

    logger.debug(f"Using weight attribute: {weight}")

    # Use igraph's shortest path distances
    logger.info("Calculating shortest path distances matrix...")
    distances = G.distances()
    logger.info("Distance matrix calculation completed")

    for i in range(n):
        # Sum of distances to all reachable nodes except itself
        total_dist = sum(
            dist
            for j, dist in enumerate(distances[i])
            if i != j and dist != float("inf")
        )
        farness[i] = total_dist
        # Normalize by number of reachable nodes minus one (excluding itself)
        reachable = sum(1 for dist in distances[i] if dist != float("inf")) - 1
        norm_farness[i] = total_dist / reachable if reachable > 0 else 0

        if i % max(1, n // 10) == 0:  # Log progress every 10%
            logger.debug(f"Processed {i}/{n} nodes ({100 * i / n:.1f}%)")

    # Add as vertex attributes
    G.vs["farness"] = [farness.get(i, 0) for i in range(n)]
    G.vs["norm_farness"] = [norm_farness.get(i, 0) for i in range(n)]

    logger.info(
        f"Farness centrality computation completed - "
        f"Avg farness: {np.mean(list(farness.values())):.2f}, "
        f"Max farness: {max(farness.values()):.2f}"
    )

    knn_dist = get_knn_distance(G, weight, n)

    return G, farness, knn_dist


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


def straightness_centrality(g: ig.Graph, weight: str = "weight"):
    """
    Compute straightness centrality for all nodes in a graph.

    Parameters
    ----------
    g : ig.Graph
        Road network graph (undirected or directed). Each node must
        have attributes "x" and "y".
    weight : str
        Edge attribute to use as distance (default = "weight"").

    Returns
    -------
    straightness : list of floats
        Straightness centrality for each node.
    """
    n = g.vcount()
    straightness = [0.0] * n

    # Extract node coordinates from attributes
    coords = {v.index: (v["x"], v["y"]) for v in g.vs}

    # Precompute all shortest path lengths
    dG = g.shortest_paths_dijkstra(weights=weight)

    for i in range(n):
        si = 0
        valid = 0
        xi, yi = coords[i]
        for j in range(n):
            if i == j:
                continue
            d_g = dG[i][j]
            if d_g == float("inf"):
                continue  # disconnected
            xj, yj = coords[j]
            d_e = np.hypot(xi - xj, yi - yj)
            if d_e > 0:
                si += d_e / d_g
                valid += 1
        straightness[i] = si / valid if valid > 0 else 0.0

    return straightness


def graph_straightness(g: ig.Graph, weight: str = "weight"):
    """
    Compute global straightness centrality for a graph.

    Parameters
    ----------
    g : ig.Graph
        Road network graph (undirected or directed). Each node must
        have attributes "x" and "y".
    weight : str
        Edge attribute to use as distance (default = "weight"").

    Returns
    -------
    float
        Global straightness centrality (graph-level detour index).
    """
    n = g.vcount()

    # Extract node coordinates
    coords = {v.index: (v["x"], v["y"]) for v in g.vs}

    # All-pairs shortest paths
    dG = g.shortest_paths_dijkstra(weights=weight)

    num, den = 0.0, 0

    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            if i == j:
                continue
            d_g = dG[i][j]
            if d_g == float("inf"):
                continue  # disconnected pair
            xj, yj = coords[j]
            d_e = np.hypot(xi - xj, yi - yj)
            if d_e > 0:
                num += d_e / d_g
                den += 1

    return num / den if den > 0 else 0.0
