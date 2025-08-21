import openrouteservice as ors
import geopandas as gpd
import igraph as ig
import numpy as np
import logging

logger = logging.getLogger(__name__)

MAX_LOCATIONS = 59


def make_graph_from_stations_via_road_network(
    stations: gpd.GeoDataFrame,
    G_road,
    station_to_node_mapping: dict,
) -> ig.Graph:
    """
    Create a graph from stations using distances calculated via the road network.

    Parameters
    ----------
    stations : gpd.GeoDataFrame
        GeoDataFrame containing fuel stations
    G_road : NetworkX Graph
        Road network graph
    station_to_node_mapping : dict
        Mapping from station indices to road network node IDs

    Returns
    -------
    ig.Graph
        igraph Graph with stations connected by road network distances
    """
    logger.info(
        f"Creating graph from {len(stations)} fuel stations using road network distances"
    )

    if stations.empty:
        logger.error("Stations GeoDataFrame is empty")
        raise ValueError("Stations GeoDataFrame cannot be empty")

    if len(stations) < 2:
        logger.error(f"Insufficient stations: {len(stations)}")
        raise ValueError("At least 2 stations are required")

    # Extract coordinates and valid station indices
    locations = []
    station_indices = []
    road_nodes = []

    for idx, station in stations.iterrows():
        if idx in station_to_node_mapping:
            geom = station.geometry
            if geom is not None:
                # Get centroid for non-point geometries
                if hasattr(geom, "centroid"):
                    point = geom.centroid
                else:
                    point = geom

                locations.append((point.x, point.y))  # (longitude, latitude)
                station_indices.append(idx)
                road_nodes.append(station_to_node_mapping[idx])

    n = len(locations)
    logger.info(f"Found {n} stations with valid road network mappings")

    if n < 2:
        raise ValueError("At least 2 valid station mappings are required")

    # Calculate shortest path distances between all station pairs using road network
    logger.info("Computing shortest path distances via road network...")
    distances = np.full((n, n), np.inf)

    try:
        import networkx as nx

        # Compute all-pairs shortest paths for the station nodes
        for i, source_node in enumerate(road_nodes):
            if source_node in G_road:
                # Compute shortest paths from this source to all other stations
                try:
                    path_lengths = nx.single_source_dijkstra_path_length(
                        G_road, source_node, weight="length"
                    )

                    for j, target_node in enumerate(road_nodes):
                        if target_node in path_lengths:
                            distances[i, j] = path_lengths[target_node]

                except nx.NetworkXNoPath:
                    # Some nodes might not be reachable
                    pass

            if (i + 1) % max(1, n // 10) == 0:
                logger.debug(f"Computed distances from {i + 1}/{n} stations")

    except Exception as e:
        logger.error(f"Failed to compute road network distances: {e}")
        raise

    logger.info("Road network distance calculation completed")

    # Create igraph Graph
    logger.info("Creating igraph directed graph...")
    G = ig.Graph(directed=True)

    # Add vertices
    G.add_vertices(n)
    logger.debug(f"Added {n} vertices to graph")

    # Set coordinates as vertex attributes
    for idx, coord in enumerate(locations):
        G.vs[idx]["x"] = coord[0]  # longitude
        G.vs[idx]["y"] = coord[1]  # latitude
        G.vs[idx]["station_id"] = station_indices[idx]  # Original station index
        G.vs[idx]["road_node"] = road_nodes[idx]  # Corresponding road network node

    # Add edges with distances
    logger.info("Adding edges with distance weights...")
    edges = []
    weights = []
    lengths = []
    finite_edges = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((i, j))
                weight = distances[i, j]
                weights.append(weight)
                lengths.append(weight)
                if weight != np.inf:
                    finite_edges += 1

    G.add_edges(edges)
    G.es["weight"] = weights
    G.es["length"] = lengths

    logger.info(
        f"Graph creation completed: {G.vcount()} vertices, {G.ecount()} edges "
        f"({finite_edges} with finite weights, {G.ecount() - finite_edges} infinite)"
    )

    return G


def make_graph_from_stations(
    stations: gpd.GeoDataFrame,
    api_key: str = None,
    profile: str = "driving-car",
    use_ors: bool = True,
    G_road=None,
    station_to_node_mapping: dict = None,
) -> ig.Graph:
    """
    Create a graph from stations using either OpenRouteService or road network distances.

    Parameters
    ----------
    stations : gpd.GeoDataFrame
        GeoDataFrame containing fuel stations
    api_key : str, optional
        OpenRouteService API key (required if use_ors=True)
    profile : str, optional
        OpenRouteService routing profile (default: "driving-car")
    use_ors : bool, optional
        Whether to use OpenRouteService (True) or road network (False)
    G_road : NetworkX Graph, optional
        Road network graph (required if use_ors=False)
    station_to_node_mapping : dict, optional
        Mapping from station indices to road network nodes (required if use_ors=False)

    Returns
    -------
    ig.Graph
        igraph Graph with stations connected by calculated distances
    """
    if use_ors:
        if not api_key:
            raise ValueError("API key is required when use_ors=True")
        logger.info("Using OpenRouteService for station distance calculation")
        return make_graph_from_stations_ors(stations, api_key, profile)
    else:
        if G_road is None or station_to_node_mapping is None:
            raise ValueError(
                "G_road and station_to_node_mapping are required when use_ors=False"
            )
        logger.info("Using road network for station distance calculation")
        return make_graph_from_stations_via_road_network(
            stations, G_road, station_to_node_mapping
        )


def make_graph_from_stations_ors(
    stations: gpd.GeoDataFrame,
    api_key: str,
    profile: str = "driving-car",
) -> ig.Graph:
    """
    Create a graph from stations using OpenRouteService API.
    This is the original implementation renamed for clarity.
    """
    logger.info(
        f"Creating graph from {len(stations)} fuel stations using profile: {profile}"
    )

    if stations.empty:
        logger.error("Stations GeoDataFrame is empty")
        raise ValueError("Stations GeoDataFrame cannot be empty")

    if "geometry" not in stations.columns:
        logger.error("Stations GeoDataFrame missing 'geometry' column")
        raise ValueError("Stations must have a 'geometry' column")

    # Extract coordinates from stations (longitude, latitude)
    locations = []
    station_indices = []

    logger.debug("Extracting coordinates from station geometries...")
    for idx, station in stations.iterrows():
        geom = station.geometry
        if geom is not None:
            # Get centroid for non-point geometries
            if hasattr(geom, "centroid"):
                point = geom.centroid
            else:
                point = geom

            locations.append((point.x, point.y))  # (longitude, latitude)
            station_indices.append(idx)

    if len(locations) < 2:
        logger.error(f"Insufficient valid locations: {len(locations)}")
        raise ValueError("At least 2 valid station locations are required")

    n = len(locations)
    logger.info(f"Successfully extracted {n} valid station locations")

    # --- Helper: chunk indices ---
    def chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    chunk_indices = list(chunks(range(n), MAX_LOCATIONS))
    total_chunks = len(chunk_indices)
    logger.info(
        f"Split into {total_chunks} chunks of max {MAX_LOCATIONS} locations each"
    )

    # --- ORS client ---
    logger.debug("Initializing OpenRouteService client...")
    try:
        client = ors.Client(key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize ORS client: {e}")
        raise

    # Prepare adjacency matrix
    distances = np.zeros((n, n))
    logger.info("Initialized distance matrix")

    # --- Loop over chunk pairs ---
    total_requests = total_chunks * total_chunks
    completed_requests = 0

    for src_chunk_idx, src_chunk in enumerate(chunk_indices):
        for dst_chunk_idx, dst_chunk in enumerate(chunk_indices):
            completed_requests += 1
            logger.debug(
                f"Processing chunk pair {completed_requests}/{total_requests} "
                f"(src: {src_chunk_idx + 1}/{total_chunks}, dst: {dst_chunk_idx + 1}/{total_chunks})"
            )

            coords_src = [locations[i] for i in src_chunk]
            coords_dst = [locations[i] for i in dst_chunk]

            combined_coords = coords_src + coords_dst
            sources_idx = list(range(len(coords_src)))
            destinations_idx = list(
                range(len(coords_src), len(coords_src) + len(coords_dst))
            )

            try:
                logger.debug(
                    f"Requesting distance matrix for {len(coords_src)} sources to {len(coords_dst)} destinations"
                )
                matrix = client.distance_matrix(
                    locations=combined_coords,
                    profile=profile,
                    metrics=["distance"],
                    sources=sources_idx,
                    destinations=destinations_idx,
                )

                for i, src_i in enumerate(src_chunk):
                    for j, dst_j in enumerate(dst_chunk):
                        distances[src_i, dst_j] = (
                            matrix["distances"][i][j]
                            if matrix["distances"][i][j]
                            else np.inf
                        )

            except Exception as e:
                logger.error(
                    f"Failed to get distance matrix for chunk pair {src_chunk_idx}-{dst_chunk_idx}: {e}"
                )
                # Fill with infinity for failed requests
                for src_i in src_chunk:
                    for dst_j in dst_chunk:
                        distances[src_i, dst_j] = np.inf

            if completed_requests % max(1, total_requests // 10) == 0:
                logger.info(
                    f"Distance matrix progress: {completed_requests}/{total_requests} "
                    f"({100 * completed_requests / total_requests:.1f}%)"
                )

    logger.info("Distance matrix calculation completed")

    # Create igraph Graph
    logger.info("Creating igraph directed graph...")
    G = ig.Graph(directed=True)

    # Add vertices
    G.add_vertices(n)
    logger.debug(f"Added {n} vertices to graph")

    # Set coordinates as vertex attributes
    for idx, coord in enumerate(locations):
        G.vs[idx]["x"] = coord[0]  # longitude
        G.vs[idx]["y"] = coord[1]  # latitude

    # Add edges with distances
    logger.info("Adding edges with distance weights...")
    edges = []
    weights = []
    lengths = []
    finite_edges = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((i, j))
                weight = distances[i, j]
                weights.append(weight)
                lengths.append(
                    weight
                )  # Store as both weight and length for compatibility
                if weight != np.inf:
                    finite_edges += 1

    G.add_edges(edges)
    G.es["weight"] = weights
    G.es["length"] = lengths  # Add length attribute for edge filtering

    logger.info(
        f"Graph creation completed: {G.vcount()} vertices, {G.ecount()} edges "
        f"({finite_edges} with finite weights, {G.ecount() - finite_edges} infinite)"
    )

    return G
