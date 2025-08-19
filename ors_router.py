import openrouteservice as ors
import geopandas as gpd
import igraph as ig
import numpy as np
import logging

logger = logging.getLogger(__name__)

MAX_LOCATIONS = 59


def make_graph_from_stations(
    stations: gpd.GeoDataFrame,
    api_key: str,
    profile: str = "driving-car",
) -> ig.Graph:
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
        G.vs[idx]["coord"] = coord

    # Add edges with distances
    logger.info("Adding edges with distance weights...")
    edges = []
    weights = []
    finite_edges = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((i, j))
                weight = distances[i, j]
                weights.append(weight)
                if weight != np.inf:
                    finite_edges += 1

    G.add_edges(edges)
    G.es["weight"] = weights

    logger.info(
        f"Graph creation completed: {G.vcount()} vertices, {G.ecount()} edges "
        f"({finite_edges} with finite weights, {G.ecount() - finite_edges} infinite)"
    )

    return G
