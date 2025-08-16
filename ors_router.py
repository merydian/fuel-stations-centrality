import openrouteservice as ors
import geopandas as gpd
from typing import List, Tuple, Optional
import networkx as nx
import numpy as np

MAX_LOCATIONS = 59


def make_graph_from_stations(
    stations: gpd.GeoDataFrame,
    api_key: str,
    profile: str = "driving-car",
    max_distance: Optional[float] = None,
) -> nx.Graph:
    if stations.empty:
        raise ValueError("Stations GeoDataFrame cannot be empty")

    if "geometry" not in stations.columns:
        raise ValueError("Stations must have a 'geometry' column")

    # Extract coordinates from stations (longitude, latitude)
    locations = []
    station_indices = []

    for idx, station in stations.iterrows():
        geom = station.geometry
        if geom is not None:
            # Get centroid for non-point geometries
            if hasattr(geom, "centroid"):
                pass
                point = geom.centroid
            else:
                point = geom

            locations.append((point.x, point.y))  # (longitude, latitude)
            station_indices.append(idx)

    if len(locations) < 2:
        raise ValueError("At least 2 valid station locations are required")

    n = len(locations)

    # --- Helper: chunk indices ---
    def chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    chunk_indices = list(chunks(range(n), MAX_LOCATIONS))

    # --- ORS client ---
    client = ors.Client(key=api_key)

    # Prepare adjacency matrix
    distances = np.zeros((n, n))

    # --- Loop over chunk pairs ---
    for src_chunk in chunk_indices:
        for dst_chunk in chunk_indices:
            coords_src = [locations[i] for i in src_chunk]
            coords_dst = [locations[i] for i in dst_chunk]

            combined_coords = coords_src + coords_dst
            sources_idx = list(range(len(coords_src)))
            destinations_idx = list(
                range(len(coords_src), len(coords_src) + len(coords_dst))
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
                    distances[src_i, dst_j] = matrix["distances"][i][j]

    G = nx.DiGraph()

    # Add nodes with coordinates
    for idx, coord in enumerate(locations):
        G.add_node(idx, coord=coord)

    # Add edges with distances
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, weight=distances[i, j])  # weight in meters

    return G
