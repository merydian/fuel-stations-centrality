"""
Enhanced main module with improved error handling and code structure.
"""

import random

from config import Config
from centrality import get_knn_dists
import stats_cpp
from utils import (
    get_gas_stations_from_graph,
    convert_networkx_to_igraph,
)
import osmnx as ox
import networkx as nx


def main():
    """Main function for fuel station centrality analysis."""
    Config.ensure_directories()
    Config.validate_config()

    (Config.LOCAL_PBF_PATH.stem if hasattr(Config, "LOCAL_PBF_PATH") else "unknown")

    road_filepath = Config.get_road_filepath()
    G_road = ox.load_graphml(road_filepath)
    G_road.remove_edges_from(nx.selfloop_edges(G_road))

    stations = get_gas_stations_from_graph(G_road)

    print("Gas stations found:", stations)

    G_stations = G_road.subgraph(stations).copy()

    G_stations = convert_networkx_to_igraph(G_stations)

    G_stations, knn_dist = get_knn_dists(G_stations, weight="weight", n=Config.K_NN)

    G_road_ig = convert_networkx_to_igraph(G_road)

    sorted_stations = sorted(knn_dist.items(), key=lambda x: x[1], reverse=True)
    stations_to_remove = [
        station_id for station_id, _ in sorted_stations[: Config.N_REMOVE]
    ]
    G_road_filtered = G_road.copy()
    G_road_filtered.remove_nodes_from(stations_to_remove)
    G_road_filtered_ig = convert_networkx_to_igraph(G_road_filtered)
    ig_indices = [
        v.index for v in G_road_filtered_ig.vs if v["name"] in stations_to_remove
    ]
    G_road_filtered_ig = G_road_filtered_ig.delete_vertices(ig_indices)

    random.seed(Config.RANDOM_SEED)
    random_stations_to_remove = random.sample(
        stations, min(Config.N_REMOVE, len(stations))
    )
    assert random_stations_to_remove != stations_to_remove
    G_road_ig_random = G_road.copy()
    G_road_ig_random.remove_nodes_from(random_stations_to_remove)
    G_road_ig_random_ig = convert_networkx_to_igraph(G_road_ig_random)
    ig_indices = [
        v.index
        for v in G_road_ig_random_ig.vs
        if v["name"] in random_stations_to_remove
    ]
    G_road_ig_random_ig = G_road_ig_random_ig.delete_vertices(ig_indices)

    old_stats = stats_cpp.GraphStatsCalculator.get_graph_stats(G_road_ig)
    smart_stats = stats_cpp.GraphStatsCalculator.get_graph_stats(G_road_ig)
    random_stats = stats_cpp.GraphStatsCalculator.get_graph_stats(G_road_ig_random)

    print("Old stats:", old_stats)
    print("Smart stats:", smart_stats)
    print("Random stats:", random_stats)


if __name__ == "__main__":
    main()
