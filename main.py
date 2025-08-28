"""
Enhanced main module with improved error handling and code structure.
"""

import time
import sys
import random
from datetime import datetime

from config import Config
from centrality import get_knn_dists
import stats_cpp
from utils import (
    save_graph_to_geopackage,
    remove_edges_far_from_stations_graph,
    get_gas_stations_from_graph,
    find_stations_in_road_network,
    convert_networkx_to_igraph,
    save_removed_stations_to_geopackage,
    save_stations_to_geopackage,
    make_graph_from_stations,
    process_fuel_stations,
    remove_stations_from_road_network
)
import osmnx as ox
import networkx as nx


def main():
    """Main function for fuel station centrality analysis."""
    Config.ensure_directories()
    Config.validate_config()

    pbf_stem = Config.LOCAL_PBF_PATH.stem if hasattr(Config, "LOCAL_PBF_PATH") else "unknown"

    road_filepath = Config.get_road_filepath()
    G_road = ox.load_graphml(road_filepath)
    G_road.remove_edges_from(nx.selfloop_edges(G_road))

    stations = get_gas_stations_from_graph()
    stations = process_fuel_stations(stations)

    save_stations_to_geopackage(
        stations, out_file=f"all_gas_stations_{Config.get_road_filename()}.gpkg"
    )

    station_to_node_mapping = find_stations_in_road_network(G_road, stations)

    G_stations = make_graph_from_stations(
        stations,
        use_ors=False,
        G_road=G_road,
        station_to_node_mapping=station_to_node_mapping,
    )

    G_stations, knn_dist = get_knn_dists(
        G_stations, weight="weight", n=Config.K_NN
    )

    G_road_ig = convert_networkx_to_igraph(G_road)

    old_stats = stats_cpp.GraphStatsCalculator.get_graph_stats(G_road_ig)

    save_graph_to_geopackage(
        G_road_ig,
        out_file=f"road_network_baseline_{pbf_stem}.gpkg",
    )

    sorted_stations = sorted(knn_dist.items(), key=lambda x: x[1], reverse=True)
    stations_to_remove = [
        station_id for station_id, _ in sorted_stations[: Config.N_REMOVE]
    ]

    save_removed_stations_to_geopackage(
        stations,
        stations_to_remove,
        out_file=f"removed_stations_smart_{pbf_stem}.gpkg",
        removal_type="smart_knn",
        knn_dist=knn_dist,
    )

    G_road_filtered = remove_stations_from_road_network(G_road, station_to_node_mapping, stations_to_remove)
    G_road_ig = convert_networkx_to_igraph(G_road_filtered)
    
    G_road_ig, edges_removed = remove_edges_far_from_stations_graph(
        G_road_ig, stations, Config.MAX_DISTANCE, station_to_node_mapping
    )

    random.seed(Config.RANDOM_SEED)
    all_station_indices = list(station_to_node_mapping.keys())
    random_stations_to_remove = random.sample(
        all_station_indices, min(Config.N_REMOVE, len(all_station_indices))
    )
    assert random_stations_to_remove != stations_to_remove

    save_removed_stations_to_geopackage(
        stations,
        random_stations_to_remove,
        out_file=f"removed_stations_random_{pbf_stem}.gpkg",
        removal_type="random",
        knn_dist=knn_dist,
    )

    G_road_random_filtered = remove_stations_from_road_network(G_road, station_to_node_mapping, random_stations_to_remove)

    G_road_ig_random = convert_networkx_to_igraph(G_road_random_filtered)

    smart_stats = stats_cpp.GraphStatsCalculator.get_graph_stats(G_road_ig, base_convex_hull=None)
    random_stats = stats_cpp.GraphStatsCalculator.get_graph_stats(G_road_ig_random, base_convex_hull=None)

    save_graph_to_geopackage(
        G_road_ig,
        out_file=f"road_network_smart_filtered_{pbf_stem}.gpkg",
    )
    save_graph_to_geopackage(
        G_road_ig_random,
        out_file=f"road_network_random_filtered_{pbf_stem}.gpkg",
    )

if __name__ == "__main__":
    main()
