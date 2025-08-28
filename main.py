import random

from config import Config
from centrality import nodes_highest_avg_knn_distance_nx, graph_straightness
from utils import (
    get_gas_stations_from_graph,
    convert_networkx_to_igraph,
    igraph_edges_to_gpkg,
    nx_nodes_to_gpkg,
)
import osmnx as ox
import networkx as nx
import time


def main():
    """Main function for fuel station centrality analysis."""
    start = time.time()
    Config.ensure_directories()
    Config.validate_config()

    (Config.LOCAL_PBF_PATH.stem if hasattr(Config, "LOCAL_PBF_PATH") else "unknown")

    road_filepath = Config.get_road_filepath()
    G_road = ox.load_graphml(road_filepath)
    G_road.remove_edges_from(nx.selfloop_edges(G_road))

    stations = get_gas_stations_from_graph(G_road)

    stations_to_remove = nodes_highest_avg_knn_distance_nx(G_road, knn=Config.K_NN, n=Config.N_REMOVE, node_subset=stations)

    G_road_filtered = G_road.copy()
    G_road_filtered.remove_nodes_from(stations_to_remove)
    G_road_filtered_ig = convert_networkx_to_igraph(G_road_filtered)
    ig_indices = [
        v.index for v in G_road_filtered_ig.vs if v["name"] in stations_to_remove
    ]
    G_road_filtered_ig.delete_vertices(ig_indices)

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
    G_road_ig_random_ig.delete_vertices(ig_indices)

    assert G_road_ig_random_ig is not None
    assert G_road_filtered_ig is not None

    igraph_edges_to_gpkg(G_road_ig_random_ig, "random")
    igraph_edges_to_gpkg(G_road_filtered_ig, "knn")
    G_road_ig = convert_networkx_to_igraph(G_road)
    igraph_edges_to_gpkg(G_road_ig, "base")

    nx_nodes_to_gpkg(G_road, stations_to_remove, "knn")
    nx_nodes_to_gpkg(G_road, random_stations_to_remove, "random")
    nx_nodes_to_gpkg(G_road, stations, "all_stations")

    graphs = {
        "Original Road Graph": G_road_ig,
        "Filtered Road Graph": G_road_filtered_ig,
        "Randomized Road Graph": G_road_ig_random_ig
    }

    print("=== Straightness Centrality ===")
    for name, graph in graphs.items():
        centrality = graph_straightness(graph)
        print(f"{name}: {centrality:.4f}")  # rounds to 4 decimal places

    end = time.time()
    elapsed = end - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total time taken: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


if __name__ == "__main__":
    main()
