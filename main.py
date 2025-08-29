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

    road_filepath = Config.get_road_filepath()
    print(f"Loading road network from {road_filepath}")
    G_road = ox.load_graphml(road_filepath)
    G_road.remove_edges_from(nx.selfloop_edges(G_road))

    print("Extracting gas stations from OSM...")
    stations = get_gas_stations_from_graph(G_road)

    print(f"Total gas stations found: {len(stations)}")
    print(f"Removing {Config.N_REMOVE} gas stations based on highest avg {Config.K_NN}-NN distance...")
    stations_to_remove = nodes_highest_avg_knn_distance_nx(G_road, knn=Config.K_NN, n=Config.N_REMOVE, node_subset=stations)
    print(f"Stations to remove: {stations_to_remove}")

    G_road_filtered = G_road.copy()
    G_road_filtered.remove_nodes_from(stations_to_remove)
    print(f"Before removal - stations: {len(stations)}")
    print(f"Remaining stations: {len(stations) - len(stations_to_remove)}")
    print(f"Before removal - nodes: {len(G_road.nodes)}, edges: {len(G_road.edges)}")
    print(f"After removal - nodes: {len(G_road_filtered.nodes)}, edges: {len(G_road_filtered.edges)}")
    print("Converting graphs to igraph format for centrality calculations...")
    G_road_filtered_ig = convert_networkx_to_igraph(G_road_filtered)
    ig_indices = [
        v.index for v in G_road_filtered_ig.vs if v["name"] in stations_to_remove
    ]
    print(f"Deleting {len(ig_indices)} vertices corresponding to removed stations...")
    G_road_filtered_ig.delete_vertices(ig_indices)

    random.seed(Config.RANDOM_SEED)
    print(f"Selecting {Config.N_REMOVE} random stations to remove...")
    random_stations_to_remove = random.sample(stations, Config.N_REMOVE)
    assert random_stations_to_remove != stations_to_remove

    G_road_ig_random = G_road.copy()
    print(f"Removing random stations: {random_stations_to_remove}")
    G_road_ig_random.remove_nodes_from(random_stations_to_remove)
    print(f"Remaining stations: {len(stations) - len(random_stations_to_remove)}")
    print(f"Remaining nodes in graph: {len(G_road_ig_random.nodes)}")
    print(f"Remaining edges in graph: {len(G_road_ig_random.edges)}")
    print("Converting graphs to igraph format for centrality calculations...")
    G_road_ig_random_ig = convert_networkx_to_igraph(G_road_ig_random)
    ig_indices = [
        v.index
        for v in G_road_ig_random_ig.vs
        if v["name"] in random_stations_to_remove
    ]
    print(f"Deleting {len(ig_indices)} vertices corresponding to removed stations...")
    G_road_ig_random_ig.delete_vertices(ig_indices)

    assert G_road_ig_random_ig is not None
    assert G_road_filtered_ig is not None

    print(f"Exporting igraph edges to GeoPackage...")
    igraph_edges_to_gpkg(G_road_ig_random_ig, "random")
    igraph_edges_to_gpkg(G_road_filtered_ig, "knn")
    G_road_ig = convert_networkx_to_igraph(G_road)
    igraph_edges_to_gpkg(G_road_ig, "base")

    print(f"Exporting igraph nodes to GeoPackage...")
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
    print("Nodes:", G_road_ig.vcount())
    print("Edges:", G_road_ig.ecount())


if __name__ == "__main__":
    main()
