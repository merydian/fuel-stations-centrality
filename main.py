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
import logging
import sys

# Configure logging at the very beginning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to console
        logging.FileHandler('fuel_stations_analysis.log')  # Also save to file
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function for fuel station centrality analysis."""
    start = time.time()
    Config.ensure_directories()
    Config.validate_config()

    road_filepath = Config.get_road_filepath()
    logger.info(f"Loading road network from {road_filepath}")
    G_road = ox.load_graphml(road_filepath)
    G_road.remove_edges_from(nx.selfloop_edges(G_road))

    logger.info("Extracting gas stations from OSM...")
    stations = get_gas_stations_from_graph(G_road)

    logger.info(f"Total gas stations found: {len(stations)}")
    logger.info(f"Removing {Config.N_REMOVE} gas stations based on highest avg {Config.K_NN}-NN distance...")
    stations_to_remove = nodes_highest_avg_knn_distance_nx(G_road, knn=Config.K_NN, n=Config.N_REMOVE, node_subset=stations)
    logger.info(f"Stations to remove: {stations_to_remove}")

    G_road_filtered = G_road.copy()
    G_road_filtered.remove_nodes_from(stations_to_remove)
    logger.info(f"Before removal - stations: {len(stations)}")
    logger.info(f"Remaining stations: {len(stations) - len(stations_to_remove)}")
    logger.info(f"Before removal - nodes: {len(G_road.nodes)}, edges: {len(G_road.edges)}")
    logger.info(f"After removal - nodes: {len(G_road_filtered.nodes)}, edges: {len(G_road_filtered.edges)}")
    logger.info("Converting graphs to igraph format for centrality calculations...")
    G_road_filtered_ig = convert_networkx_to_igraph(G_road_filtered)
    ig_indices = [
        v.index for v in G_road_filtered_ig.vs if v["name"] in stations_to_remove
    ]
    logger.info(f"Deleting {len(ig_indices)} vertices corresponding to removed stations...")
    G_road_filtered_ig.delete_vertices(ig_indices)

    random.seed(Config.RANDOM_SEED)
    logger.info(f"Selecting {Config.N_REMOVE} random stations to remove...")
    random_stations_to_remove = random.sample(stations, Config.N_REMOVE)
    assert random_stations_to_remove != stations_to_remove

    G_road_ig_random = G_road.copy()
    logger.info(f"Removing random stations: {random_stations_to_remove}")
    G_road_ig_random.remove_nodes_from(random_stations_to_remove)
    logger.info(f"Remaining stations: {len(stations) - len(random_stations_to_remove)}")
    logger.info(f"Remaining nodes in graph: {len(G_road_ig_random.nodes)}")
    logger.info(f"Remaining edges in graph: {len(G_road_ig_random.edges)}")
    logger.info("Converting graphs to igraph format for centrality calculations...")
    G_road_ig_random_ig = convert_networkx_to_igraph(G_road_ig_random)
    ig_indices = [
        v.index
        for v in G_road_ig_random_ig.vs
        if v["name"] in random_stations_to_remove
    ]
    logger.info(f"Deleting {len(ig_indices)} vertices corresponding to removed stations...")
    G_road_ig_random_ig.delete_vertices(ig_indices)

    assert G_road_ig_random_ig is not None
    assert G_road_filtered_ig is not None

    logger.info(f"Exporting igraph edges to GeoPackage...")
    igraph_edges_to_gpkg(G_road_ig_random_ig, "random")
    igraph_edges_to_gpkg(G_road_filtered_ig, "knn")
    G_road_ig = convert_networkx_to_igraph(G_road)
    igraph_edges_to_gpkg(G_road_ig, "base")

    logger.info(f"Exporting igraph nodes to GeoPackage...")
    nx_nodes_to_gpkg(G_road, stations_to_remove, "knn")
    nx_nodes_to_gpkg(G_road, random_stations_to_remove, "random")
    nx_nodes_to_gpkg(G_road, stations, "all_stations")

    graphs = {
        "Original Road Graph": G_road_ig,
        "Filtered Road Graph": G_road_filtered_ig,
        "Randomized Road Graph": G_road_ig_random_ig
    }

    logger.info("=== Straightness Centrality ===")
    for name, graph in graphs.items():
        centrality = graph_straightness(graph)
        logger.info(f"{name}: {centrality:.4f}")  # rounds to 4 decimal places

    end = time.time()
    elapsed = end - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Total time taken: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    logger.info(f"Nodes: {G_road_ig.vcount()}")
    logger.info(f"Edges: {G_road_ig.ecount()}")


if __name__ == "__main__":
    logger.info("Starting fuel station centrality analysis...")
    main()
    print("Analysis complete.")
