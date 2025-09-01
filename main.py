import random

from config import Config
from centrality import nodes_highest_avg_knn_distance_ig, graph_straightness
from utils import (
    get_gas_stations_from_graph,
    convert_networkx_to_igraph,
    igraph_edges_to_gpkg,
    ig_nodes_to_gpkg,
    nx_knn_nodes_to_gpkg,
)
from utils import prune_igraph_by_distance
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
    Config.ensure_directories()
    Config.validate_config()

    # prepare network and stations
    road_filepath = Config.get_road_filepath()
    G_road_nx = ox.load_graphml(road_filepath)
    G_road_nx.remove_edges_from(nx.selfloop_edges(G_road_nx))

    stations_nx = get_gas_stations_from_graph(G_road_nx)

    G_road_ig = convert_networkx_to_igraph(G_road_nx)
    name_to_idx = {v["name"]: v.index for v in G_road_ig.vs}
    stations_ig = [name_to_idx[str(nx_node)] for nx_node in stations_nx if str(nx_node) in name_to_idx]
    stations_ig = list(set(stations_ig)) 
    
    # 1. Prune base network
    G_road_ig = prune_igraph_by_distance(G_road_ig, stations_ig, Config.MAX_DISTANCE)

    # 2. Prune base network based on knn stations
    stations_knn_ig = nodes_highest_avg_knn_distance_ig(G_road_ig, knn=Config.K_NN, n=Config.N_REMOVE, node_subset=stations_ig)
    remaining_stations_knn_ig = set(stations_ig) - set(stations_knn_ig)
    G_road_filtered_ig = prune_igraph_by_distance(G_road_ig.copy(), remaining_stations_knn_ig, Config.MAX_DISTANCE)

    # 3. Prune base network based on random stations
    random.seed(Config.RANDOM_SEED)
    random_stations_ig = random.sample(stations_ig, Config.N_REMOVE)
    remaining_stations_random_ig = set(stations_ig) - set(random_stations_ig)
    G_road_random_ig = prune_igraph_by_distance(G_road_ig.copy(), remaining_stations_random_ig, Config.MAX_DISTANCE)
    
    # Assertions to check validity of analysis
    assert len(stations_ig) == len(stations_nx)
    assert random_stations_ig != stations_knn_ig
    assert G_road_random_ig is not None
    assert G_road_filtered_ig is not None

    # Save the things for visualization
    logger.info(f"Exporting igraph edges to GeoPackage...")
    igraph_edges_to_gpkg(G_road_random_ig, "random")
    igraph_edges_to_gpkg(G_road_filtered_ig, "knn")
    igraph_edges_to_gpkg(G_road_ig, "base")

    logger.info(f"Exporting igraph nodes to GeoPackage...")
    ig_nodes_to_gpkg(G_road_ig, remaining_stations_knn_ig, "knn_remaining")
    ig_nodes_to_gpkg(G_road_ig, remaining_stations_random_ig, "random_remaining")
    ig_nodes_to_gpkg(G_road_ig, stations_ig, "all_stations")

    """graphs = {
        "Original Road Graph": G_road_ig,
        "Filtered Road Graph": G_road_filtered_ig,
        "Randomized Road Graph": G_road_random_ig
    }

    logger.info("=== Centrality Measures ===")
    for name, graph in graphs.items():
        closeness = graph.closeness()
        betweenness = graph.betweenness()
        degree = graph.degree()
        straightness = graph_straightness(graph)
        logger.info(f"{name} - Closeness (avg): {sum(closeness)/len(closeness):.4f}")
        logger.info(f"{name} - Betweenness (avg): {sum(betweenness)/len(betweenness):.4f}")
        logger.info(f"{name} - Degree (avg): {sum(degree)/len(degree):.4f}")
        logger.info(f"{name} - Straightness (avg): {straightness:.4f}")

    end = time.time()
    elapsed = end - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Total time taken: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    logger.info(f"Nodes: {G_road_ig.vcount()}")
    logger.info(f"Edges: {G_road_ig.ecount()}")"""


if __name__ == "__main__":
    logger.info("Starting fuel station centrality analysis...")
    main()
    logger.info("Analysis complete.")
