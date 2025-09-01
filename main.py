import random

from config import Config
from centrality import nodes_highest_avg_knn_distance_nx, graph_straightness
from utils import (
    get_gas_stations_from_graph,
    convert_networkx_to_igraph,
    igraph_edges_to_gpkg,
    nx_nodes_to_gpkg,
)
from utils import prune_graph_by_distance
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
    G_road_nx = ox.load_graphml(road_filepath)
    G_road_nx.remove_edges_from(nx.selfloop_edges(G_road_nx))

    stations_nx = get_gas_stations_from_graph(G_road_nx)

    logger.info(f"Total gas stations found: {len(stations_nx)}")
    logger.info(f"Removing {Config.N_REMOVE} gas stations based on highest avg {Config.K_NN}-NN distance...")
    stations_knn_nx = nodes_highest_avg_knn_distance_nx(G_road_nx, knn=Config.K_NN, n=Config.N_REMOVE, node_subset=stations_nx)
    logger.info(f"Stations to remove: {stations_knn_nx}")

    logger.info(f"Remove far edges further than {Config.MAX_DISTANCE} from graph from base graph...")
    G_road_nx = prune_graph_by_distance(G_road_nx, stations_nx, Config.MAX_DISTANCE)
    
    G_road_filtered_nx = G_road_nx.copy()
    # G_road_filtered_nx.remove_nodes_from(stations_knn_nx)
    remaining_stations_knn_nx = set(stations_nx) - set(stations_knn_nx)
    logger.info(f"Remove far edges further than {Config.MAX_DISTANCE} from graph...")
    G_road_filtered_nx = prune_graph_by_distance(G_road_filtered_nx, remaining_stations_knn_nx, Config.MAX_DISTANCE)
    logger.info(f"Before removal - stations: {len(stations_nx)}")
    logger.info(f"Remaining stations: {len(stations_nx) - len(stations_knn_nx)}")
    logger.info(f"Before removal - nodes: {len(G_road_nx.nodes)}, edges: {len(G_road_nx.edges)}")
    logger.info(f"After removal - nodes: {len(G_road_filtered_nx.nodes)}, edges: {len(G_road_filtered_nx.edges)}")
    logger.info("Converting graphs to igraph format for centrality calculations...")
    G_road_filtered_ig = convert_networkx_to_igraph(G_road_filtered_nx)
    ig_indices = [
        v.index for v in G_road_filtered_ig.vs if v["name"] in stations_knn_nx
    ]
    logger.info(f"Deleting {len(ig_indices)} vertices corresponding to removed stations...")
    G_road_filtered_ig.delete_vertices(ig_indices)

    random.seed(Config.RANDOM_SEED)
    logger.info(f"Selecting {Config.N_REMOVE} random stations to remove...")
    random_stations_nx = random.sample(stations_nx, Config.N_REMOVE)
    assert random_stations_nx != stations_knn_nx

    G_road_random_nx = G_road_nx.copy()
    # G_road_random_nx.remove_nodes_from(random_stations_nx)
    remaining_stations_random_nx = set(stations_nx) - set(random_stations_nx)
    logger.info(f"Remove far edges further than {Config.MAX_DISTANCE} from graph...")
    G_road_random_nx = prune_graph_by_distance(G_road_random_nx, remaining_stations_random_nx, Config.MAX_DISTANCE)
    logger.info(f"Remaining stations: {len(stations_nx) - len(random_stations_nx)}")
    logger.info(f"Remaining nodes in graph: {len(G_road_random_nx.nodes)}")
    logger.info(f"Remaining edges in graph: {len(G_road_random_nx.edges)}")
    logger.info("Converting graphs to igraph format for centrality calculations...")
    G_road_random_ig = convert_networkx_to_igraph(G_road_random_nx)
    ig_indices = [
        v.index
        for v in G_road_random_ig.vs
        if v["name"] in random_stations_nx
    ]
    logger.info(f"Deleting {len(ig_indices)} vertices corresponding to removed stations...")
    G_road_random_ig.delete_vertices(ig_indices)

    assert G_road_random_ig is not None
    assert G_road_filtered_ig is not None

    logger.info(f"Exporting igraph edges to GeoPackage...")
    igraph_edges_to_gpkg(G_road_random_ig, "random")
    igraph_edges_to_gpkg(G_road_filtered_ig, "knn")
    G_road_ig = convert_networkx_to_igraph(G_road_nx)
    igraph_edges_to_gpkg(G_road_ig, "base")

    logger.info(f"Exporting igraph nodes to GeoPackage...")
    nx_nodes_to_gpkg(G_road_nx, remaining_stations_knn_nx, "knn")
    nx_nodes_to_gpkg(G_road_nx, remaining_stations_random_nx, "random")
    nx_nodes_to_gpkg(G_road_nx, stations_nx, "all_stations")

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
