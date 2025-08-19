"""
Enhanced main module with improved error handling and code structure.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import osmnx as ox
import igraph as ig
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, MultiPoint

from config import Config
from centrality import farness_centrality
from stats import get_graph_stats, compare_graph_stats
from utils import (
    graph_to_gdf,
    filter_graph_stations,
    remove_random_stations,
    save_graph_to_geopackage,
    save_voronoi_to_geopackage,
    remove_long_edges,
    remove_disconnected_nodes,
    get_gas_stations_from_graph,
)
from ors_router import make_graph_from_stations


# Configure logging with more detailed format
def setup_logging():
    """Set up comprehensive logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOG_FILE, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set specific log levels for different modules
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)
    logging.getLogger("geopandas").setLevel(logging.WARNING)
    logging.getLogger("shapely").setLevel(logging.WARNING)
    logging.getLogger("pyproj").setLevel(logging.WARNING)


def log_step_start(step_num, description):
    """Log the start of a major step with timing."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"STEP {step_num}: {description}")
    logger.info("=" * 60)
    return time.time()


def log_step_end(start_time, step_num, description):
    """Log the completion of a major step with duration."""
    logger = logging.getLogger(__name__)
    duration = time.time() - start_time
    logger.info(f"STEP {step_num} COMPLETED: {description} (Duration: {duration:.2f}s)")
    logger.info("-" * 60)


def create_base_convex_hull(G):
    """Create base convex hull from graph coordinates."""
    try:
        coords = [(G.vs[i]["x"], G.vs[i]["y"]) for i in range(G.vcount())]
        points = [Point(coord[0], coord[1]) for coord in coords]
        multipoint = MultiPoint(points)
        return multipoint.convex_hull
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to compute base convex hull: {e}")
        return None


def download_or_load_road_network(place: str):
    """Download road network or load from cache if available."""
    logger = logging.getLogger(__name__)
    
    road_filename = Config.get_road_filename(place)
    road_filepath = Config.DATA_DIR / road_filename
    
    if road_filepath.exists():
        logger.info(f"Loading cached road network from {road_filepath}")
        try:
            G_road = ox.load_graphml(road_filepath)
            logger.info(f"✓ Cached road network loaded: {len(G_road.nodes):,} nodes, {len(G_road.edges):,} edges")
            return G_road
        except Exception as e:
            logger.warning(f"Failed to load cached road network: {e}. Downloading fresh copy.")
    
    # Download fresh network
    logger.info(f"Downloading road network for: {place}")
    G_road = ox.graph_from_place(place, network_type="drive")
    
    # Save for future use
    ox.save_graphml(G_road, road_filepath)
    logger.info(f"✓ Road network downloaded and saved: {len(G_road.nodes):,} nodes, {len(G_road.edges):,} edges")
    
    return G_road


def process_fuel_stations(stations, max_stations=None):
    """Process and validate fuel stations data."""
    logger = logging.getLogger(__name__)
    
    original_count = len(stations)
    
    # Limit number of stations if needed
    if max_stations and len(stations) > max_stations:
        logger.warning(f"Found {len(stations)} stations, limiting to {max_stations} for performance")
        stations = stations.sample(n=max_stations, random_state=Config.RANDOM_SEED).reset_index(drop=True)

    logger.info(f"✓ Fuel stations processed: {len(stations)} stations")

    if len(stations) < Config.MIN_STATIONS_REQUIRED:
        raise ValueError(f"Insufficient fuel stations: {len(stations)} < {Config.MIN_STATIONS_REQUIRED} minimum required")
    
    return stations


def main():
    """Main function for fuel station centrality analysis."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration validation and setup
    Config.ensure_directories()
    Config.validate_config()
    
    # Log analysis start
    analysis_start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"FUEL STATION CENTRALITY ANALYSIS STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info("Analysis Configuration:")
    logger.info(f"  • Location: {Config.PLACE}")
    logger.info(f"  • Max edge distance: {Config.MAX_DISTANCE:,} meters")
    logger.info(f"  • Stations to remove: {Config.N_REMOVE}")
    logger.info(f"  • k-NN parameter: {Config.K_NN}")
    logger.info(f"  • Removal criteria: {Config.REMOVAL_KIND}")
    logger.info(f"  • ORS API Key: {'✓ SET' if Config.ORS_API_KEY else '✗ NOT SET'}")
    logger.info("")

    try:
        # Step 0: Download road network
        step_start = log_step_start("0", "Downloading/Loading road network from OpenStreetMap")
        G_road = download_or_load_road_network(Config.PLACE)
        log_step_end(step_start, "0", "Road network acquisition")

        # Step 1: Load fuel stations
        step_start = log_step_start("1", "Extracting fuel stations from road network area")
        stations = get_gas_stations_from_graph(G_road)
        stations = process_fuel_stations(stations, Config.MAX_STATIONS_FOR_ANALYSIS)
        log_step_end(step_start, "1", "Fuel station extraction")

        # Step 2: Create initial graph
        step_start = log_step_start("2", "Building station connectivity graph")
        logger.info("  Using OpenRouteService API for precise driving distances...")
        G = make_graph_from_stations(stations, api_key=Config.ORS_API_KEY)
        logger.info(f"✓ Station graph created: {G.vcount()} nodes, {G.ecount()} edges")
        log_step_end(step_start, "2", "Station graph creation")

        # Step 2.1: Extract and store the original convex hull
        step_start = log_step_start("2.1", "Computing base geometry for consistent analysis")
        base_convex_hull = create_base_convex_hull(G)
        if base_convex_hull:
            logger.info("✓ Base convex hull computed for consistent Voronoi clipping")
        else:
            logger.warning("⚠ Could not compute base convex hull")
        log_step_end(step_start, "2.1", "Base geometry computation")

        # Step 3: Filter long edges
        step_start = log_step_start("3", "Filtering unrealistic long-distance connections")
        initial_edges = G.ecount()
        G = remove_long_edges(G, Config.MAX_DISTANCE)
        removed_edges = initial_edges - G.ecount()
        logger.info(f"✓ Edge filtering complete: removed {removed_edges} edges ({100 * removed_edges / initial_edges:.1f}% if initial_edges > 0 else 0)")
        log_step_end(step_start, "3", "Edge filtering")

        # Step 4: Calculate farness centrality
        step_start = log_step_start("4", "Computing centrality measures")
        logger.info("  Computing farness centrality and k-NN distances...")
        G, farness, knn_dist = farness_centrality(G, weight="weight", n=Config.K_NN)
        logger.info(f"✓ Centrality computation complete for {len(farness)} nodes")
        log_step_end(step_start, "4", "Centrality computation")

        # Step 5: Get initial statistics
        step_start = log_step_start("5", "Computing baseline graph statistics")
        old_stats = get_graph_stats(G, base_convex_hull=base_convex_hull)
        logger.info("✓ Baseline statistics computed")
        log_step_end(step_start, "5", "Baseline statistics")

        # Step 5b: Save Voronoi diagram for initial graph
        step_start = log_step_start("5b", "Saving baseline Voronoi diagram")
        try:
            save_voronoi_to_geopackage(G, out_file="voronoi_initial.gpkg")
            logger.info("✓ Baseline Voronoi diagram saved")
        except Exception as e:
            logger.warning(f"⚠ Could not save baseline Voronoi diagram: {e}")
        log_step_end(step_start, "5b", "Baseline Voronoi save")

        # Step 6: Save initial graph
        step_start = log_step_start("6", "Saving baseline graph data")
        save_graph_to_geopackage(G, farness=farness, knn_dist=knn_dist, out_file="fuel_stations.gpkg")
        logger.info("✓ Baseline graph saved to GeoPackage")
        log_step_end(step_start, "6", "Baseline graph save")

        # Step 7: Filter stations and create new graph
        step_start = log_step_start("7", f"Applying {Config.REMOVAL_KIND}-based station filtering")
        logger.info(f"  Selecting top {Config.N_REMOVE} stations for removal based on {Config.REMOVAL_KIND} values...")
        
        # Get stations with highest knn_dist values for removal
        sorted_stations = sorted(knn_dist.items(), key=lambda x: x[1], reverse=True)
        remove_ids = [station_id for station_id, _ in sorted_stations[:Config.N_REMOVE]]
        
        initial_nodes = G.vcount()
        G_filtered = filter_graph_stations(G.copy(), remove_ids)
        G_filtered = remove_disconnected_nodes(G_filtered)
        
        logger.info(f"✓ Filtered graph: {initial_nodes} → {G_filtered.vcount()} nodes")
        
        # Apply additional processing
        G_filtered = remove_long_edges(G_filtered, Config.MAX_DISTANCE)
        G_filtered = remove_disconnected_nodes(G_filtered)
        
        logger.info(f"✓ Optimized filtered graph: {G_filtered.vcount()} nodes, {G_filtered.ecount()} edges")
        log_step_end(step_start, "7", "Station filtering")

        # Step 7b: Create random comparison graph
        step_start = log_step_start("7b", "Creating random comparison graph")
        logger.info("  Removing random stations for comparison...")
        G_random = remove_random_stations(G.copy(), Config.N_REMOVE, seed=Config.RANDOM_SEED)
        G_random = remove_long_edges(G_random, Config.MAX_DISTANCE)
        G_random = remove_disconnected_nodes(G_random)
        logger.info("✓ Random comparison graph created")
        log_step_end(step_start, "7b", "Random comparison graph")

        # Step 8: Calculate farness for filtered graph
        step_start = log_step_start("8", "Computing farness centrality for filtered graph")
        G_filtered, farness_filtered, knn_dist_filtered = farness_centrality(G_filtered, weight="weight", n=Config.K_NN)
        logger.info("✓ Filtered graph farness computation completed")
        log_step_end(step_start, "8", "Filtered graph farness")

        # Step 8b: Calculate farness for random comparison graph
        step_start = log_step_start("8b", "Computing farness centrality for random comparison graph")
        G_random, farness_random, knn_dist_random = farness_centrality(G_random, weight="weight", n=Config.K_NN)
        logger.info("✓ Random comparison graph farness computation completed")
        log_step_end(step_start, "8b", "Random graph farness")

        # Step 9: Save filtered graph
        step_start = log_step_start("9", "Saving filtered graph")
        save_graph_to_geopackage(G_filtered, farness=farness_filtered, knn_dist=knn_dist_filtered, out_file="fuel_stations_filtered.gpkg")
        logger.info("✓ Filtered graph saved successfully")
        log_step_end(step_start, "9", "Filtered graph save")

        # Step 9b: Save random comparison graph
        step_start = log_step_start("9b", "Saving random comparison graph")
        save_graph_to_geopackage(G_random, farness=farness_random, knn_dist=knn_dist_random, out_file="fuel_stations_random.gpkg")
        logger.info("✓ Random comparison graph saved successfully")
        log_step_end(step_start, "9b", "Random graph save")

        # Step 10: Generate comparison
        step_start = log_step_start("10", "Generating comparison statistics")
        
        new_stats = get_graph_stats(G_filtered, base_convex_hull=base_convex_hull)
        random_stats = get_graph_stats(G_random, base_convex_hull=base_convex_hull)

        # Compare farness-based filtering vs original
        farness_comparison = compare_graph_stats(old_stats, new_stats, title1="Original Graph", title2="Farness-Filtered Graph")

        # Compare random removal vs original
        random_comparison = compare_graph_stats(old_stats, random_stats, title1="Original Graph", title2="Random-Filtered Graph")

        # Compare farness-based vs random removal
        method_comparison = compare_graph_stats(new_stats, random_stats, title1="Farness-Filtered Graph", title2="Random-Filtered Graph")

        logger.info("✓ Comparison statistics generated")
        log_step_end(step_start, "10", "Comparison statistics")

        # Print results
        print(farness_comparison)
        print(random_comparison)
        print(method_comparison)

        # Log final results
        total_duration = time.time() - analysis_start_time
        logger.info("=" * 80)
        logger.info(f"ANALYSIS COMPLETED SUCCESSFULLY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total Duration: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)")
        logger.info("=" * 80)
        logger.info("Generated Files:")
        logger.info("  • fuel_stations.gpkg - Baseline graph data")
        logger.info("  • fuel_stations_filtered.gpkg - Optimized graph data")
        logger.info("  • fuel_stations_random.gpkg - Random comparison data")
        logger.info("  • voronoi_*.gpkg - Service area diagrams (if generated)")
        logger.info("  • fuel_stations.log - Detailed analysis log")
        logger.info("")
        logger.info("Note: All analyses use consistent parameters for fair comparison")

    except Exception as e:
        total_duration = time.time() - analysis_start_time
        logger.error("=" * 80)
        logger.error(f"ANALYSIS FAILED AFTER {total_duration:.2f} SECONDS")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error("Check the log file for detailed error information")
        logger.error("=" * 80)
        raise


if __name__ == "__main__":
    main()
