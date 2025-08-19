import logging
import time
from datetime import datetime
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
import os
import osmnx as ox
import igraph as ig

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "fuel_stations.log", mode="w"
        ),  # Overwrite log file each run
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

logger = logging.getLogger(__name__)


def log_step_start(step_num, description):
    """Log the start of a major step with timing."""
    logger.info("=" * 60)
    logger.info(f"STEP {step_num}: {description}")
    logger.info("=" * 60)
    return time.time()


def log_step_end(start_time, step_num, description):
    """Log the completion of a major step with duration."""
    duration = time.time() - start_time
    logger.info(f"STEP {step_num} COMPLETED: {description} (Duration: {duration:.2f}s)")
    logger.info("-" * 60)


def main():
    place = "Heidelberg, Germany"
    MAX_DISTANCE = 20000  # meters
    n_remove = 5
    n = 5
    kind = "knn_dist"

    # Log analysis start
    analysis_start_time = time.time()
    logger.info("=" * 80)
    logger.info(
        f"FUEL STATION CENTRALITY ANALYSIS STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 80)
    logger.info("Analysis Configuration:")
    logger.info(f"  • Location: {place}")
    logger.info(f"  • Max edge distance: {MAX_DISTANCE:,} meters")
    logger.info(f"  • Stations to remove: {n_remove}")
    logger.info(f"  • k-NN parameter: {n}")
    logger.info(f"  • Removal criteria: {kind}")
    logger.info(
        f"  • ORS API Key: {'✓ SET' if os.getenv('ORS_API_KEY') else '✗ NOT SET'}"
    )
    logger.info("")

    try:
        # Step 0: Download road network
        step_start = log_step_start("0", "Downloading road network from OpenStreetMap")
        G_road = ox.graph_from_place(place, network_type="drive")
        logger.info(
            f"✓ Road network downloaded: {len(G_road.nodes):,} nodes, {len(G_road.edges):,} edges"
        )
        log_step_end(step_start, "0", "Road network download")

        # Step 1: Load fuel stations
        step_start = log_step_start(
            "1", "Extracting fuel stations from road network area"
        )
        stations = get_gas_stations_from_graph(G_road)

        # Limit number of stations for testing if too many
        original_count = len(stations)
        if len(stations) > 100:
            stations = stations.sample(n=100, random_state=42).reset_index(drop=True)
            logger.info(
                f"⚠ Sampled {len(stations)} stations from {original_count} for analysis performance"
            )

        logger.info(f"✓ Fuel stations loaded: {len(stations)} stations")

        if len(stations) < 10:
            logger.error(
                f"✗ Insufficient fuel stations: {len(stations)} < 10 minimum required"
            )
            return
        log_step_end(step_start, "1", "Fuel station extraction")

        # Step 2: Create initial graph
        step_start = log_step_start("2", "Building station connectivity graph")
        logger.info("  Using OpenRouteService API for precise driving distances...")

        G = make_graph_from_stations(stations, api_key=os.getenv("ORS_API_KEY"))

        logger.info(f"✓ Station graph created: {G.vcount()} nodes, {G.ecount()} edges")
        log_step_end(step_start, "2", "Station graph creation")

        # Step 2.1: Extract and store the original convex hull
        step_start = log_step_start(
            "2.1", "Computing base geometry for consistent analysis"
        )

        try:
            from shapely.geometry import Point, MultiPoint

            coords = [G.vs[i]["coord"] for i in range(G.vcount())]
            points = [Point(coord[0], coord[1]) for coord in coords]
            multipoint = MultiPoint(points)
            base_convex_hull = multipoint.convex_hull
            logger.info("✓ Base convex hull computed for consistent Voronoi clipping")
        except Exception as e:
            logger.error(f"✗ Failed to compute base convex hull: {e}")
            base_convex_hull = None
        log_step_end(step_start, "2.1", "Base geometry computation")

        # Step 3: Filter long edges
        step_start = log_step_start(
            "3", "Filtering unrealistic long-distance connections"
        )

        initial_edges = G.ecount()
        G = remove_long_edges(G, MAX_DISTANCE)
        removed_edges = initial_edges - G.ecount()

        logger.info(
            f"✓ Edge filtering complete: removed {removed_edges} edges ({100 * removed_edges / initial_edges:.1f}%)"
        )
        log_step_end(step_start, "3", "Edge filtering")

        # Step 4: Calculate farness centrality
        step_start = log_step_start("4", "Computing centrality measures")
        logger.info("  Computing farness centrality and k-NN distances...")

        G, farness, knn_dist = farness_centrality(G, weight="weight", n=n)

        logger.info(f"✓ Centrality computation complete for {len(farness)} nodes")
        log_step_end(step_start, "4", "Centrality computation")

        # Load from GraphML (temporary step)
        logger.info("Loading graph from GraphML file...")
        G = ig.Graph.Read_GraphML("heidelberg_road.graphml")
        logger.info(f"✓ Graph loaded from file: {G.vcount()} nodes, {G.ecount()} edges")

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

        save_graph_to_geopackage(
            G, farness=farness, knn_dist=knn_dist, out_file="fuel_stations.gpkg"
        )

        logger.info("✓ Baseline graph saved to GeoPackage")
        log_step_end(step_start, "6", "Baseline graph save")

        # Step 7: Filter stations and create new graph
        step_start = log_step_start("7", f"Applying {kind}-based station filtering")

        logger.info(f"  Identifying stations for removal based on {kind} values...")
        remove_ids = list(knn_dist.keys())
        initial_nodes = G.vcount()

        G_filtered = filter_graph_stations(G, remove_ids)
        G_filtered = remove_disconnected_nodes(G_filtered)

        logger.info(f"✓ Filtered graph: {initial_nodes} → {G_filtered.vcount()} nodes")

        graph_to_gdf(G_filtered)
        G_fareness_removed = remove_long_edges(G_filtered, MAX_DISTANCE)
        G_fareness_removed = remove_disconnected_nodes(G_fareness_removed)

        logger.info(
            f"✓ Optimized filtered graph: {G_fareness_removed.vcount()} nodes, {G_fareness_removed.ecount()} edges"
        )
        log_step_end(step_start, "7", "Station filtering")

        # Step 7b: Create random comparison graph
        step_start = log_step_start("7b", "Creating random comparison graph")

        logger.info("  Removing random stations for comparison...")
        G_random = remove_random_stations(
            G.copy(), n_remove, seed=42
        )  # Use seed for reproducibility

        logger.info("  Converting random graph to GeoDataFrame...")
        graph_to_gdf(G_random)

        logger.info("  Creating new graph from random stations...")
        G_random_newly_calculated = remove_long_edges(G, MAX_DISTANCE)
        G_random_newly_calculated = remove_disconnected_nodes(G_random_newly_calculated)

        logger.info("✓ Random comparison graph created")
        log_step_end(step_start, "7b", "Random comparison graph")

        # Step 8: Calculate farness for filtered graph
        step_start = log_step_start(
            "8", "Computing farness centrality for filtered graph"
        )

        (
            G_fareness_removed,
            farness_filtered_newly_calculated,
            knn_dist_filtered_newly_calculated,
        ) = farness_centrality(G_fareness_removed, weight="weight", n=n)

        logger.info("✓ Filtered graph farness computation completed")
        log_step_end(step_start, "8", "Filtered graph farness")

        # Step 8b: Calculate farness for random comparison graph
        step_start = log_step_start(
            "8b", "Computing farness centrality for random comparison graph"
        )

        G_random_newly_calculated, farness_random_newly_calculated, knn_dist_random = (
            farness_centrality(G_random_newly_calculated, weight="weight", n=n)
        )

        logger.info("✓ Random comparison graph farness computation completed")
        log_step_end(step_start, "8b", "Random graph farness")

        # Step 9: Save filtered graph
        step_start = log_step_start("9", "Saving filtered graph")

        save_graph_to_geopackage(
            G_fareness_removed,
            farness=farness_filtered_newly_calculated,
            knn_dist=knn_dist_filtered_newly_calculated,
            out_file="fuel_stations_filtered.gpkg",
        )

        logger.info("✓ Filtered graph saved successfully")

        # Step 9.1: Save Voronoi diagram for filtered graph (using base convex hull)
        step_start = log_step_start("9.1", "Saving Voronoi diagram for filtered graph")

        # Step 9b: Save random comparison graph
        step_start = log_step_start("9b", "Saving random comparison graph")

        save_graph_to_geopackage(
            G_random_newly_calculated,
            farness=farness_random_newly_calculated,
            knn_dist=knn_dist_random,
            out_file="fuel_stations_random.gpkg",
        )

        logger.info("✓ Random comparison graph saved successfully")

        # Step 9.2: Save Voronoi diagram for random comparison graph (using base convex hull)
        step_start = log_step_start(
            "9.2", "Saving Voronoi diagram for random comparison graph"
        )

        try:
            # Get stats first to generate Voronoi data (with base convex hull)
            random_stats = get_graph_stats(
                G_random_newly_calculated, base_convex_hull=base_convex_hull
            )
            save_voronoi_to_geopackage(
                G_random_newly_calculated, out_file="voronoi_random.gpkg"
            )
            logger.info("✓ Random comparison Voronoi diagram saved successfully")
        except Exception as e:
            logger.warning(f"⚠ Failed to save random comparison Voronoi diagram: {e}")
        log_step_end(step_start, "9.2", "Random Voronoi save")

        # Step 10: Generate comparison (all using base convex hull)
        step_start = log_step_start("10", "Generating comparison statistics")

        new_stats = get_graph_stats(
            G_fareness_removed, base_convex_hull=base_convex_hull
        )
        random_stats = get_graph_stats(
            G_random_newly_calculated, base_convex_hull=base_convex_hull
        )

        # Compare farness-based filtering vs original
        farness_comparison = compare_graph_stats(
            old_stats,
            new_stats,
            title1="Original Graph",
            title2="Farness-Filtered Graph",
        )

        # Compare random removal vs original
        random_comparison = compare_graph_stats(
            old_stats,
            random_stats,
            title1="Original Graph",
            title2="Random-Filtered Graph",
        )

        # Compare farness-based vs random removal
        method_comparison = compare_graph_stats(
            new_stats,
            random_stats,
            title1="Farness-Filtered Graph",
            title2="Random-Filtered Graph",
        )

        logger.info("✓ Comparison statistics generated")
        log_step_end(step_start, "10", "Comparison statistics")

        # Print results
        print(farness_comparison)
        print(random_comparison)
        print(method_comparison)

        # Log final results
        total_duration = time.time() - analysis_start_time
        logger.info("=" * 80)
        logger.info(
            f"ANALYSIS COMPLETED SUCCESSFULLY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(
            f"Total Duration: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)"
        )
        logger.info("=" * 80)
        logger.info("Generated Files:")
        logger.info("  • fuel_stations.gpkg - Baseline graph data")
        logger.info("  • fuel_stations_filtered.gpkg - Optimized graph data")
        logger.info("  • fuel_stations_random.gpkg - Random comparison data")
        logger.info("  • voronoi_*.gpkg - Service area diagrams")
        logger.info("  • fuel_stations.log - Detailed analysis log")
        logger.info("")
        logger.info(
            "Note: All Voronoi diagrams use consistent clipping for fair comparison"
        )

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
