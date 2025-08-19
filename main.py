import logging
from centrality import farness_centrality
from stats import get_graph_stats, compare_graph_stats

from utils import (
    graph_to_gdf,
    filter_graph_stations,
    remove_random_stations,
    save_graph_to_geopackage,
    save_voronoi_to_geopackage,
    remove_long_edges,
)
from ors_router import make_graph_from_stations
import os
import geopandas as gpd

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fuel_stations.log"),
        logging.StreamHandler()
    ],
)

# Set specific log levels for different modules
logging.getLogger("urllib3").setLevel(logging.WARNING)  # Reduce HTTP request noise
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("fiona").setLevel(logging.WARNING)  # Reduce geopandas noise

logger = logging.getLogger(__name__)

def main():
    place = "Luxembourg"
    MAX_DISTANCE = 200000  # meters
    n_remove = 50
    n = 5
    
    logger.info("="*60)
    logger.info("FUEL STATION CENTRALITY ANALYSIS STARTING")
    logger.info("="*60)
    logger.info("Analysis parameters:")
    logger.info(f"  Place: {place}")
    logger.info(f"  Max distance: {MAX_DISTANCE:,} meters")
    logger.info(f"  Stations to remove: {n_remove}")
    logger.info(f"  ORS API Key: {'SET' if os.getenv('ORS_API_KEY') else 'NOT SET'}")

    try:
        # Step 1: Load fuel stations
        logger.info("STEP 1: Loading fuel stations data...")
        
        # stations = get_fuel_stations(place)
        stations_file = "stations_iran.gpkg"
        logger.info(f"Loading stations from file: {stations_file}")
        stations = gpd.read_file(stations_file)[:500]

        logger.info(f"Loaded {len(stations)} fuel stations")

        # Step 2: Create initial graph
        logger.info("STEP 2: Creating station graph using OpenRouteService...")
        
        G = make_graph_from_stations(
            stations, api_key=os.getenv("ORS_API_KEY")
        )
        
        logger.info("Initial graph created successfully")

        # Step 2.1: Extract and store the original convex hull for consistent clipping
        logger.info("STEP 2.1: Computing base convex hull for consistent Voronoi clipping...")
        
        try:
            from shapely.geometry import Point, MultiPoint
            coords = [G.vs[i]['coord'] for i in range(G.vcount())]
            points = [Point(coord[0], coord[1]) for coord in coords]
            multipoint = MultiPoint(points)
            base_convex_hull = multipoint.convex_hull
            logger.info("Base convex hull computed successfully - will be used for all Voronoi clipping")
        except Exception as e:
            logger.warning(f"Failed to compute base convex hull: {e}")
            base_convex_hull = None

        # Step 3: Filter long edges
        logger.info("STEP 3: Filtering long-distance edges...")
        
        G = remove_long_edges(G, MAX_DISTANCE)
        
        logger.info("Edge filtering completed")

        # Step 4: Calculate farness centrality
        logger.info("STEP 4: Computing farness centrality...")
        
        G, farness, knn_dist = farness_centrality(G, weight="weight", n=n)
        
        logger.info("Farness centrality computation completed")

        # Step 5: Get initial statistics (using base convex hull)
        logger.info("STEP 5: Computing initial graph statistics...")
        
        old_stats = get_graph_stats(G, base_convex_hull=base_convex_hull)
        
        logger.info("Initial statistics computed")
        
        # Step 5b: Save Voronoi diagram for initial graph
        logger.info("STEP 5b: Saving Voronoi diagram for initial graph...")
        
        try:
            save_voronoi_to_geopackage(G, out_file="voronoi_initial.gpkg")
            logger.info("Initial Voronoi diagram saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save initial Voronoi diagram: {e}")

        # Step 6: Save initial graph
        logger.info("STEP 6: Saving initial graph to GeoPackage...")
        
        save_graph_to_geopackage(G, farness=farness, knn_dist=knn_dist, out_file="fuel_stations.gpkg")
        
        logger.info("Initial graph saved successfully")

        # Step 7: Filter stations and create new graph
        logger.info("STEP 7: Filtering stations and creating optimized graph...")
        
        logger.info("Filtering stations based on farness centrality...")
        G_filtered = filter_graph_stations(G, n_remove)
        
        logger.info("Converting filtered graph to GeoDataFrame...")
        dgf_filtered = graph_to_gdf(G_filtered)
        
        logger.info("Creating new graph from filtered stations...")
        G_fareness_removed = make_graph_from_stations(
            dgf_filtered, api_key=os.getenv("ORS_API_KEY")
        )
        G_fareness_removed = remove_long_edges(G_fareness_removed, MAX_DISTANCE)
        
        logger.info("Filtered graph creation completed")

        # Step 7b: Create random comparison graph
        logger.info("STEP 7b: Creating random comparison graph...")
        
        logger.info("Removing random stations for comparison...")
        G_random = remove_random_stations(G.copy(), n_remove, seed=42)  # Use seed for reproducibility
        
        logger.info("Converting random graph to GeoDataFrame...")
        dgf_random = graph_to_gdf(G_random)
        
        logger.info("Creating new graph from random stations...")
        G_random_newly_calculated = make_graph_from_stations(
            dgf_random, api_key=os.getenv("ORS_API_KEY")
        )
        G_random_newly_calculated = remove_long_edges(G_random_newly_calculated, MAX_DISTANCE)
        
        logger.info("Random comparison graph creation completed")

        # Step 8: Calculate farness for filtered graph
        logger.info("STEP 8: Computing farness centrality for filtered graph...")
        
        G_fareness_removed, farness_filtered_newly_calculated, knn_dist_filtered_newly_calculated = farness_centrality(
            G_fareness_removed, weight="weight", n=n
        )
        
        logger.info("Filtered graph farness computation completed")

        # Step 8b: Calculate farness for random comparison graph
        logger.info("STEP 8b: Computing farness centrality for random comparison graph...")
        
        G_random_newly_calculated, farness_random_newly_calculated, knn_dist_random = farness_centrality(
            G_random_newly_calculated, weight="weight", n=n
        )
        
        logger.info("Random comparison graph farness computation completed")

        # Step 9: Save filtered graph
        logger.info("STEP 9: Saving filtered graph...")
        
        save_graph_to_geopackage(
            G_fareness_removed, 
            farness=farness_filtered_newly_calculated, 
            knn_dist=knn_dist_filtered_newly_calculated,
            out_file="fuel_stations_filtered.gpkg"
        )
        
        logger.info("Filtered graph saved successfully")
        
        # Step 9.1: Save Voronoi diagram for filtered graph (using base convex hull)
        logger.info("STEP 9.1: Saving Voronoi diagram for filtered graph...")
        
        try:
            # Get stats first to generate Voronoi data (with base convex hull)
            filtered_stats = get_graph_stats(G_fareness_removed, base_convex_hull=base_convex_hull)
            save_voronoi_to_geopackage(G_fareness_removed, out_file="voronoi_filtered.gpkg")
            logger.info("Filtered Voronoi diagram saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save filtered Voronoi diagram: {e}")

        # Step 9b: Save random comparison graph
        logger.info("STEP 9b: Saving random comparison graph...")
        
        save_graph_to_geopackage(
            G_random_newly_calculated, 
            farness=farness_random_newly_calculated, 
            knn_dist=knn_dist_random,
            out_file="fuel_stations_random.gpkg"
        )
        
        logger.info("Random comparison graph saved successfully")
        
        # Step 9.2: Save Voronoi diagram for random comparison graph (using base convex hull)
        logger.info("STEP 9.2: Saving Voronoi diagram for random comparison graph...")
        
        try:
            # Get stats first to generate Voronoi data (with base convex hull)
            random_stats = get_graph_stats(G_random_newly_calculated, base_convex_hull=base_convex_hull)
            save_voronoi_to_geopackage(G_random_newly_calculated, out_file="voronoi_random.gpkg")
            logger.info("Random comparison Voronoi diagram saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save random comparison Voronoi diagram: {e}")

        # Step 10: Generate comparison (all using base convex hull)
        logger.info("STEP 10: Generating comparison statistics...")
        
        new_stats = get_graph_stats(G_fareness_removed, base_convex_hull=base_convex_hull)
        random_stats = get_graph_stats(G_random_newly_calculated, base_convex_hull=base_convex_hull)
        
        # Compare farness-based filtering vs original
        farness_comparison = compare_graph_stats(old_stats, new_stats, 
                                               title1="Original Graph", 
                                               title2="Farness-Filtered Graph")
        
        # Compare random removal vs original
        random_comparison = compare_graph_stats(old_stats, random_stats,
                                              title1="Original Graph",
                                              title2="Random-Filtered Graph")
        
        # Compare farness-based vs random removal
        method_comparison = compare_graph_stats(new_stats, random_stats,
                                              title1="Farness-Filtered Graph",
                                              title2="Random-Filtered Graph")
        
        logger.info("Comparison statistics generated")

        # Print results
        print(farness_comparison)
        print(random_comparison)
        print(method_comparison)
        
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("Note: All Voronoi diagrams were clipped using the original stations convex hull for consistency")

    except Exception as e:
        logger.error(f"CRITICAL ERROR in analysis pipeline: {e}", exc_info=True)
        logger.error("Analysis failed - check logs above for details")
        raise

if __name__ == "__main__":
    main()
