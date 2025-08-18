import logging
from centrality import farness_centrality, get_graph_stats, format_graph_stats, compare_graph_stats

from utils import (
    graph_to_gdf,
    filter_graph_stations,
    save_graph_to_geopackage,
    remove_long_edges,
)
from downloader import get_fuel_stations
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
    MAX_DISTANCE = 500000  # meters
    n_remove = 50
    
    logger.info("="*60)
    logger.info("FUEL STATION CENTRALITY ANALYSIS STARTING")
    logger.info("="*60)
    logger.info(f"Analysis parameters:")
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
        stations = gpd.read_file(stations_file)
        
        logger.info(f"Loaded {len(stations)} fuel stations")

        # Step 2: Create initial graph
        logger.info("STEP 2: Creating station graph using OpenRouteService...")
        
        G = make_graph_from_stations(
            stations, api_key=os.getenv("ORS_API_KEY")
        )
        
        logger.info("Initial graph created successfully")

        # Step 3: Filter long edges
        logger.info("STEP 3: Filtering long-distance edges...")
        
        G = remove_long_edges(G, MAX_DISTANCE)
        
        logger.info("Edge filtering completed")

        # Step 4: Calculate farness centrality
        logger.info("STEP 4: Computing farness centrality...")
        
        G, farness = farness_centrality(G, weight="weight")
        
        logger.info("Farness centrality computation completed")

        # Step 5: Get initial statistics
        logger.info("STEP 5: Computing initial graph statistics...")
        
        old_stats = get_graph_stats(G)
        
        logger.info("Initial statistics computed")

        # Step 6: Save initial graph
        logger.info("STEP 6: Saving initial graph to GeoPackage...")
        
        save_graph_to_geopackage(G, farness=farness, out_file="fuel_stations.gpkg")
        
        logger.info("Initial graph saved successfully")

        # Step 7: Filter stations and create new graph
        logger.info("STEP 7: Filtering stations and creating optimized graph...")
        
        logger.info("Filtering stations based on farness centrality...")
        G_filtered = filter_graph_stations(G, n_remove)
        
        logger.info("Converting filtered graph to GeoDataFrame...")
        dgf_filtered = graph_to_gdf(G_filtered)
        
        logger.info("Creating new graph from filtered stations...")
        G_filtered_newly_calculated = make_graph_from_stations(
            dgf_filtered, api_key=os.getenv("ORS_API_KEY")
        )
        G_filtered_newly_calculated = remove_long_edges(G_filtered_newly_calculated, MAX_DISTANCE)
        
        logger.info("Filtered graph creation completed")

        # Step 8: Calculate farness for filtered graph
        logger.info("STEP 8: Computing farness centrality for filtered graph...")
        
        G_filtered_newly_calculated, farness_filtered_newly_calculated = farness_centrality(
            G_filtered_newly_calculated, weight="weight"
        )
        
        logger.info("Filtered graph farness computation completed")

        # Step 9: Save filtered graph
        logger.info("STEP 9: Saving filtered graph...")
        
        save_graph_to_geopackage(
            G_filtered_newly_calculated, 
            farness=farness_filtered_newly_calculated, 
            out_file="fuel_stations_filtered.gpkg"
        )
        
        logger.info("Filtered graph saved successfully")

        # Step 10: Generate comparison
        logger.info("STEP 10: Generating comparison statistics...")
        
        new_stats = get_graph_stats(G_filtered_newly_calculated)
        comparison = compare_graph_stats(old_stats, new_stats, 
                                       title1="Original Graph", 
                                       title2="Filtered Graph")
        
        logger.info("Comparison statistics generated")

        # Print results
        print(comparison)
        
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"CRITICAL ERROR in analysis pipeline: {e}", exc_info=True)
        logger.error("Analysis failed - check logs above for details")
        raise

if __name__ == "__main__":
    main()
        print(comparison)
        
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"CRITICAL ERROR in analysis pipeline: {e}", exc_info=True)
        logger.error("Analysis failed - check logs above for details")
        raise

if __name__ == "__main__":
    main()
