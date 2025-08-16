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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fuel_stations.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

place = "Luxembourg"
logger.info(f"Starting fuel station analysis for: {place}")
MAX_DISTANCE = 300000  # meters
n_remove = 50

try:
    logger.info("Downloading fuel stations from OpenStreetMap...")
    stations = get_fuel_stations(place)
    logger.info(f"Found {len(stations)} fuel stations")

    logger.info("Calculating stations graph using OpenRouteService...")
    G = make_graph_from_stations(
        stations, api_key=os.getenv("ORS_API_KEY"), max_distance=MAX_DISTANCE
    )

    G = remove_long_edges(G, MAX_DISTANCE)

    logger.info("Calculating farness centrality...")
    G, farness = farness_centrality(G, weight="weight")

    old_stats = get_graph_stats(G)

    logger.info("Saving graph to GeoPackage...")
    save_graph_to_geopackage(G, farness=farness, out_file="fuel_stations.gpkg")

    logger.info("Modify base graph according to fuel stations farness nodes...")

    G_filtered = filter_graph_stations(G, n_remove)
    dgf_filtered = graph_to_gdf(G_filtered)
    G_filtered_newly_calculated = make_graph_from_stations(
        dgf_filtered, api_key=os.getenv("ORS_API_KEY"), max_distance=MAX_DISTANCE
    )
    G_filtered_newly_calculated = remove_long_edges(G_filtered_newly_calculated, MAX_DISTANCE)

    logger.info("Calculating farness centrality for filtered graph...")
    G_filtered_newly_calculated, farness_filtered_newly_calculated = farness_centrality(G_filtered_newly_calculated, weight="weight")

    logger.info("Saving filtered graph to GPKG...")
    save_graph_to_geopackage(
        G_filtered_newly_calculated, farness=farness_filtered_newly_calculated, out_file="fuel_stations_filtered.gpkg"
    )

    new_stats = get_graph_stats(G_filtered_newly_calculated)

    print(compare_graph_stats(old_stats, new_stats))

except Exception as e:
    logger.error(f"Error in main execution: {e}")
    raise
