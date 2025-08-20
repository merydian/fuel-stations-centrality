import osmnx as ox
import logging
from config import Config

logger = logging.getLogger(__name__)


def get_fuel_stations(place):
    """
    Downloads fuel stations from OSM using osmnx for the given place.
    Returns a GeoDataFrame of fuel station locations.
    """
    logger.info(f"Downloading fuel stations from OpenStreetMap for: {place}")

    try:
        # Configure OSM settings
        ox.settings.overpass_settings = "[maxsize:20000000000]"
        logger.debug("OSM overpass settings configured")

        tags = {"amenity": "fuel"}
        logger.debug(f"Using OSM tags: {tags}")

        logger.info("Downloading street network...")
        G = ox.graph_from_place(place, network_type="drive")
        logger.info(
            f"Street network downloaded: {len(G.nodes)} nodes, {len(G.edges)} edges"
        )

        logger.info("Downloading fuel station features...")
        fuel_gdf = ox.features_from_place(place, tags)

        logger.info(f"Successfully downloaded {len(fuel_gdf)} fuel stations from OSM")
        return fuel_gdf

    except Exception as e:
        logger.error(f"Failed to download fuel stations for {place}: {e}")
        raise


if __name__ == "__main__":
    import sys

    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    place = "Iceland"
    logger.info(f"Starting fuel station download test for: {place}")

    try:
        fuel_stations = get_fuel_stations(place)
        logger.info(
            f"Test completed successfully: Found {len(fuel_stations)} fuel stations in {place}"
        )
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

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
