from config import Config
import logging
import osmnx as ox

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"Using local PBF file: {Config.LOCAL_PBF_PATH}")
logger.info("Loading road network from PBF...")
G_road = ox.graph_from_xml(Config.LOCAL_PBF_PATH, simplify=Config.SIMPLIFY_ROAD_NETWORK)
logger.info("Projecting road network to CRS %s...", f"EPSG:{Config.EPSG_CODE}")
G_road = ox.project_graph(G_road, to_crs=f"EPSG:{Config.EPSG_CODE}")
road_path = Config.get_road_filepath()
logger.info(f"Saving road network to {road_path}...")
ox.save_graphml(G_road, filepath=road_path)
logger.info("Road network saved successfully.")
