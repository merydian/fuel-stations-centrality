from config import Config
import logging
import osmnx as ox

logger = logging.getLogger(__name__)

logger.info(f"Using local PBF file: {Config.LOCAL_PBF_PATH}")
G_road = ox.graph_from_xml(Config.LOCAL_PBF_PATH, simplify=Config.SIMPLIFY_ROAD_NETWORK)
G_road = ox.project_graph(G_road, to_crs=f"EPSG:{Config.EPSG_CODE}")
road_path = Config.get_road_filename(Config.PLACE)
ox.save_graphml(G_road, filepath=road_path)