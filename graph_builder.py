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

# Check initial CRS
initial_crs = None
if hasattr(G_road.graph, 'crs'):
    initial_crs = G_road.graph['crs']
    logger.info(f"Initial graph CRS: {initial_crs}")

logger.info(f"Projecting road network to CRS {Config.get_target_crs()}...")
G_road = ox.project_graph(G_road, to_crs=Config.get_target_crs())

# Verify projection worked correctly
sample_nodes = list(G_road.nodes(data=True))[:5]
if sample_nodes:
    coords = [(data['x'], data['y']) for node, data in sample_nodes]
    x_vals = [x for x, y in coords]
    y_vals = [y for x, y in coords]
    logger.info(f"Sample projected coordinates:")
    logger.info(f"  X range: {min(x_vals):.1f} to {max(x_vals):.1f}")
    logger.info(f"  Y range: {min(y_vals):.1f} to {max(y_vals):.1f}")
    
    # Mongolia UTM coordinates should be roughly:
    # X: 200,000 to 800,000 (easting)
    # Y: 4,500,000 to 5,500,000 (northing)
    if max(x_vals) < 1000 or max(y_vals) < 1000:
        logger.error("❌ PROJECTION FAILED: Coordinates still appear to be in degrees!")
        logger.error(f"Expected UTM coordinates for Mongolia, got: X={max(x_vals):.1f}, Y={max(y_vals):.1f}")
    else:
        logger.info("✓ Projection appears successful - coordinates are in projected units")

road_path = Config.get_road_filepath()
logger.info(f"Saving road network to {road_path}...")
ox.save_graphml(G_road, filepath=road_path)
logger.info("Road network saved successfully.")
