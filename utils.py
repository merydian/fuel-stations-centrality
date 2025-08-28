import logging
import osmnx as ox
from config import Config


logger = logging.getLogger(__name__)


def get_gas_stations_from_graph(G):
    """
    Get gas stations within the area of a NetworkX graph from OSMnx.
    Uses unified projection logic from Config and clusters nearby stations.

    Args:
        G: NetworkX graph from osmnx (should be in projected CRS)
        area_polygon: optional Shapely polygon to restrict the search

    Returns:
        GeoDataFrame of gas stations with Point geometries in target CRS (clustered)
    """
    logger.info("Extracting gas stations from OSM using unified projection logic")

    try:
        # Query gas stations from PBF file
        tags = {"amenity": "fuel"}
        logger.info("Getting gas stations from OpenStreetMap PBF file...")
        gas_stations = ox.features_from_xml(Config.LOCAL_PBF_PATH, tags=tags)
        logger.info(f"Got {len(gas_stations)} gas station features")

        # Use Config's unified projection logic
        gas_stations = Config.ensure_target_crs(gas_stations, "gas stations")

        if Config.MAX_STATIONS:
            gas_stations = gas_stations.head(Config.MAX_STATIONS)

        def get_xy(geom):
            if geom.geom_type == "Point":
                return geom.x, geom.y
            else:  # Polygon or MultiPolygon
                c = geom.centroid
                return c.x, c.y

        gas_stations["nearest_node"] = gas_stations.geometry.apply(
            lambda geom: ox.distance.nearest_nodes(G, *get_xy(geom))
        )

        return gas_stations["nearest_node"].tolist()

    except Exception as e:
        logger.error(f"Failed to extract gas stations: {e}")
        raise


def convert_networkx_to_igraph(G_nx):
    """
    Convert NetworkX graph to igraph for centrality calculations.
    Preserves coordinate attributes for projected CRS.

    Args:
        G_nx: NetworkX graph (should be in projected CRS)

    Returns:
        igraph.Graph with equivalent structure and preserved coordinates
    """
    import igraph as ig

    logger.info("Converting NetworkX graph to igraph...")
    logger.debug("Input graph CRS info: nodes have x,y coordinates in projected space")

    # Create node mapping
    node_list = list(G_nx.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Create edges for igraph
    edges = []
    edge_weights = []
    edge_lengths = []

    for u, v, data in G_nx.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        # Use 'length' attribute if available, fallback to 'weight', then 1.0
        length = data.get("length", data.get("weight", 1.0))
        edge_weights.append(length)
        edge_lengths.append(length)

    # Create igraph
    G_ig = ig.Graph(n=len(node_list), edges=edges, directed=False)
    G_ig.es["weight"] = edge_weights
    G_ig.es["length"] = edge_lengths

    # Add node attributes - preserve projected coordinates
    for i, node in enumerate(node_list):
        node_data = G_nx.nodes[node]
        G_ig.vs[i]["name"] = str(node)
        G_ig.vs[i]["x"] = float(node_data.get("x", 0.0))  # Projected x coordinate
        G_ig.vs[i]["y"] = float(node_data.get("y", 0.0))  # Projected y coordinate

    logger.info(
        f"✓ Converted NetworkX to igraph: {G_ig.vcount()} nodes, {G_ig.ecount()} edges"
    )
    logger.debug("Preserved projected coordinates in igraph node attributes")
    return G_ig
