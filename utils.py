import logging
import osmnx as ox
from config import Config
from shapely.geometry import LineString, Point
import geopandas as gpd


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
        gas_stations = gas_stations.to_crs(G.graph["crs"])
        logger.info(f"Got {len(gas_stations)} gas station features")

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

        # Remove stations further than Config.STATIONS_MAX_RADIUS from their nearest node
        def node_distance(row):
            node = row["nearest_node"]
            geom = row.geometry
            node_x = G.nodes[node]["x"]
            node_y = G.nodes[node]["y"]
            return Point(node_x, node_y).distance(geom)

        gas_stations["node_dist"] = gas_stations.apply(node_distance, axis=1)
        gas_stations = gas_stations[gas_stations["node_dist"] <= Config.STATIONS_MAX_RADIUS]

        assert len(set(gas_stations["nearest_node"].tolist())) > 1

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

def igraph_edges_to_gpkg(g, name):
    edges = g.es
    edge_gdf = gpd.GeoDataFrame(
        {"source": [e.source for e in edges],
        "target": [e.target for e in edges]},
        geometry=[LineString([(g.vs[e.source]["x"], g.vs[e.source]["y"]),
                            (g.vs[e.target]["x"], g.vs[e.target]["y"])])
                for e in edges],
        crs=f"EPSG:{Config.EPSG_CODE}"
    )
    edge_gdf.to_file(f"{Config.OUTPUT_DIR}/{name}_{Config.PLACE.lower()}_edges.gpkg", layer=name, driver="GPKG")

def nx_nodes_to_gpkg(G, selected_nodes, name):
    """
    Export NetworkX nodes to a GeoPackage as points.

    Parameters:
    - G: networkx.Graph
    - name: name of the output layer
    - output_dir: folder to save the GPKG
    - epsg: coordinate reference system (default 4326)
    """
    geometries = []
    node_attrs = []

    for node in selected_nodes:
        attrs = G.nodes[node]
        x = attrs.get("x")
        y = attrs.get("y")
        geometries.append(Point(x, y))
        node_attrs.append(attrs)

    node_gdf = gpd.GeoDataFrame(node_attrs, geometry=geometries, crs=f"EPSG:{Config.EPSG_CODE}")
    node_gdf.to_file(f"{Config.OUTPUT_DIR}/{name}_nodes.gpkg", layer=name, driver="GPKG")
