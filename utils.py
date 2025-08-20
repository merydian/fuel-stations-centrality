from shapely.geometry import Point, LineString
import geopandas as gpd
import logging
import os
import random
import pyproj
from shapely.ops import transform
import osmnx as ox
from config import Config
import time
import igraph as ig

logger = logging.getLogger(__name__)


def save_graph_to_geopackage(G, farness=None, knn_dist=None, out_file="graph.gpkg"):
    logger.info(f"Saving graph to GeoPackage: {out_file}")

    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Nodes as points
    logger.debug("Converting graph nodes to GeoDataFrame...")
    node_geoms = []
    node_ids = []

    for i in range(G.vcount()):
        lon = G.vs[i]["x"]
        lat = G.vs[i]["y"]
        node_geoms.append(Point(lon, lat))
        node_ids.append(i)
    gdf_nodes = gpd.GeoDataFrame(
        {"node_id": node_ids}, geometry=node_geoms, crs="EPSG:4326"
    )
    logger.debug(f"Created nodes GeoDataFrame with {len(gdf_nodes)} features")

    # Edges as lines
    logger.debug("Converting graph edges to GeoDataFrame...")
    edge_geoms = []
    src_ids = []
    dst_ids = []
    weights = []
    finite_weight_count = 0

    for edge in G.es:
        u, v = edge.source, edge.target
        lon1, lat1 = G.vs[u]["x"], G.vs[u]["y"]
        lon2, lat2 = G.vs[v]["x"], G.vs[v]["y"]
        edge_geoms.append(LineString([(lon1, lat1), (lon2, lat2)]))
        src_ids.append(u)
        dst_ids.append(v)

        # Use length attribute instead of weight
        weight = float(edge["length"]) if "length" in edge.attributes() else 0.0
        weights.append(weight)
        if weight != float("inf"):
            finite_weight_count += 1

    gdf_edges = gpd.GeoDataFrame(
        {"source": src_ids, "target": dst_ids, "distance_m": weights},
        geometry=edge_geoms,
        crs="EPSG:4326",
    )
    logger.debug(
        f"Created edges GeoDataFrame with {len(gdf_edges)} features "
        f"({finite_weight_count} with finite weights)"
    )

    if farness:
        logger.debug("Adding farness centrality data to nodes...")
        gdf_nodes["farness_m"] = gdf_nodes["node_id"].map(farness)

    if knn_dist:
        logger.debug("Adding k-NN distance data to nodes...")
        gdf_nodes["knn_dist_m"] = gdf_nodes["node_id"].map(knn_dist)

    # --- Save to GPKG ---
    output_path = f"{output_dir}/{out_file}"
    try:
        logger.info(f"Writing graph data to {output_path}...")
        gdf_nodes.to_file(output_path, layer="nodes", driver="GPKG")
        gdf_edges.to_file(output_path, layer="edges", driver="GPKG")
        logger.info(f"Successfully saved graph to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save graph to {output_path}: {e}")
        raise


def graph_to_gdf(G):
    logger.debug(f"Converting graph with {G.vcount()} nodes to GeoDataFrame...")

    # Convert nodes to GeoDataFrame
    nodes = []
    farness_available = "farness" in G.vs.attributes()
    knn_dist_available = "knn_dist" in G.vs.attributes()

    for i in range(G.vcount()):
        lon = G.vs[i]["x"]
        lat = G.vs[i]["y"]
        farness_val = G.vs[i]["farness"] if farness_available else 0
        knn_dist_val = G.vs[i]["knn_dist"] if knn_dist_available else 0
        nodes.append(
            {
                "id": i,
                "farness": farness_val,
                "knn_dist": knn_dist_val,
                "geometry": Point(lon, lat),
            }
        )

    gdf_nodes = gpd.GeoDataFrame(nodes, crs="EPSG:4326")
    logger.debug(
        f"Created GeoDataFrame with {len(gdf_nodes)} features, "
        f"farness data {'included' if farness_available else 'not available'}, "
        f"knn_dist data {'included' if knn_dist_available else 'not available'}"
    )

    return gdf_nodes


def filter_graph_stations(G, remove_ids):
    """
    Remove specified stations from the graph by their node IDs.

    Args:
        G: igraph Graph object
        remove_ids: List of node IDs to remove from the graph

    Returns:
        Modified graph with specified stations removed
    """
    logger.info(f"Filtering graph: removing {len(remove_ids)} specified stations")

    # Remove them from the graph
    G.delete_vertices(remove_ids)

    return G


def remove_long_edges(G, max_distance, weight_attr="length"):
    logger.info(f"Removing edges longer than {max_distance:,} meters")

    # Find edges that exceed max distance
    edges_to_remove = []
    total_edges = G.ecount()

    for i, edge in enumerate(G.es):
        if weight_attr in edge.attributes():
            if float(edge[weight_attr]) > max_distance:
                edges_to_remove.append(i)
        else:
            logger.warning(f"Edge {i} missing attribute '{weight_attr}', skipping")

    logger.info(
        f"Found {len(edges_to_remove)} edges longer than {max_distance:,} meters "
        f"({100 * len(edges_to_remove) / total_edges:.1f}% of all edges)"
    )

    if edges_to_remove:
        # Remove the long edges (in reverse order to maintain indices)
        logger.debug("Removing long edges from graph...")
        G.delete_edges(sorted(edges_to_remove, reverse=True))
        logger.info(
            f"Removed {len(edges_to_remove)} long edges. "
            f"Graph now has {G.ecount()} edges"
        )
    else:
        logger.info("No edges to remove")

    return G


def remove_random_stations(G, num_remove, seed=None):
    """
    Remove n random stations from the graph.

    Args:
        G: igraph Graph object
        num_remove: Number of stations to remove
        seed: Random seed for reproducibility (optional)

    Returns:
        Modified graph with random stations removed
    """
    logger.info(f"Removing {num_remove} random stations from graph")

    if seed is not None:
        random.seed(seed)
        logger.debug(f"Using random seed: {seed}")

    initial_length = G.vcount()
    logger.debug(f"Initial graph size: {initial_length} nodes")

    if num_remove >= initial_length:
        logger.error(
            f"Cannot remove {num_remove} stations from graph with only {initial_length} nodes"
        )
        raise ValueError(
            f"num_remove ({num_remove}) must be less than total nodes ({initial_length})"
        )

    # Get all node indices
    all_nodes = list(range(initial_length))

    # Randomly select nodes to remove
    nodes_to_remove = random.sample(all_nodes, num_remove)
    nodes_to_remove.sort()  # Sort for consistent logging

    logger.info(
        f"Selected {len(nodes_to_remove)} random nodes for removal: {nodes_to_remove[:10]}{'...' if len(nodes_to_remove) > 10 else ''}"
    )

    # Remove them from the graph
    G.delete_vertices(nodes_to_remove)

    assert num_remove > 0
    assert initial_length - G.vcount() == num_remove

    logger.info(
        f"Random station removal completed: {initial_length} → {G.vcount()} nodes"
    )

    return G


def save_voronoi_to_geopackage(G, out_file="voronoi.gpkg"):
    """
    Save Voronoi diagram from graph to GeoPackage.

    Args:
        G: igraph Graph object with voronoi_polygons attribute
        out_file: Output filename for the GeoPackage
    """
    logger.info(f"Saving Voronoi diagram to GeoPackage: {out_file}")

    if "voronoi_polygons" not in G.attributes():
        logger.error("No Voronoi polygons found in graph")
        raise ValueError("Graph must have voronoi_polygons attribute")

    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Define output path
    output_path = f"{output_dir}/{out_file}"

    try:
        # Create GeoDataFrame for Voronoi polygons
        voronoi_data = []
        voronoi_polygons = G["voronoi_polygons"]

        for i in range(G.vcount()):
            lon = G.vs[i]["x"]
            lat = G.vs[i]["y"]
            polygon = voronoi_polygons[i] if i < len(voronoi_polygons) else None

            # Calculate area if polygon is valid
            area_m2 = 0.0
            if polygon is not None and not polygon.is_empty:
                # Transform to UTM for area calculation
                centroid = polygon.centroid
                utm_zone = int((centroid.x + 180) / 6) + 1
                utm_crs = (
                    f"EPSG:{32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone}"
                )

                transformer = pyproj.Transformer.from_crs(
                    "EPSG:4326", utm_crs, always_xy=True
                )
                polygon_utm = transform(transformer.transform, polygon)
                area_m2 = polygon_utm.area

            voronoi_data.append(
                {
                    "node_id": i,
                    "station_lon": lon,
                    "station_lat": lat,
                    "area_m2": area_m2,
                    "area_km2": area_m2 / 1_000_000,
                    "geometry": polygon
                    if polygon is not None
                    else Point(lon, lat).buffer(
                        0.001
                    ),  # Small buffer for invalid polygons
                }
            )

        gdf_voronoi = gpd.GeoDataFrame(voronoi_data, crs="EPSG:4326")
        logger.debug(f"Created Voronoi GeoDataFrame with {len(gdf_voronoi)} features")

        # Create GeoDataFrame for clipping convex hull (the one used for Voronoi clipping)
        convex_hull = G.get("voronoi_convex_hull")
        if convex_hull is not None:
            gdf_hull = gpd.GeoDataFrame(
                {
                    "description": ["Convex hull used for Voronoi clipping"],
                    "type": ["clipping_hull"],
                },
                geometry=[convex_hull],
                crs="EPSG:4326",
            )
        else:
            logger.warning("No convex hull found in graph")
            gdf_hull = None

        # Create GeoDataFrame for base convex hull (original stations) if different
        base_convex_hull = G.get("base_convex_hull")
        if base_convex_hull is not None and base_convex_hull != convex_hull:
            gdf_base_hull = gpd.GeoDataFrame(
                {
                    "description": ["Original stations convex hull"],
                    "type": ["base_hull"],
                },
                geometry=[base_convex_hull],
                crs="EPSG:4326",
            )
        else:
            gdf_base_hull = None

        # Save to GeoPackage
        logger.info(f"Writing Voronoi data to {output_path}...")

        gdf_voronoi.to_file(output_path, layer="voronoi_polygons", driver="GPKG")

        if gdf_hull is not None:
            gdf_hull.to_file(output_path, layer="convex_hull", driver="GPKG")

        if gdf_base_hull is not None:
            gdf_base_hull.to_file(output_path, layer="base_convex_hull", driver="GPKG")

        # Add station points for reference
        station_points = []
        for i in range(G.vcount()):
            lon = G.vs[i]["x"]
            lat = G.vs[i]["y"]
            farness_val = G.vs[i].get("farness", 0)
            knn_dist_val = G.vs[i].get("knn_dist", 0)
            station_points.append(
                {
                    "node_id": i,
                    "farness": farness_val,
                    "knn_dist": knn_dist_val,
                    "geometry": Point(lon, lat),
                }
            )

        gdf_stations = gpd.GeoDataFrame(station_points, crs="EPSG:4326")
        gdf_stations.to_file(output_path, layer="stations", driver="GPKG")

        layers_saved = ["voronoi_polygons", "stations"]
        if gdf_hull is not None:
            layers_saved.append("convex_hull")
        if gdf_base_hull is not None:
            layers_saved.append("base_convex_hull")

        logger.info(f"Successfully saved Voronoi diagram to {output_path}")
        logger.info(f"Layers saved: {', '.join(layers_saved)}")

    except Exception as e:
        logger.error(f"Failed to save Voronoi diagram to {output_path}: {e}")
        raise


def remove_disconnected_nodes(G):
    """
    Remove all disconnected nodes (nodes with no edges) from the graph.

    Args:
        G: igraph Graph object

    Returns:
        Modified graph with disconnected nodes removed
    """
    logger.info("Removing disconnected nodes from graph")

    initial_count = G.vcount()
    logger.debug(f"Initial graph size: {initial_count} nodes")

    # Find nodes with degree 0 (disconnected)
    degrees = G.degree()
    disconnected_nodes = [i for i, degree in enumerate(degrees) if degree == 0]

    if disconnected_nodes:
        logger.info(f"Found {len(disconnected_nodes)} disconnected nodes to remove")
        logger.debug(
            f"Disconnected node indices: {disconnected_nodes[:10]}{'...' if len(disconnected_nodes) > 10 else ''}"
        )

        # Remove disconnected nodes (in reverse order to maintain indices)
        G.delete_vertices(sorted(disconnected_nodes, reverse=True))

        final_count = G.vcount()
        logger.info(
            f"Disconnected nodes removal completed: {initial_count} → {final_count} nodes"
        )
    else:
        logger.info("No disconnected nodes found")

    return G


def get_gas_stations_from_graph(G):
    """
    Get gas stations within the area of a NetworkX graph from OSMnx.

    Args:
        G: NetworkX graph from osmnx

    Returns:
        GeoDataFrame of gas stations with Point geometries
    """
    logger.info("Extracting gas stations from OSM using graph boundaries")

    try:
        # Convert graph nodes to GeoDataFrame
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        logger.debug(f"Graph has {len(nodes_gdf)} nodes")

        # Get convex hull of nodes (this should already be in EPSG:4326)
        area_polygon = nodes_gdf.unary_union.convex_hull
        logger.debug("Computed convex hull of graph nodes")

        # Query gas stations within the polygon
        tags = {"amenity": "fuel"}
        logger.info("Downloading gas stations from OpenStreetMap...")

        gas_stations = ox.features_from_polygon(area_polygon, tags=tags)
        logger.info(f"Downloaded {len(gas_stations)} gas station features")

        # Filter to only include valid geometries and convert to points
        logger.debug("Processing gas station geometries...")
        gas_points = []

        for idx, station in gas_stations.iterrows():
            geom = station.geometry
            if geom is not None and not geom.is_empty:
                if geom.geom_type == "Point":
                    gas_points.append(station)
                elif hasattr(geom, "centroid"):
                    # Convert polygons/multipolygons to centroids
                    station_copy = station.copy()
                    station_copy.geometry = geom.centroid
                    gas_points.append(station_copy)

        if gas_points:
            gas_stations_gdf = gpd.GeoDataFrame(gas_points, crs="EPSG:4326")
            gas_stations_gdf = gas_stations_gdf.reset_index(drop=True)
            logger.info(f"Successfully processed {len(gas_stations_gdf)} gas stations")
        else:
            # Create empty GeoDataFrame with expected structure
            gas_stations_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
            logger.warning("No valid gas stations found")

        return gas_stations_gdf

    except Exception as e:
        logger.error(f"Failed to extract gas stations: {e}")
        raise

def create_base_convex_hull(stations):
    """
    Create a convex hull from station coordinates for consistent geometric analysis.
    
    Args:
        stations: GeoDataFrame with fuel station data
        
    Returns:
        Polygon representing the convex hull of all stations
    """
    try:
        from shapely.geometry import MultiPoint
        
        # Extract coordinates from stations
        coords = [(station.geometry.x, station.geometry.y) for _, station in stations.iterrows()]
        
        if len(coords) < 3:
            logger.warning("Not enough stations to create convex hull")
            return None
            
        # Create MultiPoint and get convex hull
        points = MultiPoint(coords)
        convex_hull = points.convex_hull
        
        logger.info(f"✓ Created convex hull from {len(coords)} station coordinates")
        return convex_hull
        
    except Exception as e:
        logger.error(f"Error creating convex hull: {e}")
        return None


# Configure logging with more detailed format
def setup_logging():
    """Set up comprehensive logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOG_FILE, mode="w"),
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


def log_step_start(step_num, description):
    """Log the start of a major step with timing."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"STEP {step_num}: {description}")
    logger.info("=" * 60)
    return time.time()


def log_step_end(start_time, step_num, description):
    """Log the completion of a major step with duration."""
    logger = logging.getLogger(__name__)
    duration = time.time() - start_time
    logger.info(f"STEP {step_num} COMPLETED: {description} (Duration: {duration:.2f}s)")
    logger.info("-" * 60)


def log_step_start(step_num, description):
    """Log the start of a processing step and return timestamp."""
    timestamp = time.time()
    logger.info("=" * 60)
    logger.info(f"STEP {step_num}: {description.upper()}")
    logger.info("=" * 60)
    return timestamp


def find_stations_in_road_network(G_road, stations):
    """
    Find the road network nodes that correspond to fuel stations.
    
    Args:
        G_road: NetworkX road network graph
        stations: GeoDataFrame with fuel station data
        
    Returns:
        Dictionary mapping station indices to road network node IDs
    """
    try:
        import numpy as np
        from scipy.spatial import cKDTree
        
        # Get road network node coordinates
        road_nodes = []
        road_node_ids = []
        for node_id, data in G_road.nodes(data=True):
            road_nodes.append([data['x'], data['y']])
            road_node_ids.append(node_id)
        
        road_nodes = np.array(road_nodes)
        
        # Build KDTree for efficient nearest neighbor search
        tree = cKDTree(road_nodes)
        
        # Find nearest road nodes for each station
        station_to_node = {}
        for idx, station in stations.iterrows():
            station_coords = np.array([station.geometry.x, station.geometry.y])
            
            # Find nearest road node
            distance, nearest_idx = tree.query(station_coords)
            nearest_node_id = road_node_ids[nearest_idx]
            
            station_to_node[idx] = nearest_node_id
            
        logger.info(f"✓ Mapped {len(station_to_node)} stations to road network nodes")
        return station_to_node
        
    except Exception as e:
        logger.error(f"Error mapping stations to road network: {e}")
        return {}


def remove_stations_from_road_network(G_road, station_to_node_mapping, stations_to_remove):
    """
    Remove station nodes from the road network.
    
    Args:
        G_road: NetworkX road network graph
        station_to_node_mapping: Dictionary mapping station indices to road network node IDs
        stations_to_remove: List of station indices to remove
        
    Returns:
        Modified NetworkX graph with station nodes removed
    """
    try:
        G_filtered = G_road.copy()
        nodes_to_remove = []
        
        # Collect nodes to remove
        for station_idx in stations_to_remove:
            if station_idx in station_to_node_mapping:
                node_id = station_to_node_mapping[station_idx]
                if node_id in G_filtered:
                    nodes_to_remove.append(node_id)
        
        # Remove nodes from graph
        G_filtered.remove_nodes_from(nodes_to_remove)
        
        logger.info(f"✓ Removed {len(nodes_to_remove)} station nodes from road network")
        logger.info(f"  Road network: {G_road.number_of_nodes()} → {G_filtered.number_of_nodes()} nodes")
        logger.info(f"  Road network: {G_road.number_of_edges()} → {G_filtered.number_of_edges()} edges")
        
        return G_filtered
        
    except Exception as e:
        logger.error(f"Error removing stations from road network: {e}")
        return G_road.copy()


def convert_networkx_to_igraph(G_nx):
    """
    Convert NetworkX graph to igraph for centrality calculations.
    
    Args:
        G_nx: NetworkX graph
        
    Returns:
        igraph.Graph with equivalent structure
    """
    try:
        import igraph as ig
        import numpy as np
        
        # Create node mapping
        node_list = list(G_nx.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Create edges for igraph
        edges = []
        edge_weights = []
        
        for u, v, data in G_nx.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            # Use 'length' or 'weight' attribute, defaulting to 1.0
            weight = data.get('length', data.get('weight', 1.0))
            edge_weights.append(weight)
        
        # Create igraph
        G_ig = ig.Graph(n=len(node_list), edges=edges, directed=False)
        G_ig.es['weight'] = edge_weights
        
        # Add node attributes
        for i, node in enumerate(node_list):
            node_data = G_nx.nodes[node]
            G_ig.vs[i]['name'] = str(node)
            G_ig.vs[i]['x'] = node_data.get('x', 0.0)
            G_ig.vs[i]['y'] = node_data.get('y', 0.0)
        
        logger.info(f"✓ Converted NetworkX to igraph: {G_ig.vcount()} nodes, {G_ig.ecount()} edges")
        return G_ig
        
    except Exception as e:
        logger.error(f"Error converting NetworkX to igraph: {e}")
        return None


def find_stations_in_road_network(G_road, stations_gdf):
    """
    Find fuel stations in the road network and return mapping information.
    
    Args:
        G_road: NetworkX road network graph
        stations_gdf: GeoDataFrame of fuel stations
    
    Returns:
        dict: Mapping from station index to road network node
    """
    logger = logging.getLogger(__name__)
    logger.info("Mapping fuel stations to road network nodes...")
    
    # Get road network nodes as GeoDataFrame
    road_nodes_gdf = ox.graph_to_gdfs(G_road, edges=False)
    
    # Auto-detect appropriate projected CRS based on the centroid
    try:
        # Get the centroid of the road network to determine appropriate UTM zone
        bounds = road_nodes_gdf.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        
        # Calculate UTM zone
        utm_zone = int((center_lon + 180) / 6) + 1
        hemisphere = 'N' if center_lat >= 0 else 'S'
        epsg_code = 32600 + utm_zone if hemisphere == 'N' else 32700 + utm_zone
        projected_crs = f'EPSG:{epsg_code}'
        
        logger.info(f"Using projected CRS: {projected_crs} for accurate distance calculations")
        
        # Project both geometries to the same projected CRS
        road_nodes_projected = road_nodes_gdf.to_crs(projected_crs)
        stations_projected = stations_gdf.to_crs(projected_crs)
        
        station_to_node_mapping = {}
        
        for idx, station in stations_projected.iterrows():
            # Find nearest road network node to each station
            station_point = station.geometry
            
            # Calculate distances to all road nodes (now in meters)
            distances = road_nodes_projected.geometry.distance(station_point)
            nearest_node_idx = distances.idxmin()
            
            station_to_node_mapping[idx] = nearest_node_idx
            
        logger.info(f"✓ Mapped {len(station_to_node_mapping)} stations to road network nodes using {projected_crs}")
        
    except Exception as e:
        logger.warning(f"CRS projection failed, falling back to geographic distances: {e}")
        # Fallback to original method if projection fails
        station_to_node_mapping = {}
        
        for idx, station in stations_gdf.iterrows():
            # Find nearest road network node to each station
            station_point = station.geometry
            
            # Calculate distances to all road nodes
            distances = road_nodes_gdf.geometry.distance(station_point)
            nearest_node_idx = distances.idxmin()
            
            station_to_node_mapping[idx] = nearest_node_idx
            
        logger.info(f"✓ Mapped {len(station_to_node_mapping)} stations to road network nodes (using geographic CRS)")
    
    return station_to_node_mapping


def remove_stations_from_road_network(G_road, station_to_node_mapping, stations_to_remove):
    """
    Remove station-related nodes from the road network.
    
    Args:
        G_road: NetworkX road network graph
        station_to_node_mapping: Mapping from station index to road node
        stations_to_remove: List of station indices to remove
    
    Returns:
        NetworkX graph with stations removed
    """
    logger = logging.getLogger(__name__)
    
    # Get road nodes to remove
    nodes_to_remove = [station_to_node_mapping[station_idx] 
                      for station_idx in stations_to_remove 
                      if station_idx in station_to_node_mapping]
    
    logger.info(f"Removing {len(nodes_to_remove)} nodes from road network (corresponding to {len(stations_to_remove)} stations)")
    
    # Create copy and remove nodes
    G_modified = G_road.copy()
    G_modified.remove_nodes_from(nodes_to_remove)
    
    logger.info(f"✓ Road network modified: {len(G_road.nodes)} → {len(G_modified.nodes)} nodes")
    
    return G_modified


def convert_networkx_to_igraph(G_nx):
    """
    Convert NetworkX graph to igraph with proper attributes.
    
    Args:
        G_nx: NetworkX graph
    
    Returns:
        igraph Graph
    """
    logger = logging.getLogger(__name__)
    logger.info("Converting NetworkX graph to igraph...")
    
    # Convert to igraph
    G_ig = ig.Graph.from_networkx(G_nx)
    
    logger.info(f"✓ Converted to igraph: {G_ig.vcount()} nodes, {G_ig.ecount()} edges")
    
    return G_ig


def process_fuel_stations(stations, max_stations=None):
    """Process and validate fuel stations data."""
    logger = logging.getLogger(__name__)
    
    original_count = len(stations)
    
    # Limit number of stations if needed
    if max_stations and len(stations) > max_stations:
        logger.warning(f"Found {len(stations)} stations, limiting to {max_stations} for performance")
        stations = stations.sample(n=max_stations, random_state=Config.RANDOM_SEED).reset_index(drop=True)

    logger.info(f"✓ Fuel stations processed: {len(stations)} stations")

    if len(stations) < Config.MIN_STATIONS_REQUIRED:
        raise ValueError(f"Insufficient fuel stations: {len(stations)} < {Config.MIN_STATIONS_REQUIRED} minimum required")
    
    return stations


def save_removed_stations_to_geopackage(stations_gdf, removed_indices, out_file="removed_stations.gpkg", removal_type="unknown", knn_dist=None):
    """
    Save removed stations to GeoPackage.
    
    Args:
        stations_gdf: Original GeoDataFrame with all stations
        removed_indices: List of station indices that were removed
        out_file: Output filename for the GeoPackage
        removal_type: Type of removal (e.g., "smart", "random")
        knn_dist: Dictionary mapping station indices to k-NN distances (optional)
    """
    logger.info(f"Saving {len(removed_indices)} removed stations ({removal_type}) to GeoPackage: {out_file}")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Define output path
    output_path = f"{output_dir}/{out_file}"
    
    try:
        # Filter stations to only include removed ones
        removed_stations = stations_gdf[stations_gdf.index.isin(removed_indices)].copy()
        
        if removed_stations.empty:
            logger.warning("No removed stations found to save")
            return
        
        # Add metadata about removal
        removed_stations['removal_type'] = removal_type
        removed_stations['removal_order'] = range(1, len(removed_stations) + 1)
        removed_stations['station_index'] = removed_stations.index
        
        # Add k-NN distance data if provided
        if knn_dist is not None:
            removed_stations['knn_dist_m'] = removed_stations.index.map(knn_dist)
            logger.debug(f"Added k-NN distance data for {len([idx for idx in removed_indices if idx in knn_dist])} stations")
        else:
            removed_stations['knn_dist_m'] = None
            logger.debug("No k-NN distance data provided")
        
        # Reset index for cleaner output
        removed_stations = removed_stations.reset_index(drop=True)
        
        logger.info(f"Writing {len(removed_stations)} removed stations to {output_path}...")
        removed_stations.to_file(output_path, layer="removed_stations", driver="GPKG")
        
        logger.info(f"Successfully saved removed stations to {output_path}")
        logger.debug(f"  Removal type: {removal_type}")
        logger.debug(f"  Number of stations: {len(removed_stations)}")
        logger.debug(f"  k-NN data included: {'Yes' if knn_dist is not None else 'No'}")
        
    except Exception as e:
        logger.error(f"Failed to save removed stations to {output_path}: {e}")
        raise

def save_stations_to_geopackage(stations_gdf, out_file="all_gas_stations.gpkg"):
    """
    Save all gas stations to GeoPackage.
    
    Args:
        stations_gdf: GeoDataFrame with gas station data
        out_file: Output filename for the GeoPackage
    """
    logger.info(f"Saving {len(stations_gdf)} gas stations to GeoPackage: {out_file}")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Define output path
    output_path = f"{output_dir}/{out_file}"
    
    try:
        # Create a copy to avoid modifying original data
        stations_to_save = stations_gdf.copy()
        
        # Add metadata
        stations_to_save['station_index'] = stations_to_save.index
        stations_to_save['extraction_source'] = 'OpenStreetMap'
        
        # Ensure geometry is valid
        stations_to_save = stations_to_save[~stations_to_save.geometry.is_empty]
        
        # Reset index for cleaner output
        stations_to_save = stations_to_save.reset_index(drop=True)
        
        logger.info(f"Writing {len(stations_to_save)} gas stations to {output_path}...")
        stations_to_save.to_file(output_path, layer="gas_stations", driver="GPKG")
        
        logger.info(f"Successfully saved gas stations to {output_path}")
        logger.debug(f"  Number of stations: {len(stations_to_save)}")
        logger.debug(f"  CRS: {stations_to_save.crs}")
        
    except Exception as e:
        logger.error(f"Failed to save gas stations to {output_path}: {e}")
        raise