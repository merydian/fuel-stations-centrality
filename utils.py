from shapely import union_all
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
import numpy as np

logger = logging.getLogger(__name__)


def save_graph_to_geopackage(
    G, farness=None, knn_dist=None, out_file="graph.gpkg", suffix=None
):
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
        {"node_id": node_ids}, geometry=node_geoms, crs=f"EPSG:{Config.EPSG_CODE}"
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
        crs=f"EPSG:{Config.EPSG_CODE}",
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

    gdf_nodes = gpd.GeoDataFrame(nodes, crs=f"EPSG:{Config.EPSG_CODE}")
    logger.debug(
        f"Created GeoDataFrame with {len(gdf_nodes)} features, "
        f"farness data {'included' if farness_available else 'not available'}, "
        f"knn_dist data {'included' if knn_dist_available else 'not available'}"
    )

    return gdf_nodes


def save_voronoi_to_geopackage(G, out_file="voronoi.gpkg", suffix=None):
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

    if suffix:
        output_path = output_path.replace(".gpkg", f"_{suffix}.gpkg")

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

        gdf_voronoi = gpd.GeoDataFrame(voronoi_data, crs=f"EPSG:{Config.EPSG_CODE}")
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
                crs=f"EPSG:{Config.EPSG_CODE}",
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
                crs=f"EPSG:{Config.EPSG_CODE}",
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

        gdf_stations = gpd.GeoDataFrame(station_points, crs=f"EPSG:{Config.EPSG_CODE}")
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


def get_gas_stations_from_graph(G, area_polygon=None):
    """
    Get gas stations within the area of a NetworkX graph from OSMnx.

    Args:
        G: NetworkX graph from osmnx
        area_polygon: optional Shapely polygon to restrict the search

    Returns:
        GeoDataFrame of gas stations with Point geometries
    """
    logger.info("Extracting gas stations from OSM using graph boundaries")

    try:
        # Convert graph nodes to GeoDataFrame
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        nodes_gdf.fillna(0, inplace=True)
        logger.debug(f"Graph has {len(nodes_gdf)} nodes")

        # Compute area polygon if not provided
        if area_polygon is None:
            area_polygon = union_all(nodes_gdf.geometry).convex_hull
            logger.debug("Computed convex hull of graph nodes")

        # Query gas stations within the polygon
        tags = {"amenity": "fuel"}
        logger.info("Gettimg gas stations from OpenStreetMap...")
        gas_stations = ox.features_from_xml(Config.LOCAL_PBF_PATH, polygon=area_polygon, tags=tags)
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
            gas_stations_gdf = gpd.GeoDataFrame(
                gas_points, crs=f"EPSG:{Config.EPSG_CODE}"
            ).reset_index(drop=True)
            logger.info(f"Successfully processed {len(gas_stations_gdf)} gas stations")
        else:
            # Create empty GeoDataFrame with expected structure
            gas_stations_gdf = gpd.GeoDataFrame(
                columns=["geometry"], crs=f"EPSG:{Config.EPSG_CODE}"
            )
            logger.warning("No valid gas stations found")

        return gas_stations_gdf

    except Exception as e:
        logger.error(f"Failed to extract gas stations: {e}")
        raise


def remove_edges_far_from_stations(
    G, stations_gdf, max_distance, station_to_node_mapping=None
):
    """
    Remove edges that are farther than max_distance from any gas station.
    Uses precise UTM projection for accurate distance calculations.

    Args:
        G: igraph Graph object (road network)
        stations_gdf: GeoDataFrame with gas station data
        max_distance: Maximum distance in meters from any station
        station_to_node_mapping: Optional mapping from station indices to graph node indices

    Returns:
        Modified graph with distant edges removed
    """
    import numpy as np
    from scipy.spatial import cKDTree
    import pyproj

    logger.info(
        f"Removing edges farther than {max_distance:,} meters from any gas station"
    )

    if G.vcount() == 0:
        logger.warning("Empty graph provided")
        return G

    if stations_gdf.empty:
        logger.warning("No stations provided - keeping all edges")
        return G

    # Determine appropriate UTM zone from the center of all coordinates
    all_coords = []

    # Add station coordinates
    for _, station in stations_gdf.iterrows():
        all_coords.append([station.geometry.x, station.geometry.y])

    # Add graph node coordinates (sample)
    sample_size = min(1000, G.vcount())  # Sample for efficiency
    sample_indices = np.random.choice(G.vcount(), sample_size, replace=False)
    for i in sample_indices:
        all_coords.append([G.vs[i]["x"], G.vs[i]["y"]])

    all_coords = np.array(all_coords)
    center_lon = np.mean(all_coords[:, 0])
    center_lat = np.mean(all_coords[:, 1])

    # Calculate UTM zone
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = "N" if center_lat >= 0 else "S"
    epsg_code = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone
    utm_crs = f"EPSG:{epsg_code}"

    logger.info(f"Using UTM projection {utm_crs} for precise distance calculations")

    # Create transformer for coordinate conversion
    transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    # Transform station coordinates to UTM
    station_coords_utm = []
    for _, station in stations_gdf.iterrows():
        lon, lat = station.geometry.x, station.geometry.y
        x_utm, y_utm = transformer.transform(lon, lat)
        station_coords_utm.append([x_utm, y_utm])

    if not station_coords_utm:
        logger.warning("No valid station coordinates found - keeping all edges")
        return G

    station_coords_utm = np.array(station_coords_utm)
    logger.debug(
        f"Transformed {len(station_coords_utm)} station locations to UTM coordinates"
    )

    # Build KDTree for efficient nearest neighbor search in UTM coordinates
    station_tree = cKDTree(station_coords_utm)

    # Process edges and check distances
    edges_to_remove = []
    total_edges = G.ecount()

    logger.debug("Computing distances from edge midpoints to nearest stations...")

    for i, edge in enumerate(G.es):
        u, v = edge.source, edge.target

        # Get geographic coordinates of edge endpoints
        lon1, lat1 = G.vs[u]["x"], G.vs[u]["y"]
        lon2, lat2 = G.vs[v]["x"], G.vs[v]["y"]

        # Calculate edge midpoint in geographic coordinates
        midpoint_lon = (lon1 + lon2) / 2
        midpoint_lat = (lat1 + lat2) / 2

        # Transform midpoint to UTM coordinates
        midpoint_x_utm, midpoint_y_utm = transformer.transform(
            midpoint_lon, midpoint_lat
        )
        edge_midpoint_utm = np.array([midpoint_x_utm, midpoint_y_utm])

        # Find distance to nearest station in UTM coordinates (meters)
        distance_to_nearest_station, _ = station_tree.query(edge_midpoint_utm)

        if distance_to_nearest_station > max_distance:
            edges_to_remove.append(i)

        if (i + 1) % max(1, total_edges // 10) == 0:
            logger.debug(
                f"Processed {i + 1}/{total_edges} edges ({100 * (i + 1) / total_edges:.1f}%)"
            )

    logger.info(
        f"Found {len(edges_to_remove)} edges farther than {max_distance:,} meters from stations "
        f"({100 * len(edges_to_remove) / total_edges:.1f}% of all edges)"
    )

    if edges_to_remove:
        # Remove the distant edges (in reverse order to maintain indices)
        logger.debug("Removing edges far from stations...")
        G.delete_edges(sorted(edges_to_remove, reverse=True))
        logger.info(
            f"Removed {len(edges_to_remove)} distant edges. "
            f"Graph now has {G.ecount()} edges"
        )
    else:
        logger.info("No edges to remove")

    return G


def remove_long_edges(G, max_distance, weight_attr="length"):
    """
    DEPRECATED: Use remove_edges_far_from_stations instead.
    Remove edges longer than max_distance based on edge weight/length.
    """
    logger.warning(
        "remove_long_edges is deprecated, use remove_edges_far_from_stations instead"
    )

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


def save_removed_stations_to_geopackage(
    stations_gdf,
    removed_indices,
    out_file="removed_stations.gpkg",
    removal_type="unknown",
    knn_dist=None,
    suffix=None,
):
    """
    Save removed stations to GeoPackage.

    Args:
        stations_gdf: Original GeoDataFrame with all stations
        removed_indices: List of station indices that were removed
        out_file: Output filename for the GeoPackage
        removal_type: Type of removal (e.g., "smart", "random")
        knn_dist: Dictionary mapping station indices to k-NN distances (optional)
    """
    logger.info(
        f"Saving {len(removed_indices)} removed stations ({removal_type}) to GeoPackage: {out_file}"
    )

    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Define output path
    output_path = f"{output_dir}/{out_file}"

    if suffix:
        output_path = output_path.replace(".gpkg", f"_{suffix}.gpkg")

    try:
        # Filter stations to only include removed ones
        removed_stations = stations_gdf[stations_gdf.index.isin(removed_indices)].copy()

        if removed_stations.empty:
            logger.warning("No removed stations found to save")
            return

        # Add metadata about removal
        removed_stations["removal_type"] = removal_type
        removed_stations["removal_order"] = range(1, len(removed_stations) + 1)
        removed_stations["station_index"] = removed_stations.index

        # Add k-NN distance data if provided
        if knn_dist is not None:
            removed_stations["knn_dist_m"] = removed_stations.index.map(knn_dist)
            logger.debug(
                f"Added k-NN distance data for {len([idx for idx in removed_indices if idx in knn_dist])} stations"
            )
        else:
            removed_stations["knn_dist_m"] = None
            logger.debug("No k-NN distance data provided")

        # Reset index for cleaner output
        removed_stations = removed_stations.reset_index(drop=True)

        logger.info(
            f"Writing {len(removed_stations)} removed stations to {output_path}..."
        )
        removed_stations.to_file(output_path, layer="removed_stations", driver="GPKG")

        logger.info(f"Successfully saved removed stations to {output_path}")
        logger.debug(f"  Removal type: {removal_type}")
        logger.debug(f"  Number of stations: {len(removed_stations)}")
        logger.debug(f"  k-NN data included: {'Yes' if knn_dist is not None else 'No'}")

    except Exception as e:
        logger.error(f"Failed to save removed stations to {output_path}: {e}")
        raise


def save_stations_to_geopackage(
    stations_gdf, out_file="all_gas_stations.gpkg", suffix=None
):
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

    if suffix:
        output_path = output_path.replace(".gpkg", f"_{suffix}.gpkg")

    try:
        # Create a copy to avoid modifying original data
        stations_to_save = stations_gdf.copy()

        # Add metadata
        stations_to_save["station_index"] = stations_to_save.index
        stations_to_save["extraction_source"] = "OpenStreetMap"

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
            road_nodes.append([data["x"], data["y"]])
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


def remove_stations_from_road_network(
    G_road, station_to_node_mapping, stations_to_remove
):
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
        logger.info(
            f"  Road network: {G_road.number_of_nodes()} → {G_filtered.number_of_nodes()} nodes"
        )
        logger.info(
            f"  Road network: {G_road.number_of_edges()} → {G_filtered.number_of_edges()} edges"
        )

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

        # Create node mapping
        node_list = list(G_nx.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Create edges for igraph
        edges = []
        edge_weights = []

        for u, v, data in G_nx.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            # Use 'length' or 'weight' attribute, defaulting to 1.0
            weight = data.get("length", data.get("weight", 1.0))
            edge_weights.append(weight)

        # Create igraph
        G_ig = ig.Graph(n=len(node_list), edges=edges, directed=False)
        G_ig.es["weight"] = edge_weights

        # Add node attributes
        for i, node in enumerate(node_list):
            node_data = G_nx.nodes[node]
            G_ig.vs[i]["name"] = str(node)
            G_ig.vs[i]["x"] = node_data.get("x", 0.0)
            G_ig.vs[i]["y"] = node_data.get("y", 0.0)

        logger.info(
            f"✓ Converted NetworkX to igraph: {G_ig.vcount()} nodes, {G_ig.ecount()} edges"
        )
        return G_ig

    except Exception as e:
        logger.error(f"Error converting NetworkX to igraph: {e}")
        return None

def make_graph_from_stations(
    stations: gpd.GeoDataFrame,
    api_key: str = None,
    profile: str = "driving-car",
    use_ors: bool = True,
    G_road=None,
    station_to_node_mapping: dict = None,
) -> ig.Graph:
    """
    Create a graph from stations using either OpenRouteService or road network distances.

    Parameters
    ----------
    stations : gpd.GeoDataFrame
        GeoDataFrame containing fuel stations
    api_key : str, optional
        OpenRouteService API key (required if use_ors=True)
    profile : str, optional
        OpenRouteService routing profile (default: "driving-car")
    use_ors : bool, optional
        Whether to use OpenRouteService (True) or road network (False)
    G_road : NetworkX Graph, optional
        Road network graph (required if use_ors=False)
    station_to_node_mapping : dict, optional
        Mapping from station indices to road network nodes (required if use_ors=False)

    Returns
    -------
    ig.Graph
        igraph Graph with stations connected by calculated distances
    """
    if G_road is None or station_to_node_mapping is None:
        raise ValueError(
            "G_road and station_to_node_mapping are required when use_ors=False"
        )
    logger.info("Using road network for station distance calculation")
    return make_graph_from_stations_via_road_network(
        stations, G_road, station_to_node_mapping
    )


def make_graph_from_stations_via_road_network(
    stations: gpd.GeoDataFrame,
    G_road,
    station_to_node_mapping: dict,
) -> ig.Graph:
    """
    Create a graph from stations using distances calculated via the road network.

    Parameters
    ----------
    stations : gpd.GeoDataFrame
        GeoDataFrame containing fuel stations
    G_road : NetworkX Graph
        Road network graph
    station_to_node_mapping : dict
        Mapping from station indices to road network node IDs

    Returns
    -------
    ig.Graph
        igraph Graph with stations connected by road network distances
    """
    logger.info(
        f"Creating graph from {len(stations)} fuel stations using road network distances"
    )

    if stations.empty:
        logger.error("Stations GeoDataFrame is empty")
        raise ValueError("Stations GeoDataFrame cannot be empty")

    if len(stations) < 2:
        logger.error(f"Insufficient stations: {len(stations)}")
        raise ValueError("At least 2 stations are required")

    # Extract coordinates and valid station indices
    locations = []
    station_indices = []
    road_nodes = []

    for idx, station in stations.iterrows():
        if idx in station_to_node_mapping:
            geom = station.geometry
            if geom is not None:
                # Get centroid for non-point geometries
                if hasattr(geom, "centroid"):
                    point = geom.centroid
                else:
                    point = geom

                locations.append((point.x, point.y))  # (longitude, latitude)
                station_indices.append(idx)
                road_nodes.append(station_to_node_mapping[idx])

    n = len(locations)
    logger.info(f"Found {n} stations with valid road network mappings")

    if n < 2:
        raise ValueError("At least 2 valid station mappings are required")

    # Calculate shortest path distances between all station pairs using road network
    logger.info("Computing shortest path distances via road network...")
    distances = np.full((n, n), np.inf)

    try:
        import networkx as nx

        # Compute all-pairs shortest paths for the station nodes
        for i, source_node in enumerate(road_nodes):
            if source_node in G_road:
                # Compute shortest paths from this source to all other stations
                try:
                    path_lengths = nx.single_source_dijkstra_path_length(
                        G_road, source_node, weight="length"
                    )

                    for j, target_node in enumerate(road_nodes):
                        if target_node in path_lengths:
                            distances[i, j] = path_lengths[target_node]

                except nx.NetworkXNoPath:
                    # Some nodes might not be reachable
                    pass

            if (i + 1) % max(1, n // 10) == 0:
                logger.debug(f"Computed distances from {i + 1}/{n} stations")

    except Exception as e:
        logger.error(f"Failed to compute road network distances: {e}")
        raise

    logger.info("Road network distance calculation completed")

    # Create igraph Graph
    logger.info("Creating igraph directed graph...")
    G = ig.Graph(directed=True)

    # Add vertices
    G.add_vertices(n)
    logger.debug(f"Added {n} vertices to graph")

    # Set coordinates as vertex attributes
    for idx, coord in enumerate(locations):
        G.vs[idx]["x"] = coord[0]  # longitude
        G.vs[idx]["y"] = coord[1]  # latitude
        G.vs[idx]["station_id"] = station_indices[idx]  # Original station index
        G.vs[idx]["road_node"] = road_nodes[idx]  # Corresponding road network node

    # Add edges with distances
    logger.info("Adding edges with distance weights...")
    edges = []
    weights = []
    lengths = []
    finite_edges = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((i, j))
                weight = distances[i, j]
                weights.append(weight)
                lengths.append(weight)
                if weight != np.inf:
                    finite_edges += 1

    G.add_edges(edges)
    G.es["weight"] = weights
    G.es["length"] = lengths

    logger.info(
        f"Graph creation completed: {G.vcount()} vertices, {G.ecount()} edges "
        f"({finite_edges} with finite weights, {G.ecount() - finite_edges} infinite)"
    )

    return G


def process_fuel_stations(stations, max_stations=None):
    """Process and validate fuel stations data."""
    logger = logging.getLogger(__name__)

    len(stations)

    # Limit number of stations if needed
    if max_stations and len(stations) > max_stations:
        logger.warning(
            f"Found {len(stations)} stations, limiting to {max_stations} for performance"
        )
        stations = stations.sample(
            n=max_stations, random_state=Config.RANDOM_SEED
        ).reset_index(drop=True)

    logger.info(f"✓ Fuel stations processed: {len(stations)} stations")

    if len(stations) < Config.MIN_STATIONS_REQUIRED:
        raise ValueError(
            f"Insufficient fuel stations: {len(stations)} < {Config.MIN_STATIONS_REQUIRED} minimum required"
        )

    return stations
