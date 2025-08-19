from shapely.geometry import Point, LineString
import geopandas as gpd
import logging
import os
import random
import pyproj
from shapely.ops import transform
import osmnx as ox

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

    # List all vertex attribute names
    print(G.vs.attributes())


    for i in range(G.vcount()):
        lon = G.vs[i]['x']
        lat = G.vs[i]['y']
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
    logger.debug(f"Created edges GeoDataFrame with {len(gdf_edges)} features "
                f"({finite_weight_count} with finite weights)")

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
        lon = G.vs[i]['x']
        lat = G.vs[i]['y']
        farness_val = G.vs[i]["farness"] if farness_available else 0
        knn_dist_val = G.vs[i]["knn_dist"] if knn_dist_available else 0
        nodes.append(
            {"id": i, "farness": farness_val, "knn_dist": knn_dist_val, "geometry": Point(lon, lat)}
        )

    gdf_nodes = gpd.GeoDataFrame(nodes, crs="EPSG:4326")
    logger.debug(f"Created GeoDataFrame with {len(gdf_nodes)} features, "
                f"farness data {'included' if farness_available else 'not available'}, "
                f"knn_dist data {'included' if knn_dist_available else 'not available'}")

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

    logger.info(f"Found {len(edges_to_remove)} edges longer than {max_distance:,} meters "
               f"({100*len(edges_to_remove)/total_edges:.1f}% of all edges)")

    if edges_to_remove:
        # Remove the long edges (in reverse order to maintain indices)
        logger.debug("Removing long edges from graph...")
        G.delete_edges(sorted(edges_to_remove, reverse=True))
        logger.info(f"Removed {len(edges_to_remove)} long edges. "
                   f"Graph now has {G.ecount()} edges")
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
        logger.error(f"Cannot remove {num_remove} stations from graph with only {initial_length} nodes")
        raise ValueError(f"num_remove ({num_remove}) must be less than total nodes ({initial_length})")
    
    # Get all node indices
    all_nodes = list(range(initial_length))
    
    # Randomly select nodes to remove
    nodes_to_remove = random.sample(all_nodes, num_remove)
    nodes_to_remove.sort()  # Sort for consistent logging
    
    logger.info(f"Selected {len(nodes_to_remove)} random nodes for removal: {nodes_to_remove[:10]}{'...' if len(nodes_to_remove) > 10 else ''}")
    
    # Remove them from the graph
    G.delete_vertices(nodes_to_remove)
    
    assert num_remove > 0
    assert initial_length - G.vcount() == num_remove
    
    logger.info(f"Random station removal completed: {initial_length} → {G.vcount()} nodes")
    
    return G


def save_voronoi_to_geopackage(G, out_file="voronoi.gpkg"):
    """
    Save Voronoi diagram from graph to GeoPackage.
    
    Args:
        G: igraph Graph object with voronoi_polygons attribute
        out_file: Output filename for the GeoPackage
    """
    logger.info(f"Saving Voronoi diagram to GeoPackage: {out_file}")
    
    if 'voronoi_polygons' not in G.attributes():
        logger.error("No Voronoi polygons found in graph")
        raise ValueError("Graph must have voronoi_polygons attribute")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    try:
        # Create GeoDataFrame for Voronoi polygons
        voronoi_data = []
        voronoi_polygons = G['voronoi_polygons']
        
        for i in range(G.vcount()):
            lon, lat = G.vs[i]["coord"]
            polygon = voronoi_polygons[i] if i < len(voronoi_polygons) else None
            
            # Calculate area if polygon is valid
            area_m2 = 0.0
            if polygon is not None and not polygon.is_empty:
                # Transform to UTM for area calculation
                centroid = polygon.centroid
                utm_zone = int((centroid.x + 180) / 6) + 1
                utm_crs = f"EPSG:{32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone}"
                
                transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
                polygon_utm = transform(transformer.transform, polygon)
                area_m2 = polygon_utm.area
            
            voronoi_data.append({
                'node_id': i,
                'station_lon': lon,
                'station_lat': lat,
                'area_m2': area_m2,
                'area_km2': area_m2 / 1_000_000,
                'geometry': polygon if polygon is not None else Point(lon, lat).buffer(0.001)  # Small buffer for invalid polygons
            })
        
        gdf_voronoi = gpd.GeoDataFrame(voronoi_data, crs="EPSG:4326")
        logger.debug(f"Created Voronoi GeoDataFrame with {len(gdf_voronoi)} features")
        
        # Create GeoDataFrame for clipping convex hull (the one used for Voronoi clipping)
        convex_hull = G.get('voronoi_convex_hull')
        if convex_hull is not None:
            gdf_hull = gpd.GeoDataFrame(
                {'description': ['Convex hull used for Voronoi clipping'], 'type': ['clipping_hull']},
                geometry=[convex_hull],
                crs="EPSG:4326"
            )
        else:
            logger.warning("No convex hull found in graph")
            gdf_hull = None
        
        # Create GeoDataFrame for base convex hull (original stations) if different
        base_convex_hull = G.get('base_convex_hull')
        if base_convex_hull is not None and base_convex_hull != convex_hull:
            gdf_base_hull = gpd.GeoDataFrame(
                {'description': ['Original stations convex hull'], 'type': ['base_hull']},
                geometry=[base_convex_hull],
                crs="EPSG:4326"
            )
        else:
            gdf_base_hull = None
        
        # Save to GeoPackage
        output_path = f"{output_dir}/output/{out_file}"
        logger.info(f"Writing Voronoi data to {output_path}...")
        
        gdf_voronoi.to_file(output_path, layer="voronoi_polygons", driver="GPKG")
        
        if gdf_hull is not None:
            gdf_hull.to_file(output_path, layer="convex_hull", driver="GPKG")
        
        if gdf_base_hull is not None:
            gdf_base_hull.to_file(output_path, layer="base_convex_hull", driver="GPKG")
        
        # Add station points for reference
        station_points = []
        for i in range(G.vcount()):
            lon, lat = G.vs[i]["coord"]
            farness_val = G.vs[i].get("farness", 0)
            knn_dist_val = G.vs[i].get("knn_dist", 0)
            station_points.append({
                'node_id': i,
                'farness': farness_val,
                'knn_dist': knn_dist_val,
                'geometry': Point(lon, lat)
            })
        
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
        logger.debug(f"Disconnected node indices: {disconnected_nodes[:10]}{'...' if len(disconnected_nodes) > 10 else ''}")
        
        # Remove disconnected nodes (in reverse order to maintain indices)
        G.delete_vertices(sorted(disconnected_nodes, reverse=True))
        
        final_count = G.vcount()
        logger.info(f"Disconnected nodes removal completed: {initial_count} → {final_count} nodes")
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
                if geom.geom_type == 'Point':
                    gas_points.append(station)
                elif hasattr(geom, 'centroid'):
                    # Convert polygons/multipolygons to centroids
                    station_copy = station.copy()
                    station_copy.geometry = geom.centroid
                    gas_points.append(station_copy)

        if gas_points:
            gas_stations_gdf = gpd.GeoDataFrame(gas_points, crs='EPSG:4326')
            gas_stations_gdf = gas_stations_gdf.reset_index(drop=True)
            logger.info(f"Successfully processed {len(gas_stations_gdf)} gas stations")
        else:
            # Create empty GeoDataFrame with expected structure
            gas_stations_gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
            logger.warning("No valid gas stations found")

        return gas_stations_gdf

    except Exception as e:
        logger.error(f"Failed to extract gas stations: {e}")
        raise
