from shapely.geometry import Point, LineString
import geopandas as gpd
import logging
import os
import random
import pyproj
from shapely.ops import transform

logger = logging.getLogger(__name__)


def save_graph_to_geopackage(G, farness=None, out_file="graph.gpkg"):
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
        lon, lat = G.vs[i]["coord"]
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
        lon1, lat1 = G.vs[u]["coord"]
        lon2, lat2 = G.vs[v]["coord"]
        edge_geoms.append(LineString([(lon1, lat1), (lon2, lat2)]))
        src_ids.append(u)
        dst_ids.append(v)
        weight = edge["weight"]
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

    for i in range(G.vcount()):
        lon, lat = G.vs[i]["coord"]
        farness_val = G.vs[i]["farness"] if farness_available else 0
        nodes.append(
            {"id": i, "farness": farness_val, "geometry": Point(lon, lat)}
        )

    gdf_nodes = gpd.GeoDataFrame(nodes, crs="EPSG:4326")
    logger.debug(f"Created GeoDataFrame with {len(gdf_nodes)} features, "
                f"farness data {'included' if farness_available else 'not available'}")

    return gdf_nodes


def filter_graph_stations(G, num_remove):
    logger.info(f"Filtering graph: removing {num_remove} stations with highest farness")

    initial_length = G.vcount()
    logger.debug(f"Initial graph size: {initial_length} nodes")

    # Get all nodes with the specified attribute, sorted by value (descending)
    nodes_with_attribute = []
    if "farness" in G.vs.attributes():
        for i in range(G.vcount()):
            nodes_with_attribute.append((i, G.vs[i]["farness"]))
        logger.debug(f"Found farness data for all {len(nodes_with_attribute)} nodes")
    else:
        logger.error("No farness attribute found in graph vertices")
        raise ValueError("Graph vertices must have 'farness' attribute")

    nodes_with_attribute.sort(key=lambda x: x[1], reverse=True)

    # Get top nodes to remove
    nodes_to_remove = [node for node, _ in nodes_with_attribute[:num_remove]]
    assert len(nodes_to_remove) == num_remove, "Not enough nodes to remove"

    # Log some statistics about nodes being removed
    farness_values_to_remove = [farness for _, farness in nodes_with_attribute[:num_remove]]
    logger.info(f"Removing nodes with farness range: {min(farness_values_to_remove):.2f} - {max(farness_values_to_remove):.2f}")

    # Remove them from the graph
    G.delete_vertices(nodes_to_remove)

    assert num_remove > 0
    assert initial_length - G.vcount() == num_remove

    logger.info(f"Graph filtering completed: {initial_length} → {G.vcount()} nodes")

    return G


def remove_long_edges(G, max_distance, weight_attr="weight"):
    logger.info(f"Removing edges longer than {max_distance:,} meters")

    # Find edges that exceed max distance
    edges_to_remove = []
    total_edges = G.ecount()

    for i, edge in enumerate(G.es):
        if edge[weight_attr] > max_distance:
            edges_to_remove.append(i)

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
        output_path = f"{output_dir}/{out_file}"
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
            station_points.append({
                'node_id': i,
                'farness': farness_val,
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
