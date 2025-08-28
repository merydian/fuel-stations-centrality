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
from numba import njit, prange
from sklearn.cluster import DBSCAN

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
        {"node_id": node_ids}, geometry=node_geoms, crs=Config.get_target_crs()
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
        crs=Config.get_target_crs(),
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

    gdf_nodes = gpd.GeoDataFrame(nodes, crs=Config.get_target_crs())
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

        gdf_voronoi = gpd.GeoDataFrame(voronoi_data, crs=Config.get_target_crs())
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
                crs=Config.get_target_crs(),
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
                crs=Config.get_target_crs(),
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

        gdf_stations = gpd.GeoDataFrame(station_points, crs=Config.get_target_crs())
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


def cluster_nearby_stations(stations_gdf, radius_meters=500):
    """
    Cluster stations within specified radius and combine them into single representative stations.
    
    Args:
        stations_gdf: GeoDataFrame with fuel station data (in projected CRS)
        radius_meters: Maximum distance between stations to be clustered (default: 500m)
        
    Returns:
        GeoDataFrame with clustered stations, where nearby stations are combined
    """
    logger.info(f"Clustering stations within {radius_meters}m radius...")
    
    if stations_gdf.empty:
        logger.warning("Empty stations GeoDataFrame provided")
        return stations_gdf
    
    # Ensure stations are in projected CRS for accurate distance calculation
    stations_gdf = Config.ensure_target_crs(stations_gdf, "stations for clustering")
    
    # Extract coordinates for clustering
    coords = np.array([[geom.x, geom.y] for geom in stations_gdf.geometry])
    
    # Use DBSCAN clustering with specified radius
    # eps = radius in projected units (meters)
    # min_samples = 1 means any station can form a cluster
    clustering = DBSCAN(eps=radius_meters, min_samples=1, metric='euclidean')
    cluster_labels = clustering.fit_predict(coords)
    
    logger.info(f"Found {len(set(cluster_labels))} clusters from {len(stations_gdf)} original stations")
    
    # Create clustered stations
    clustered_stations = []
    cluster_info = []
    
    for cluster_id in set(cluster_labels):
        # Get all stations in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_stations = stations_gdf[cluster_mask]
        
        if len(cluster_stations) == 1:
            # Single station - keep as is
            station = cluster_stations.iloc[0].copy()
            station['cluster_id'] = cluster_id
            station['stations_in_cluster'] = 1
            station['original_indices'] = [cluster_stations.index[0]]
            clustered_stations.append(station)
        else:
            # Multiple stations - combine them
            # Use centroid of all station locations as representative location
            cluster_geoms = cluster_stations.geometry
            centroid_x = cluster_geoms.x.mean()
            centroid_y = cluster_geoms.y.mean()
            representative_point = Point(centroid_x, centroid_y)
            
            # Create representative station with combined attributes
            representative_station = cluster_stations.iloc[0].copy()
            representative_station.geometry = representative_point
            representative_station['cluster_id'] = cluster_id
            representative_station['stations_in_cluster'] = len(cluster_stations)
            representative_station['original_indices'] = list(cluster_stations.index)
            
            # Combine names if available
            if 'name' in cluster_stations.columns:
                names = cluster_stations['name'].dropna().unique()
                representative_station['name'] = ' | '.join(names) if len(names) > 0 else None
            
            clustered_stations.append(representative_station)
            
            cluster_info.append({
                'cluster_id': cluster_id,
                'num_stations': len(cluster_stations),
                'original_indices': list(cluster_stations.index),
                'centroid': (centroid_x, centroid_y)
            })
    
    # Create new GeoDataFrame with clustered stations
    clustered_gdf = gpd.GeoDataFrame(clustered_stations, crs=stations_gdf.crs)
    clustered_gdf = clustered_gdf.reset_index(drop=True)
    
    # Log clustering statistics
    original_count = len(stations_gdf)
    clustered_count = len(clustered_gdf)
    stations_combined = original_count - clustered_count
    
    logger.info(f"Station clustering completed:")
    logger.info(f"  • Original stations: {original_count}")
    logger.info(f"  • Clustered stations: {clustered_count}")
    logger.info(f"  • Stations combined: {stations_combined}")
    logger.info(f"  • Reduction: {100 * stations_combined / original_count:.1f}%")
    
    if cluster_info:
        multi_station_clusters = [c for c in cluster_info if c['num_stations'] > 1]
        logger.info(f"  • Multi-station clusters: {len(multi_station_clusters)}")
        for cluster in multi_station_clusters[:5]:  # Log first 5 clusters
            logger.debug(f"    Cluster {cluster['cluster_id']}: {cluster['num_stations']} stations combined")
    
    return clustered_gdf


def get_gas_stations_from_graph():
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
        logger.info(f"Downloaded {len(gas_stations)} gas station features")

        # Use Config's unified projection logic
        gas_stations = Config.ensure_target_crs(gas_stations, "gas stations")

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
                gas_points, crs=Config.get_target_crs()
            ).reset_index(drop=True)
            logger.info(f"Successfully processed {len(gas_stations_gdf)} gas stations in CRS {Config.get_target_crs()}")
            
            # Cluster nearby stations within 500m radius
            gas_stations_gdf = cluster_nearby_stations(gas_stations_gdf, radius_meters=500)
            logger.info(f"Final clustered stations: {len(gas_stations_gdf)}")
        else:
            # Create empty GeoDataFrame with expected structure
            gas_stations_gdf = gpd.GeoDataFrame(
                columns=["geometry"], crs=Config.get_target_crs()
            )
            logger.warning("No valid gas stations found")

        return gas_stations_gdf

    except Exception as e:
        logger.error(f"Failed to extract gas stations: {e}")
        raise


def remove_edges_far_from_stations_graph(
    G, stations_gdf, max_distance, station_to_node_mapping=None
):
    """
    Remove edges that are farther than max_distance (network distance) 
    from any gas station, using igraph's built-in shortest path functions.
    Uses unified projection logic from Config.

    Args:
        G: igraph Graph object (road network, in target projected CRS)
        stations_gdf: GeoDataFrame with gas station geometries  
        max_distance: Maximum distance in meters along the graph
        station_to_node_mapping: Optional mapping {station_idx -> graph_node_idx}

    Returns:
        tuple: (Modified graph with distant edges removed, edges_removed_count)
    """
    import numpy as np

    logger.info(
        f"Removing edges farther than {max_distance:,} meters (network distance) from any gas station using igraph"
    )

    if G.vcount() == 0:
        logger.warning("Empty graph provided")
        return G, 0

    if stations_gdf.empty:
        logger.warning("No stations provided - keeping all edges")
        return G, 0

    # --- Step 1: Ensure consistent CRS and determine station nodes ---
    if station_to_node_mapping is not None:
        # The mapping contains OSM node IDs, but igraph uses sequential indices
        # We need to convert OSM node IDs to igraph node indices
        station_nodes = []
        
        # Create mapping from OSM node ID to igraph index
        osm_to_igraph = {}
        for i in range(G.vcount()):
            if "name" in G.vs[i].attributes() and G.vs[i]["name"] is not None:
                try:
                    osm_id = int(G.vs[i]["name"])  # OSM node ID stored in 'name' attribute
                    osm_to_igraph[osm_id] = i
                except (ValueError, TypeError):
                    # Skip nodes with invalid name attributes
                    continue
        
        logger.debug(f"Created OSM to igraph mapping for {len(osm_to_igraph)} nodes")
        
        for station_idx, osm_node_id in station_to_node_mapping.items():
            if osm_node_id in osm_to_igraph:
                igraph_node_idx = osm_to_igraph[osm_node_id]
                station_nodes.append(igraph_node_idx)
            else:
                logger.warning(f"OSM node ID {osm_node_id} for station {station_idx} not found in igraph")
    else:
        # Use Config's unified projection logic
        stations_gdf = Config.ensure_target_crs(stations_gdf, "stations for edge removal")
        
        station_nodes = []
        for _, station in stations_gdf.iterrows():
            sx, sy = station.geometry.x, station.geometry.y
            nearest_node = min(
                range(G.vcount()),
                key=lambda i: (G.vs[i]["x"] - sx) ** 2 + (G.vs[i]["y"] - sy) ** 2
            )
            station_nodes.append(nearest_node)

    station_nodes = list(set(station_nodes))  # remove duplicates

    if not station_nodes:
        logger.warning("No valid station-to-node mapping found - keeping all edges")
        return G, 0

    logger.info(f"Using {len(station_nodes)} valid stations mapped to graph nodes")

    # --- Step 2: Use igraph's distances method to compute distances from all station nodes ---
    logger.debug("Computing shortest paths from station nodes using igraph...")
    
    # Debug: Check coordinate ranges to ensure proper projection
    if G.vcount() > 0:
        x_coords = [G.vs[i]["x"] for i in range(min(10, G.vcount()))]
        y_coords = [G.vs[i]["y"] for i in range(min(10, G.vcount()))]
        logger.debug(f"Sample node coordinates - X range: {min(x_coords):.1f} to {max(x_coords):.1f}")
        logger.debug(f"Sample node coordinates - Y range: {min(y_coords):.1f} to {max(y_coords):.1f}")
        
        # Check if coordinates look like they're in degrees (WGS84) instead of meters (projected)
        if max(x_coords) < 200 and max(y_coords) < 200:
            logger.warning("⚠️  Coordinates appear to be in degrees (WGS84) rather than meters (projected CRS)")
            logger.warning("This would cause distance calculations to be in degrees instead of meters")
    
    # Get shortest path distances from all station nodes to all other nodes
    weight_attr = "length" if "length" in G.es.attributes() else None
    logger.debug(f"Using weight attribute: {weight_attr}")
    
    # Debug: Check a few edge lengths
    if G.ecount() > 0:
        sample_edges = range(min(5, G.ecount()))
        edge_lengths = []
        for i in sample_edges:
            if weight_attr and weight_attr in G.es[i].attributes():
                length = G.es[i][weight_attr]
                edge_lengths.append(length)
        
        if edge_lengths:
            logger.debug(f"Sample edge lengths: {[f'{l:.2f}' for l in edge_lengths[:5]]}")
            if max(edge_lengths) < 1:
                logger.warning("⚠️  Edge lengths are very small - possibly in degrees instead of meters")
    
    try:
        # Use distances method (current) instead of shortest_paths (deprecated)
        distances_matrix = G.distances(
            source=station_nodes, 
            target=None, 
            weights=weight_attr, 
            mode="out"
        )
        
        # For each node, find the minimum distance to any station
        min_distances = np.full(G.vcount(), np.inf)
        for i in range(len(station_nodes)):
            for j in range(G.vcount()):
                if distances_matrix[i][j] < min_distances[j]:
                    min_distances[j] = distances_matrix[i][j]
                    
        logger.debug(f"Computed distances to {len([d for d in min_distances if d != np.inf])} reachable nodes")
        
        # Log distribution of minimum distances
        finite_distances = min_distances[min_distances != np.inf]
        if len(finite_distances) > 0:
            logger.info(f"Minimum distance statistics:")
            logger.info(f"  • Nodes within reach: {len(finite_distances):,} / {len(min_distances):,}")
            logger.info(f"  • Min distance: {np.min(finite_distances):.1f}m")
            logger.info(f"  • Max distance: {np.max(finite_distances):.1f}m")
            logger.info(f"  • Mean distance: {np.mean(finite_distances):.1f}m")
            logger.info(f"  • Median distance: {np.median(finite_distances):.1f}m")
            
            # Check if distances might be in wrong units
            if np.max(finite_distances) < 10:
                logger.warning("⚠️  PROJECTION ISSUE DETECTED: Distances are unusually small!")
                logger.warning("This suggests coordinates are in degrees rather than meters.")
                logger.warning("Check CRS configuration and graph projection.")
                # Convert to what distances would be in meters if they're currently in degrees
                approx_meters_max = np.max(finite_distances) * 111000  # rough degrees to meters
                logger.warning(f"If distances are in degrees, max would be ~{approx_meters_max:.0f}m")
            
            # Count nodes at different distance thresholds
            within_max = np.sum(finite_distances <= max_distance)
            beyond_max = np.sum(finite_distances > max_distance)
            logger.info(f"  • Nodes within {max_distance:,}m: {within_max:,}")
            logger.info(f"  • Nodes beyond {max_distance:,}m: {beyond_max:,}")
            
            # Show some percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                val = np.percentile(finite_distances, p)
                logger.debug(f"  • {p}th percentile: {val:.1f}m")
                
            # Export distance analysis to GeoPackage
            logger.info("Exporting distance analysis to GeoPackage...")
            save_distance_analysis_to_geopackage(
                G, min_distances, max_distance, station_nodes, 
                out_file=f"distance_analysis_{max_distance}m_threshold.gpkg"
            )
            
        else:
            logger.warning("No finite distances found - all nodes unreachable from stations")
        
    except Exception as e:
        logger.error(f"Failed to compute shortest paths using igraph: {e}")
        return G, 0

    # --- Step 3: Remove edges with both endpoints farther than max_distance ---
    edges_to_remove = []
    for i, edge in enumerate(G.es):
        u, v = edge.source, edge.target
        # Remove edge if EITHER endpoint is farther than max_distance from any station
        if min_distances[u] > max_distance or min_distances[v] > max_distance:
            edges_to_remove.append(i)

    edges_removed_count = len(edges_to_remove)
    
    logger.info(
        f"Found {edges_removed_count} edges beyond {max_distance:,}m network distance from stations"
    )

    if edges_to_remove:
        G.delete_edges(sorted(edges_to_remove, reverse=True))
        logger.info(
            f"Removed {edges_removed_count} distant edges. "
            f"Graph now has {G.ecount()} edges"
        )
    else:
        logger.info("No edges to remove")

    return G, edges_removed_count

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
    Save all gas stations to GeoPackage, including clustering metadata.

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

        # Clean problematic column names
        problematic_columns = ['FIXME', 'fixme']  # Add other problematic names as needed
        columns_to_drop = [col for col in stations_to_save.columns if col in problematic_columns]
        
        if columns_to_drop:
            logger.info(f"Dropping problematic columns: {columns_to_drop}")
            stations_to_save = stations_to_save.drop(columns=columns_to_drop)
        
        # Also clean any columns with special characters that might cause issues
        rename_dict = {}
        for col in stations_to_save.columns:
            if col != 'geometry':  # Don't rename geometry column
                # Replace problematic characters
                clean_col = col.replace(':', '_').replace(' ', '_').replace('-', '_')
                # Remove or replace other special characters
                clean_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_col)
                # Ensure it doesn't start with a number
                if clean_col and clean_col[0].isdigit():
                    clean_col = 'col_' + clean_col
                # Limit length to avoid database field name limits
                clean_col = clean_col[:63]  # Many databases have 63 char limit
                
                if clean_col != col:
                    rename_dict[col] = clean_col
        
        if rename_dict:
            logger.info(f"Renaming columns for compatibility: {len(rename_dict)} columns")
            stations_to_save = stations_to_save.rename(columns=rename_dict)
        
        stations_to_save.to_file(output_path, layer="gas_stations", driver="GPKG")
        logger.info(f"✓ Gas stations saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save gas stations to {output_path}: {e}")
        raise


# Configure logging with more detailed format
def setup_logging():
    """Setup logging configuration with level from config."""
    from config import Config
    
    # Convert string level to logging constant
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("fuel_stations_analysis.log")
        ]
    )


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
    Uses unified projection logic from Config.

    Args:
        G_road: NetworkX road network graph (in target projected CRS)
        stations: GeoDataFrame with fuel station data

    Returns:
        Dictionary mapping station indices to road network node IDs
    """
    try:
        import numpy as np
        from scipy.spatial import cKDTree

        logger.info("Mapping fuel stations to road network nodes using unified projection...")
        
        # Use Config's unified projection logic
        stations = Config.ensure_target_crs(stations, "stations for road network mapping")

        # Get road network node coordinates (already in target projected CRS)
        road_nodes = []
        road_node_ids = []
        for node_id, data in G_road.nodes(data=True):
            road_nodes.append([data["x"], data["y"]])
            road_node_ids.append(node_id)

        road_nodes = np.array(road_nodes)
        logger.debug(f"Extracted {len(road_nodes)} road network nodes")

        # Build KDTree for efficient nearest neighbor search
        tree = cKDTree(road_nodes)

        # Find nearest road nodes for each station
        station_to_node = {}
        for idx, station in stations.iterrows():
            # Extract coordinates (now guaranteed to be in target CRS)
            station_coords = np.array([station.geometry.x, station.geometry.y])

            # Find nearest road node
            distance, nearest_idx = tree.query(station_coords)
            nearest_node_id = road_node_ids[nearest_idx]

            station_to_node[idx] = nearest_node_id
            
            if distance > 1000:  # Log warning for stations > 1km from road
                logger.warning(f"Station {idx} is {distance:.1f}m from nearest road node")

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
    Preserves coordinate attributes for projected CRS.

    Args:
        G_nx: NetworkX graph (should be in projected CRS)

    Returns:
        igraph.Graph with equivalent structure and preserved coordinates
    """
    try:
        import igraph as ig

        logger.info("Converting NetworkX graph to igraph...")
        logger.debug(f"Input graph CRS info: nodes have x,y coordinates in projected space")

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
        logger.debug(f"Preserved projected coordinates in igraph node attributes")
        return G_ig

    except Exception as e:
        logger.error(f"Error converting NetworkX to igraph: {e}")
        return None

@njit(parallel=True, cache=True)
def _compute_station_distances_numba(road_nodes_array, n_stations, path_lengths_dict_keys, path_lengths_dict_values):
    """
    Numba-optimized computation of station-to-station distances.
    
    Parameters
    ----------
    road_nodes_array : numpy.ndarray
        Array of road node IDs for each station
    n_stations : int
        Number of stations
    path_lengths_dict_keys : numpy.ndarray
        Flattened array of all path length dictionary keys
    path_lengths_dict_values : numpy.ndarray
        Flattened array of all path length dictionary values
        
    Returns
    -------
    numpy.ndarray
        2D distance matrix between all station pairs
    """
    distances = np.full((n_stations, n_stations), np.inf, dtype=np.float64)
    
    # For each source station
    for i in prange(n_stations):
        source_node = road_nodes_array[i]
        
        # Find this source's path lengths in the flattened arrays
        start_idx = i * n_stations  # Assuming each station has paths to all others
        
        for j in range(n_stations):
            if i != j:
                target_node = road_nodes_array[j]
                
                # Search for target_node in this source's path lengths
                for k in range(len(path_lengths_dict_keys)):
                    if path_lengths_dict_keys[k] == target_node:
                        distances[i, j] = path_lengths_dict_values[k]
                        break
    
    return distances

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
    Uses unified projection logic from Config and Numba optimization for performance.

    Parameters
    ----------
    stations : gpd.GeoDataFrame
        GeoDataFrame containing fuel stations
    G_road : NetworkX Graph
        Road network graph (in target projected CRS)
    station_to_node_mapping : dict
        Mapping from station indices to road network node IDs

    Returns
    -------
    ig.Graph
        igraph Graph with stations connected by road network distances
    """
    logger.info(
        f"Creating graph from {len(stations)} fuel stations using road network distances with Numba optimization"
    )

    if stations.empty:
        logger.error("Stations GeoDataFrame is empty")
        raise ValueError("Stations GeoDataFrame cannot be empty")

    if len(stations) < 2:
        logger.error(f"Insufficient stations: {len(stations)}")
        raise ValueError("At least 2 stations are required")

    # Use Config's unified projection logic
    stations = Config.ensure_target_crs(stations, "stations for graph creation")

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

                # Store projected coordinates (guaranteed to be in target CRS)
                locations.append((point.x, point.y))
                station_indices.append(idx)
                road_nodes.append(station_to_node_mapping[idx])

    n = len(locations)
    logger.info(f"Found {n} stations with valid road network mappings")

    if n < 2:
        raise ValueError("At least 2 valid station mappings are required")

    # Calculate shortest path distances between all station pairs using road network
    logger.info("Computing shortest path distances via road network with Numba acceleration...")
    
    try:
        import networkx as nx
        # For larger graphs, use optimized approach
        logger.debug("Using Numba-optimized approach for large graph")
        
        # Pre-compute all shortest paths from station nodes
        all_path_lengths = {}
        for i, source_node in enumerate(road_nodes):
            if source_node in G_road:
                try:
                    path_lengths = nx.single_source_dijkstra_path_length(
                        G_road, source_node, weight="length"
                    )
                    all_path_lengths[i] = path_lengths
                except nx.NetworkXNoPath:
                    all_path_lengths[i] = {}
                    
            if (i + 1) % max(1, n // 10) == 0:
                logger.debug(f"Pre-computed paths from {i + 1}/{n} stations")
        
        # Convert to Numba-compatible format for distance matrix computation
        distances = np.full((n, n), np.inf)
        
        # Optimized distance matrix filling using vectorized operations
        road_nodes_array = np.array(road_nodes, dtype=np.int64)
        
        # Compute distances using Numba-optimized function
        distances = _compute_station_distances_numba(
            road_nodes_array, n, 
            np.array(list(all_path_lengths.keys())), 
            np.array([v for d in all_path_lengths.values() for v in d.values()])
        )

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

    # Set coordinates as vertex attributes (projected coordinates)
    for idx, coord in enumerate(locations):
        G.vs[idx]["x"] = coord[0]  # Projected x coordinate
        G.vs[idx]["y"] = coord[1]  # Projected y coordinate
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

    # Use config parameter if no explicit max_stations provided
    if max_stations is None:
        max_stations = Config.MAX_STATIONS

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

def save_distance_analysis_to_geopackage(
    G, min_distances, max_distance, station_nodes, out_file="distance_analysis.gpkg"
):
    """
    Save distance analysis results to GeoPackage for visualization.
    
    Args:
        G: igraph Graph object (road network)
        min_distances: numpy array of minimum distances from each node to nearest station
        max_distance: threshold distance used for filtering
        station_nodes: list of station node indices
        out_file: output filename for the GeoPackage
    """
    logger.info(f"Saving distance analysis to GeoPackage: {out_file}")
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = f"{output_dir}/{out_file}"
    
    try:
        # Create nodes layer with distance information
        node_data = []
        for i in range(G.vcount()):
            node_data.append({
                'node_id': i,
                'x': G.vs[i]["x"],
                'y': G.vs[i]["y"],
                'min_distance_to_station': min_distances[i] if min_distances[i] != np.inf else None,
                'is_station': i in station_nodes,
                'within_threshold': min_distances[i] <= max_distance if min_distances[i] != np.inf else False,
                'distance_category': _categorize_distance(min_distances[i], max_distance),
                'geometry': Point(G.vs[i]["x"], G.vs[i]["y"])
            })
        
        gdf_nodes = gpd.GeoDataFrame(node_data, crs=Config.get_target_crs())
        
        # Create station nodes layer
        station_data = []
        for station_idx in station_nodes:
            station_data.append({
                'station_node_id': station_idx,
                'x': G.vs[station_idx]["x"],
                'y': G.vs[station_idx]["y"],
                'geometry': Point(G.vs[station_idx]["x"], G.vs[station_idx]["y"])
            })
        
        gdf_stations = gpd.GeoDataFrame(station_data, crs=Config.get_target_crs())
        
        # Create edges layer with removal status
        edge_data = []
        edges_to_remove = []
        for i, edge in enumerate(G.es):
            u, v = edge.source, edge.target
            will_be_removed = min_distances[u] > max_distance or min_distances[v] > max_distance
            if will_be_removed:
                edges_to_remove.append(i)
            
            # Fix: Use proper edge attribute access
            edge_length = edge["length"] if "length" in edge.attributes() else 0.0
            
            edge_data.append({
                'edge_id': i,
                'source': u,
                'target': v,
                'source_distance': min_distances[u] if min_distances[u] != np.inf else None,
                'target_distance': min_distances[v] if min_distances[v] != np.inf else None,
                'will_be_removed': will_be_removed,
                'length': edge_length,
                'geometry': LineString([
                    (G.vs[u]["x"], G.vs[u]["y"]),
                    (G.vs[v]["x"], G.vs[v]["y"])
                ])
            })
        
        gdf_edges = gpd.GeoDataFrame(edge_data, crs=Config.get_target_crs())
        
        # Create summary statistics
        finite_distances = min_distances[min_distances != np.inf]
        summary_data = [{
            'statistic': 'total_nodes',
            'value': len(min_distances),
            'description': 'Total number of nodes in road network'
        }, {
            'statistic': 'reachable_nodes',
            'value': len(finite_distances),
            'description': 'Nodes reachable from at least one station'
        }, {
            'statistic': 'station_nodes',
            'value': len(station_nodes),
            'description': 'Number of station nodes'
        }, {
            'statistic': 'nodes_within_threshold',
            'value': int(np.sum(finite_distances <= max_distance)),
            'description': f'Nodes within {max_distance}m of a station'
        }, {
            'statistic': 'nodes_beyond_threshold',
            'value': int(np.sum(finite_distances > max_distance)),
            'description': f'Nodes beyond {max_distance}m from any station'
        }, {
            'statistic': 'edges_to_remove',
            'value': len(edges_to_remove),
            'description': 'Edges that will be removed due to distance threshold'
        }, {
            'statistic': 'min_distance',
            'value': float(np.min(finite_distances)) if len(finite_distances) > 0 else None,
            'description': 'Minimum distance to nearest station (m)'
        }, {
            'statistic': 'max_distance',
            'value': float(np.max(finite_distances)) if len(finite_distances) > 0 else None,
            'description': 'Maximum distance to nearest station (m)'
        }, {
            'statistic': 'mean_distance',
            'value': float(np.mean(finite_distances)) if len(finite_distances) > 0 else None,
            'description': 'Mean distance to nearest station (m)'
        }, {
            'statistic': 'median_distance',
            'value': float(np.median(finite_distances)) if len(finite_distances) > 0 else None,
            'description': 'Median distance to nearest station (m)'
        }]
        
        # Add percentiles
        if len(finite_distances) > 0:
            for p in [10, 25, 50, 75, 90, 95, 99]:
                summary_data.append({
                    'statistic': f'p{p}_distance',
                    'value': float(np.percentile(finite_distances, p)),
                    'description': f'{p}th percentile distance to nearest station (m)'
                })
        
        # Create a simple geometry for summary (centroid of all nodes)
        if len(finite_distances) > 0:
            all_x = [G.vs[i]["x"] for i in range(G.vcount())]
            all_y = [G.vs[i]["y"] for i in range(G.vcount())]
            centroid = Point(np.mean(all_x), np.mean(all_y))
            for item in summary_data:
                item['geometry'] = centroid
        
        gdf_summary = gpd.GeoDataFrame(summary_data, crs=Config.get_target_crs())
        
        # Save all layers to GeoPackage
        logger.info(f"Writing distance analysis to {output_path}...")
        gdf_nodes.to_file(output_path, layer="nodes_with_distances", driver="GPKG")
        gdf_stations.to_file(output_path, layer="station_nodes", driver="GPKG")
        gdf_edges.to_file(output_path, layer="edges_with_removal_status", driver="GPKG")
        gdf_summary.to_file(output_path, layer="distance_statistics", driver="GPKG")
        
        logger.info(f"✓ Distance analysis saved to {output_path}")
        logger.info(f"  Layers: nodes_with_distances, station_nodes, edges_with_removal_status, distance_statistics")
        logger.info(f"  Total nodes: {len(gdf_nodes):,}")
        logger.info(f"  Station nodes: {len(gdf_stations):,}")
        logger.info(f"  Edges: {len(gdf_edges):,} (will remove {len(edges_to_remove):,})")
        
    except Exception as e:
        logger.error(f"Failed to save distance analysis to {output_path}: {e}")
        raise


def _categorize_distance(distance, threshold):
    """Categorize distance for visualization purposes."""
    if distance == np.inf:
        return "unreachable"
    elif distance <= threshold * 0.25:
        return "very_close"
    elif distance <= threshold * 0.5:
        return "close"
    elif distance <= threshold * 0.75:
        return "moderate"
    elif distance <= threshold:
        return "far_but_within_threshold"
    else:
        return "beyond_threshold"

