from shapely.geometry import Point, LineString
import geopandas as gpd
import logging
import os

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
        logger.info(f"Writing nodes layer to {output_path}...")
        gdf_nodes.to_file(output_path, layer="nodes", driver="GPKG")
        logger.info(f"Writing edges layer to {output_path}...")
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
