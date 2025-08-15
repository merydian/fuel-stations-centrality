from shapely.geometry import Point, LineString
import geopandas as gpd


def save_graph_to_geopackage(G, farness=None, out_file="graph.gpkg"):
    # Nodes as points
    node_geoms = []
    node_ids = []
    for node, data in G.nodes(data=True):
        lon, lat = data["coord"]
        node_geoms.append(Point(lon, lat))
        node_ids.append(node)
    gdf_nodes = gpd.GeoDataFrame(
        {"node_id": node_ids}, geometry=node_geoms, crs="EPSG:4326"
    )

    # Edges as lines
    edge_geoms = []
    src_ids = []
    dst_ids = []
    weights = []
    for u, v, data in G.edges(data=True):
        lon1, lat1 = G.nodes[u]["coord"]
        lon2, lat2 = G.nodes[v]["coord"]
        edge_geoms.append(LineString([(lon1, lat1), (lon2, lat2)]))
        src_ids.append(u)
        dst_ids.append(v)
        weights.append(data["weight"])

    gdf_edges = gpd.GeoDataFrame(
        {"source": src_ids, "target": dst_ids, "distance_m": weights},
        geometry=edge_geoms,
        crs="EPSG:4326",
    )

    if farness:
        gdf_nodes["farness_m"] = gdf_nodes["node_id"].map(farness)

    # --- Save to GPKG ---
    gdf_nodes.to_file(f"output/{out_file}", layer="nodes", driver="GPKG")
    gdf_edges.to_file(f"output/{out_file}", layer="edges", driver="GPKG")


def graph_to_gdf(G):
    # Convert nodes to GeoDataFrame
    nodes = []
    for node, data in G.nodes(data=True):
        lon, lat = data["coord"]
        nodes.append(
            {"id": node, "farness": data.get("farness", 0), "geometry": Point(lon, lat)}
        )
    gdf_nodes = gpd.GeoDataFrame(nodes, crs="EPSG:4326")
    return gdf_nodes


def filter_graph_stations(G, num_remove):
    initial_length = G.number_of_nodes()
    # Get all nodes with the specified attribute, sorted by value (descending)
    nodes_with_attribute = [
        (node, data.get("farness", 0))
        for node, data in G.nodes(data=True)
        if "farness" in data
    ]
    nodes_with_attribute.sort(key=lambda x: x[1], reverse=True)

    # Get top nodes to remove
    nodes_to_remove = [node for node, _ in nodes_with_attribute[:num_remove]]
    assert len(nodes_to_remove) == num_remove, "Not enough nodes to remove"

    # Remove them from the graph
    G.remove_nodes_from(nodes_to_remove)

    assert num_remove > 0 
    assert initial_length - G.number_of_nodes() == num_remove

    return G


def remove_long_edges(G, max_distance, weight_attr="weight"):
    # Find edges that exceed max distance
    edges_to_remove = [
        (u, v)
        for u, v, data in G.edges(data=True)
        if data.get(weight_attr, 0) > max_distance
    ]
    print(f"Removing {len(edges_to_remove)} edges longer than {max_distance} meters")

    # Remove the long edges
    G.remove_edges_from(edges_to_remove)

    return G
