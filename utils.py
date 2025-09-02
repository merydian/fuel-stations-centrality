import logging
import osmnx as ox
from config import Config
from shapely.geometry import LineString, Point
import geopandas as gpd
import pandas as pd
from tabulate import tabulate


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
    logger.info("Starting gas station extraction from OSM data")
    logger.debug(f"Graph CRS: {G.graph['crs']}")

    # Query gas stations from PBF file
    tags = {"amenity": "fuel"}
    logger.info(f"Querying gas stations from PBF file: {Config.LOCAL_PBF_PATH}")
    gas_stations = ox.features_from_xml(Config.LOCAL_PBF_PATH, tags=tags)
    logger.debug(f"Raw gas stations extracted: {len(gas_stations)}")
    
    gas_stations = gas_stations.to_crs(G.graph["crs"])
    logger.info(f"Reprojected {len(gas_stations)} gas stations to target CRS")

    if Config.MAX_STATIONS:
        original_count = len(gas_stations)
        gas_stations = gas_stations.head(Config.MAX_STATIONS)
        logger.info(f"Limited gas stations from {original_count} to {len(gas_stations)} (MAX_STATIONS={Config.MAX_STATIONS})")

    logger.debug("Finding nearest network nodes for each gas station")

    def get_xy(geom):
        if geom.geom_type == "Point":
            return geom.x, geom.y
        else:  # Polygon or MultiPolygon
            c = geom.centroid
            return c.x, c.y

    gas_stations["nearest_node"] = gas_stations.geometry.apply(
        lambda geom: ox.distance.nearest_nodes(G, *get_xy(geom))
    )
    logger.debug("Completed nearest node mapping for all gas stations")

    # Remove stations further than Config.STATIONS_MAX_RADIUS from their nearest node
    logger.debug(f"Filtering stations within {Config.STATIONS_MAX_RADIUS}m of nearest network node")
    
    def node_distance(row):
        node = row["nearest_node"]
        geom = row.geometry
        node_x = G.nodes[node]["x"]
        node_y = G.nodes[node]["y"]
        return Point(node_x, node_y).distance(geom)

    gas_stations["node_dist"] = gas_stations.apply(node_distance, axis=1)
    initial_count = len(gas_stations)
    gas_stations = gas_stations[gas_stations["node_dist"] <= Config.STATIONS_MAX_RADIUS]
    filtered_count = len(gas_stations)
    logger.info(f"Filtered gas stations: {initial_count} → {filtered_count} (removed {initial_count - filtered_count} stations beyond radius)")

    unique_nodes = len(set(gas_stations["nearest_node"].tolist()))
    logger.info(f"Final result: {filtered_count} gas stations mapped to {unique_nodes} unique network nodes")
    assert unique_nodes > 1

    return gas_stations["nearest_node"].tolist()


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

    logger.info("Starting NetworkX to igraph conversion")
    logger.debug(f"Input NetworkX graph: {len(G_nx.nodes())} nodes, {len(G_nx.edges())} edges")

    # Create node mapping
    node_list = list(G_nx.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    logger.debug(f"Created node mapping for {len(node_list)} nodes")

    # Create edges for igraph
    logger.debug("Processing edges and weights")
    edges = []
    edge_weights = []
    edge_lengths = []

    for u, v, data in G_nx.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        # Use 'length' attribute if available, fallback to 'weight', then 1.0
        length = data.get("length", data.get("weight", 1.0))
        edge_weights.append(length)
        edge_lengths.append(length)

    logger.debug(f"Processed {len(edges)} edges with weights")

    # Create igraph
    logger.debug("Creating igraph instance")
    G_ig = ig.Graph(n=len(node_list), edges=edges, directed=False)
    G_ig.es["weight"] = edge_weights
    G_ig.es["length"] = edge_lengths

    # Add node attributes - preserve projected coordinates
    logger.debug("Adding node attributes (projected coordinates)")
    for i, node in enumerate(node_list):
        node_data = G_nx.nodes[node]
        G_ig.vs[i]["name"] = str(node)
        G_ig.vs[i]["x"] = float(node_data.get("x", 0.0))  # Projected x coordinate
        G_ig.vs[i]["y"] = float(node_data.get("y", 0.0))  # Projected y coordinate

    logger.info(f"✓ NetworkX to igraph conversion complete: {G_ig.vcount()} nodes, {G_ig.ecount()} edges")
    return G_ig

def igraph_edges_to_gpkg(g, name):
    logger.info(f"Exporting igraph edges to GeoPackage: {name}")
    logger.debug(f"Processing {g.ecount()} edges for export")
    
    edges = g.es
    edge_gdf = gpd.GeoDataFrame(
        {"source": [e.source for e in edges],
        "target": [e.target for e in edges]},
        geometry=[LineString([(g.vs[e.source]["x"], g.vs[e.source]["y"]),
                            (g.vs[e.target]["x"], g.vs[e.target]["y"])])
                for e in edges],
        crs=f"EPSG:{Config.EPSG_CODE}"
    )
    
    output_path = f"{Config.OUTPUT_DIR}/{name}_{Config.PLACE.lower()}_edges.gpkg"
    edge_gdf.to_file(output_path, layer=name, driver="GPKG")
    logger.info(f"✓ Edges exported to: {output_path}")

def ig_nodes_to_gpkg(G, selected_nodes, name):
    """
    Export igraph nodes to a GeoPackage as points.

    Parameters:
    - G: igraph.Graph
    - selected_nodes: list of vertex indices to export
    - name: name of the output layer
    """
    logger.info(f"Exporting {len(selected_nodes)} igraph nodes to GeoPackage: {name}")
    
    geometries = []
    node_data = []

    for node_idx in selected_nodes:
        # Get node attributes from igraph vertex
        vertex = G.vs[node_idx]
        x = vertex["x"]
        y = vertex["y"]
        
        # Create point geometry
        geometries.append(Point(x, y))
        
        # Collect node attributes - use try/except for optional attributes
        try:
            vertex_name = vertex["name"]
        except KeyError:
            vertex_name = str(node_idx)
            
        node_data.append({
            "node_id": node_idx,
            "name": vertex_name
        })

    logger.debug(f"Created {len(geometries)} point geometries")

    # Create GeoDataFrame
    node_gdf = gpd.GeoDataFrame(
        node_data, 
        geometry=geometries, 
        crs=f"EPSG:{Config.EPSG_CODE}"
    )
    
    # Export to GeoPackage
    output_path = f"{Config.OUTPUT_DIR}/{name}_nodes.gpkg"
    node_gdf.to_file(output_path, layer=name, driver="GPKG")
    logger.info(f"✓ Nodes exported to: {output_path}")


def prune_igraph_by_distance(G, stations: list, max_dist: int):
    """
    Remove edges from igraph G that are further than `max_dist` graph-distance 
    away from any node in `stations`.

    Parameters
    ----------
    G : igraph.Graph
        The input igraph (modified in-place).
    stations : list
        List of station vertex indices.
    max_dist : int
        Maximum allowed graph distance from any station.

    Returns
    -------
    igraph.Graph
        The pruned graph (same object as G).
    """    
    logger.info(f"Pruning graph by distance: max_dist={max_dist}, stations={len(stations)}")
    logger.debug(f"Initial graph: {G.vcount()} nodes, {G.ecount()} edges")
    
    # Compute shortest path lengths from all stations
    logger.debug("Computing shortest path distances from all stations")
    distances = G.distances(source=stations, weights="length", mode="all")
    
    # Find nodes within max_dist of any station
    logger.debug(f"Finding nodes within {max_dist} units of any station")
    valid_nodes = set()
    for i, station_distances in enumerate(distances):
        for node_idx, dist in enumerate(station_distances):
            if dist <= max_dist:
                valid_nodes.add(node_idx)
    
    logger.debug(f"Found {len(valid_nodes)} valid nodes within distance threshold")
    
    # Find edges where both endpoints are outside valid set
    edges_to_remove = []
    for edge in G.es:
        if edge.source not in valid_nodes or edge.target not in valid_nodes:
            edges_to_remove.append(edge.index)
    
    initial_edges = G.ecount()
    logger.debug(f"Removing {len(edges_to_remove)} edges that don't connect valid nodes")
    
    # Remove invalid edges
    G.delete_edges(edges_to_remove)
    
    final_edges = G.ecount()
    logger.info(f"✓ Graph pruning complete: {initial_edges} → {final_edges} edges (removed {initial_edges - final_edges})")
    
    return G


def nx_knn_nodes_to_gpkg(G, selected_nodes, name):
    logger.info(f"Exporting {len(selected_nodes)} NetworkX k-NN nodes to GeoPackage: {name}")
    
    geometries = []
    knn_values = []

    for node, knn in selected_nodes:
        attrs = G.nodes[node]
        x = attrs.get("x")
        y = attrs.get("y")
        geometries.append(Point(x, y))
        knn_values.append(knn)

    logger.debug(f"Created geometries and collected k-NN values for {len(selected_nodes)} nodes")

    node_gdf = gpd.GeoDataFrame(
        {"knn": knn_values}, 
        geometry=geometries, 
        crs=f"EPSG:{Config.EPSG_CODE}"
    )

    output_path = f"{Config.OUTPUT_DIR}/{name}_nodes.gpkg"
    node_gdf.to_file(output_path, layer=name, driver="GPKG")
    logger.info(f"✓ k-NN nodes exported to: {output_path}")

def generate_centrality_table(graphs_dict):
    """
    Generate centrality measures table for multiple graphs.
    
    Parameters:
    -----------
    graphs_dict : dict
        Dictionary with graph names as keys and igraph.Graph objects as values
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing centrality measures for each graph
    """
    logger.info("=== Starting Centrality Measures Computation ===")
    logger.debug(f"Processing {len(graphs_dict)} graphs: {list(graphs_dict.keys())}")
    
    results = []
    for name, graph in graphs_dict.items():
        logger.info(f"Computing centrality measures for graph: {name}")
        logger.debug(f"  Graph size: {graph.vcount()} nodes, {graph.ecount()} edges")
        
        # Compute centrality measures
        logger.debug("  Computing closeness centrality...")
        closeness = graph.closeness()
        logger.debug("  Computing betweenness centrality...")
        betweenness = graph.betweenness()
        logger.debug("  Computing degree centrality...")
        degree = graph.degree()
        
        # Calculate averages
        avg_closeness = sum(closeness) / len(closeness)
        avg_betweenness = sum(betweenness) / len(betweenness)
        avg_degree = sum(degree) / len(degree)
        
        logger.debug(f"  Results - Closeness: {avg_closeness:.4f}, Betweenness: {avg_betweenness:.4f}, Degree: {avg_degree:.2f}")
        
        # Store results
        results.append({
            'Graph Scenario': name,
            'Nodes': graph.vcount(),
            'Edges': graph.ecount(),
            'Closeness (avg)': f"{avg_closeness:.4f}",
            'Betweenness (avg)': f"{avg_betweenness:.4f}",
            'Degree (avg)': f"{avg_degree:.2f}"
        })

    logger.info("Centrality computation completed for all graphs")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print console table
    print("\n" + "="*60)
    print("CENTRALITY MEASURES SUMMARY")
    print("="*60)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Generate LaTeX table
    logger.debug("Generating LaTeX table for centrality measures")
    latex_columns = ['Graph Scenario', 'Closeness (avg)', 'Betweenness (avg)', 'Degree (avg)']
    latex_df = df[latex_columns]
    
    latex_table = latex_df.to_latex(
        index=False,
        column_format='lccc',
        caption='Centrality Measures Comparison for Different Graph Scenarios',
        label='tab:centrality_measures',
        escape=False
    )
    
    # Save LaTeX table to file
    output_file = f"{Config.OUTPUT_DIR}/centrality_table{Config.PLACE.lower()}.tex"
    with open(output_file, "w") as f:
        f.write(latex_table)
    
    logger.info(f"✓ LaTeX centrality table saved to: {output_file}")
    
    return df

def generate_graph_info_table(graphs_dict):
    """
    Generate general information table for multiple graphs including total length.
    
    Parameters:
    -----------
    graphs_dict : dict
        Dictionary with graph names as keys and igraph.Graph objects as values
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing general graph information
    """
    logger.info("=== Starting Graph Information Computation ===")
    logger.debug(f"Processing {len(graphs_dict)} graphs: {list(graphs_dict.keys())}")
    
    results = []
    for name, graph in graphs_dict.items():
        logger.info(f"Computing graph information for: {name}")
        
        # Basic graph metrics
        num_nodes = graph.vcount()
        num_edges = graph.ecount()
        logger.debug(f"  Basic metrics - Nodes: {num_nodes:,}, Edges: {num_edges:,}")
        
        # Calculate total length
        logger.debug("  Computing total edge lengths...")
        edge_lengths = graph.es["length"]
        total_length = sum(edge_lengths)
        total_length_km = total_length / 1000  # Convert to kilometers
        
        # Calculate average edge length
        avg_edge_length = total_length / num_edges if num_edges > 0 else 0
        
        # Calculate graph density
        logger.debug("  Computing graph density...")
        max_edges = num_nodes * (num_nodes - 1) / 2  # For undirected graph
        density = num_edges / max_edges if max_edges > 0 else 0
        
        logger.debug(f"  Computed metrics - Total length: {total_length_km:.2f} km, Density: {density:.6f}")
        
        # Store results
        results.append({
            'Graph Scenario': name,
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Total Length (km)': f"{total_length_km:.2f}",
            'Avg Edge Length (m)': f"{avg_edge_length:.1f}",
            'Density': f"{density:.6f}"
        })

    logger.info("Graph information computation completed for all graphs")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print console table
    print("\n" + "="*80)
    print("GRAPH INFORMATION SUMMARY")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Generate LaTeX table
    logger.debug("Generating LaTeX table for graph information")
    latex_table = df.to_latex(
        index=False,
        column_format='lrrrrr',
        caption='Graph Information Summary for Different Scenarios',
        label='tab:graph_info',
        escape=False
    )
    
    # Save LaTeX table to file
    output_file = f"{Config.OUTPUT_DIR}/graph_info_{Config.PLACE.lower()}.tex"
    with open(output_file, "w") as f:
        f.write(latex_table)
    
    logger.info(f"✓ Graph info LaTeX table saved to: {output_file}")
    
    return df
