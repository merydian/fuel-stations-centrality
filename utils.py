import logging
import osmnx as ox
from shapely.geometry import LineString, Point
import geopandas as gpd
import pandas as pd
from tabulate import tabulate
import requests
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from shapely.ops import voronoi_diagram
import numpy as np


logger = logging.getLogger(__name__)


class Utils:
    def __init__(self, config):
        self.config = config

    def get_gas_stations_from_graph(self, G):
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
        logger.info(
            f"Querying gas stations from PBF file: {self.config.LOCAL_PBF_PATH}"
        )
        gas_stations = ox.features_from_xml(self.config.LOCAL_PBF_PATH, tags=tags)
        logger.debug(f"Raw gas stations extracted: {len(gas_stations)}")

        gas_stations = gas_stations.to_crs(G.graph["crs"])
        logger.info(f"Reprojected {len(gas_stations)} gas stations to target CRS")

        if self.config.MAX_STATIONS:
            original_count = len(gas_stations)
            gas_stations = gas_stations.head(self.config.MAX_STATIONS)
            logger.info(
                f"Limited gas stations from {original_count} to {len(gas_stations)} (MAX_STATIONS={self.config.MAX_STATIONS})"
            )

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

        # Remove stations further than self.config.STATIONS_MAX_RADIUS from their nearest node
        logger.debug(
            f"Filtering stations within {self.config.STATIONS_MAX_RADIUS}m of nearest network node"
        )

        def node_distance(row):
            node = row["nearest_node"]
            geom = row.geometry
            node_x = G.nodes[node]["x"]
            node_y = G.nodes[node]["y"]
            return Point(node_x, node_y).distance(geom)

        gas_stations["node_dist"] = gas_stations.apply(node_distance, axis=1)
        initial_count = len(gas_stations)
        gas_stations = gas_stations[
            gas_stations["node_dist"] <= self.config.STATIONS_MAX_RADIUS
        ]
        filtered_count = len(gas_stations)
        logger.info(
            f"Filtered gas stations: {initial_count} → {filtered_count} (removed {initial_count - filtered_count} stations beyond radius)"
        )

        unique_nodes = len(set(gas_stations["nearest_node"].tolist()))
        logger.info(
            f"Final result: {filtered_count} gas stations mapped to {unique_nodes} unique network nodes"
        )
        assert unique_nodes > 1

        return gas_stations["nearest_node"].tolist()

    def convert_networkx_to_igraph(self, G_nx):
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
        logger.debug(
            f"Input NetworkX graph: {len(G_nx.nodes())} nodes, {len(G_nx.edges())} edges"
        )

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

        logger.info(
            f"✓ NetworkX to igraph conversion complete: {G_ig.vcount()} nodes, {G_ig.ecount()} edges"
        )
        return G_ig

    def igraph_edges_to_gpkg(self, g, name):
        logger.info(f"Exporting igraph edges to GeoPackage: {name}")
        logger.debug(f"Processing {g.ecount()} edges for export")

        edges = g.es
        edge_gdf = gpd.GeoDataFrame(
            {"source": [e.source for e in edges], "target": [e.target for e in edges]},
            geometry=[
                LineString(
                    [
                        (g.vs[e.source]["x"], g.vs[e.source]["y"]),
                        (g.vs[e.target]["x"], g.vs[e.target]["y"]),
                    ]
                )
                for e in edges
            ],
            crs=f"EPSG:{self.config.EPSG_CODE}",
        )

        output_path = (
            f"{self.config.OUTPUT_DIR}/{name}_{self.config.PLACE.lower()}_edges.gpkg"
        )
        edge_gdf.to_file(output_path, layer=name, driver="GPKG")
        logger.info(f"✓ Edges exported to: {output_path}")

    def ig_nodes_to_gpkg(self, G, selected_nodes, name):
        """
        Export igraph nodes to a GeoPackage as points.

        Parameters:
        - G: igraph.Graph
        - selected_nodes: list of vertex indices to export
        - name: name of the output layer
        """
        logger.info(
            f"Exporting {len(selected_nodes)} igraph nodes to GeoPackage: {name}"
        )

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

            node_data.append({"node_id": node_idx, "name": vertex_name})

        logger.debug(f"Created {len(geometries)} point geometries")

        # Create GeoDataFrame
        node_gdf = gpd.GeoDataFrame(
            node_data, geometry=geometries, crs=f"EPSG:{self.config.EPSG_CODE}"
        )

        # Export to GeoPackage
        output_path = f"{self.config.OUTPUT_DIR}/{name}_nodes.gpkg"
        node_gdf.to_file(output_path, layer=name, driver="GPKG")
        logger.info(f"✓ Nodes exported to: {output_path}")

    def prune_igraph_by_distance(self, G, stations: list, max_dist: int):
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
        logger.info(
            f"Pruning graph by distance: max_dist={max_dist}, stations={len(stations)}"
        )
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
        logger.debug(
            f"Removing {len(edges_to_remove)} edges that don't connect valid nodes"
        )

        # Remove invalid edges
        G.delete_edges(edges_to_remove)

        final_edges = G.ecount()
        logger.info(
            f"✓ Graph pruning complete: {initial_edges} → {final_edges} edges (removed {initial_edges - final_edges})"
        )

        return G

    def nx_knn_nodes_to_gpkg(self, G, selected_nodes, name):
        logger.info(
            f"Exporting {len(selected_nodes)} NetworkX k-NN nodes to GeoPackage: {name}"
        )

        geometries = []
        knn_values = []

        for node, knn in selected_nodes:
            attrs = G.nodes[node]
            x = attrs.get("x")
            y = attrs.get("y")
            geometries.append(Point(x, y))
            knn_values.append(knn)

        logger.debug(
            f"Created geometries and collected k-NN values for {len(selected_nodes)} nodes"
        )

        node_gdf = gpd.GeoDataFrame(
            {"knn": knn_values},
            geometry=geometries,
            crs=f"EPSG:{self.config.EPSG_CODE}",
        )

        output_path = f"{self.config.OUTPUT_DIR}/{name}_nodes.gpkg"
        node_gdf.to_file(output_path, layer=name, driver="GPKG")
        logger.info(f"✓ k-NN nodes exported to: {output_path}")

    def generate_centrality_table(self, graphs_dict):
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
        logger.debug(
            f"Processing {len(graphs_dict)} graphs: {list(graphs_dict.keys())}"
        )

        results = []
        for name, graph in graphs_dict.items():
            logger.info(f"Computing centrality measures for graph: {name}")
            logger.debug(
                f"  Graph size: {graph.vcount()} nodes, {graph.ecount()} edges"
            )

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

            logger.debug(
                f"  Results - Closeness: {avg_closeness:.4f}, Betweenness: {avg_betweenness:.4f}, Degree: {avg_degree:.2f}"
            )

            # Store results
            results.append(
                {
                    "Graph Scenario": name,
                    "Nodes": graph.vcount(),
                    "Edges": graph.ecount(),
                    "Closeness (avg)": f"{avg_closeness:.4f}",
                    "Betweenness (avg)": f"{avg_betweenness:.4f}",
                    "Degree (avg)": f"{avg_degree:.2f}",
                }
            )

        logger.info("Centrality computation completed for all graphs")

        # Create DataFrame
        df = pd.DataFrame(results)

        # Print console table
        print("\n" + "=" * 60)
        print("CENTRALITY MEASURES SUMMARY")
        print("=" * 60)
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

        # Generate LaTeX table
        logger.debug("Generating LaTeX table for centrality measures")
        latex_columns = [
            "Graph Scenario",
            "Closeness (avg)",
            "Betweenness (avg)",
            "Degree (avg)",
        ]
        latex_df = df[latex_columns]

        latex_table = latex_df.to_latex(
            index=False,
            column_format="lccc",
            caption="Centrality Measures Comparison for Different Graph Scenarios",
            label="tab:centrality_measures",
            escape=False,
        )

        # Save LaTeX table to file
        output_file = (
            f"{self.config.OUTPUT_DIR}/centrality_table{self.config.PLACE.lower()}.tex"
        )
        with open(output_file, "w") as f:
            f.write(latex_table)

        logger.info(f"✓ LaTeX centrality table saved to: {output_file}")

        df.to_csv(f"{self.config.OUTPUT_DIR}/centrality_table_{self.config.PLACE.lower()}.csv", index=False)

        return df

    def generate_graph_info_table(self, graphs_dict):
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
        logger.debug(
            f"Processing {len(graphs_dict)} graphs: {list(graphs_dict.keys())}"
        )

        results = []
        for name, graph in graphs_dict.items():
            logger.info(f"Computing graph information for: {name}")

            # Basic graph metrics
            num_nodes = graph.vcount()
            num_edges = graph.ecount()
            logger.debug(
                f"  Basic metrics - Nodes: {num_nodes:,}, Edges: {num_edges:,}"
            )

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

            logger.debug(
                f"  Computed metrics - Total length: {total_length_km:.2f} km, Density: {density:.6f}"
            )

            # Store results
            results.append(
                {
                    "Graph Scenario": name,
                    "Nodes": num_nodes,
                    "Edges": num_edges,
                    "Total Length (km)": f"{total_length_km:.2f}",
                    "Avg Edge Length (m)": f"{avg_edge_length:.1f}",
                    "Density": f"{density:.6f}",
                }
            )

        logger.info("Graph information computation completed for all graphs")

        # Create DataFrame
        df = pd.DataFrame(results)

        # Print console table
        print("\n" + "=" * 80)
        print("GRAPH INFORMATION SUMMARY")
        print("=" * 80)
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

        # Generate LaTeX table
        logger.debug("Generating LaTeX table for graph information")
        latex_table = df.to_latex(
            index=False,
            column_format="lrrrrr",
            caption="Graph Information Summary for Different Scenarios",
            label="tab:graph_info",
            escape=False,
        )

        # Save LaTeX table to file
        output_file = (
            f"{self.config.OUTPUT_DIR}/graph_info_{self.config.PLACE.lower()}.tex"
        )
        with open(output_file, "w") as f:
            f.write(latex_table)

        logger.info(f"✓ Graph info LaTeX table saved to: {output_file}")

        df.to_csv(f"{self.config.OUTPUT_DIR}/graph_info_table_{self.config.PLACE.lower()}.csv", index=False)

        return df


    def generate_voronoi_polygons(self, graphs_dict, country_name):
        """
        Generate Voronoi polygons for ALL nodes in each graph with country boundary as hull.
        
        Parameters:
        -----------
        graphs_dict : dict
            Dictionary with graph names as keys and igraph.Graph objects as values
            e.g., {"Original": G_road_ig, "KNN Filtered": G_road_filtered_ig, "Randomized Filtered": G_road_random_ig}
        country_name : str
            Name of the country to use as outer boundary (e.g., 'Germany', 'France')
            
        Returns:
        --------
        dict
            Dictionary containing area statistics for each graph scenario
        """    
        logger.info(f"=== Generating Voronoi polygons for ALL nodes across {len(graphs_dict)} graphs ===")
        logger.info(f"Country boundary: {country_name}")
        logger.info(f"Graph scenarios: {list(graphs_dict.keys())}")
        
        # Get country geometry from web (SHARED FOR ALL GRAPHS - SAME OUTER HULL)
        logger.debug("Fetching country geometry from web (shared boundary for all scenarios)...")
        try:
            # Use OSMnx to get country boundary
            logger.debug(f"Fetching {country_name} boundary using OSMnx...")
            country_gdf = ox.geocode_to_gdf(country_name)
            country_boundary = country_gdf.geometry.iloc[0]
            
            # Reproject to match graph CRS
            country_gdf_proj = country_gdf.to_crs(f"EPSG:{self.config.EPSG_CODE}")
            country_boundary_proj = country_gdf_proj.geometry.iloc[0]
            
            logger.info(f"✓ Successfully fetched {country_name} boundary (SHARED HULL)")
            
        except Exception as e:
            logger.error(f"Failed to fetch country boundary: {e}")
            # Fallback: create a large bounding box around ALL nodes from ALL graphs
            logger.warning("Using fallback bounding box as boundary (computed from all graphs)")
            
            # Collect ALL coordinates from ALL graphs for consistent bounding box
            all_x_coords = []
            all_y_coords = []
            for graph_name, G in graphs_dict.items():
                all_x_coords.extend([G.vs[i]["x"] for i in range(G.vcount())])
                all_y_coords.extend([G.vs[i]["y"] for i in range(G.vcount())])
            
            # Create expanded bounding box based on ALL graph data
            margin = 50000  # 50km margin
            min_x, max_x = min(all_x_coords) - margin, max(all_x_coords) + margin
            min_y, max_y = min(all_y_coords) - margin, max(all_y_coords) + margin
            
            country_boundary_proj = Polygon([
                (min_x, min_y), (max_x, min_y), 
                (max_x, max_y), (min_x, max_y), (min_x, min_y)
            ])
            
            logger.info(f"✓ Created fallback bounding box: ({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f})")
        
        # Log the boundary area for verification
        boundary_area_sqkm = country_boundary_proj.area / 1_000_000
        logger.info(f"Shared outer hull area: {boundary_area_sqkm:,.2f} km² (will be same for all scenarios)")
        
        # Process each graph scenario with the SAME boundary
        all_stats = {}
        
        for graph_name, G in graphs_dict.items():
            logger.info(f"Processing graph scenario: {graph_name} (using shared boundary)")
            logger.debug(f"  Graph size: {G.vcount()} nodes, {G.ecount()} edges")
            
            # Use ALL nodes in the graph
            all_node_ids = list(range(G.vcount()))
            logger.info(f"  Using ALL {len(all_node_ids)} nodes from graph {graph_name}")
            
            if len(all_node_ids) < 3:
                logger.warning(f"  Skipping {graph_name}: insufficient nodes ({len(all_node_ids)}) for Voronoi diagram")
                continue
            
            # Extract coordinates for ALL nodes
            logger.debug(f"  Extracting coordinates for ALL {len(all_node_ids)} nodes")
            points = []
            node_data = []
            
            for node_id in all_node_ids:
                vertex = G.vs[node_id]
                x, y = vertex["x"], vertex["y"]
                points.append([x, y])
                
                # Get vertex name if available
                try:
                    vertex_name = vertex["name"]
                except KeyError:
                    vertex_name = str(node_id)
                    
                node_data.append({
                    "node_id": node_id,
                    "name": vertex_name,
                    "x": x,
                    "y": y,
                    "graph_scenario": graph_name
                })
            
            points = np.array(points)
            logger.debug(f"  Prepared {len(points)} point coordinates")
            
            # Add boundary points to ensure consistent outer hull
            logger.debug(f"  Adding boundary points to ensure consistent Voronoi hull...")
            
            # Handle both Polygon and MultiPolygon geometries
            if country_boundary_proj.geom_type == 'Polygon':
                boundary_coords = list(country_boundary_proj.exterior.coords[:-1])  # Remove duplicate last point
            elif country_boundary_proj.geom_type == 'MultiPolygon':
                # Use the largest polygon from the MultiPolygon
                largest_polygon = max(country_boundary_proj.geoms, key=lambda p: p.area)
                boundary_coords = list(largest_polygon.exterior.coords[:-1])
                logger.debug(f"  Using largest polygon from MultiPolygon (area: {largest_polygon.area/1_000_000:.2f} km²)")
            else:
                # Fallback: create a simple bounding box
                logger.warning(f"  Unexpected geometry type: {country_boundary_proj.geom_type}, using bounding box")
                bounds = country_boundary_proj.bounds
                boundary_coords = [
                    (bounds[0], bounds[1]),  # min_x, min_y
                    (bounds[2], bounds[1]),  # max_x, min_y
                    (bounds[2], bounds[3]),  # max_x, max_y
                    (bounds[0], bounds[3])   # min_x, max_y
                ]
            
            # Sample boundary points to add as "virtual" stations for consistent hull
            # This ensures all Voronoi diagrams have the same outer boundary
            n_boundary_points = min(20, len(boundary_coords))  # Use up to 20 boundary points
            step = max(1, len(boundary_coords) // n_boundary_points)
            sampled_boundary_points = boundary_coords[::step]
            
            # Combine actual node points with boundary points
            all_points = np.vstack([points, np.array(sampled_boundary_points)])
            logger.debug(f"  Total points for Voronoi: {len(points)} nodes + {len(sampled_boundary_points)} boundary = {len(all_points)}")
            
            # Create Voronoi diagram with boundary points included
            logger.debug(f"  Computing Voronoi diagram for {graph_name} (with boundary points)...")
            voronoi = Voronoi(all_points)
            
            # Convert Voronoi regions to Shapely polygons (only for original node points, not boundary points)
            logger.debug(f"  Converting Voronoi regions to polygons (excluding boundary point regions)...")
            voronoi_polygons = []
            valid_indices = []
            
            # Process only the first len(points) regions (corresponding to actual nodes)
            for i in range(len(points)):
                region_idx = voronoi.point_region[i]
                region = voronoi.regions[region_idx]
                
                # Skip infinite regions
                if -1 in region or len(region) < 3:
                    continue
                    
                # Create polygon from vertices
                vertices = [voronoi.vertices[j] for j in region]
                polygon = Polygon(vertices)
                
                # Clip to country boundary (SAME BOUNDARY FOR ALL SCENARIOS)
                clipped_polygon = polygon.intersection(country_boundary_proj)
                
                if clipped_polygon.is_empty or clipped_polygon.area < 1:  # Skip tiny polygons
                    continue
                    
                voronoi_polygons.append(clipped_polygon)
                valid_indices.append(i)
            
            logger.info(f"  ✓ Created {len(voronoi_polygons)} valid Voronoi polygons for {graph_name}")
            
            # Verify all polygons are within the same boundary
            total_voronoi_area = sum(poly.area for poly in voronoi_polygons) / 1_000_000
            logger.debug(f"  Total Voronoi area: {total_voronoi_area:,.2f} km² (boundary area: {boundary_area_sqkm:,.2f} km²)")
            
            # Create GeoDataFrame
            logger.debug(f"  Creating GeoDataFrame for export...")
            valid_node_data = [node_data[i] for i in valid_indices]
            
            # Calculate areas (in square meters)
            areas = [poly.area for poly in voronoi_polygons]
            for i, area in enumerate(areas):
                valid_node_data[i]["area_sqm"] = area
                valid_node_data[i]["area_sqkm"] = area / 1_000_000
                valid_node_data[i]["boundary_area_sqkm"] = boundary_area_sqkm  # Add boundary reference
            
            voronoi_gdf = gpd.GeoDataFrame(
                valid_node_data,
                geometry=voronoi_polygons,
                crs=f"EPSG:{self.config.EPSG_CODE}"
            )
            
            # Export to GeoPackage
            output_path = f"{self.config.OUTPUT_DIR}/voronoi_polygons_{graph_name.lower().replace(' ', '_')}_{country_name.lower()}_{self.config.PLACE.lower()}.gpkg"
            voronoi_gdf.to_file(output_path, layer="voronoi_polygons", driver="GPKG")
            logger.info(f"  ✓ Voronoi polygons exported to: {output_path}")
            
            # Calculate area statistics
            logger.debug(f"  Computing area statistics for {graph_name}...")
            areas_sqkm = voronoi_gdf["area_sqkm"].values
            
            area_stats = {
                "graph_scenario": graph_name,
                "total_nodes": G.vcount(),
                "valid_polygons": len(areas_sqkm),
                "boundary_area_sqkm": boundary_area_sqkm,  # Same for all scenarios
                "total_area_sqkm": float(np.sum(areas_sqkm)),
                "mean_area_sqkm": float(np.mean(areas_sqkm)),
                "median_area_sqkm": float(np.median(areas_sqkm)),
                "std_area_sqkm": float(np.std(areas_sqkm)),
                "min_area_sqkm": float(np.min(areas_sqkm)),
                "max_area_sqkm": float(np.max(areas_sqkm)),
                "q25_area_sqkm": float(np.percentile(areas_sqkm, 25)),
                "q75_area_sqkm": float(np.percentile(areas_sqkm, 75)),
                "coverage_ratio": float(np.sum(areas_sqkm) / boundary_area_sqkm),  # How much of boundary is covered
                "polygon_success_rate": float(len(areas_sqkm) / G.vcount())  # Percentage of nodes that created valid polygons
            }
            
            all_stats[graph_name] = area_stats
        
        # Print comparative statistics with boundary verification
        print(f"\n{'='*90}")
        print(f"VORONOI POLYGON AREA STATISTICS COMPARISON - {country_name.upper()}")
        print(f"SHARED BOUNDARY AREA: {boundary_area_sqkm:,.2f} km² (SAME FOR ALL SCENARIOS)")
        print(f"{'='*90}")
        
        # Create comparison table
        comparison_data = []
        for graph_name, stats in all_stats.items():
            comparison_data.append({
                "Graph Scenario": graph_name,
                "Total Nodes": stats["total_nodes"],
                "Valid Polygons": stats["valid_polygons"],
                "Success Rate %": f"{stats['polygon_success_rate']*100:.1f}%",
                "Total Area (km²)": f"{stats['total_area_sqkm']:,.2f}",
                "Coverage %": f"{stats['coverage_ratio']*100:.1f}%",
                "Mean Area (km²)": f"{stats['mean_area_sqkm']:,.2f}",
                "Median Area (km²)": f"{stats['median_area_sqkm']:,.2f}",
                "Std Dev (km²)": f"{stats['std_area_sqkm']:,.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(tabulate(comparison_df, headers="keys", tablefmt="grid", showindex=False))
        
        # Save comparison statistics to CSV
        stats_output = f"{self.config.OUTPUT_DIR}/voronoi_area_stats_comparison_{country_name.lower()}_{self.config.PLACE.lower()}.csv"
        comparison_df.to_csv(stats_output, index=False)
        logger.info(f"✓ Comparison statistics saved to: {stats_output}")
        
        # Save detailed statistics for each graph
        for graph_name, stats in all_stats.items():
            detailed_stats_df = pd.DataFrame([stats])
            detailed_output = f"{self.config.OUTPUT_DIR}/voronoi_area_stats_{graph_name.lower().replace(' ', '_')}_{country_name.lower()}_{self.config.PLACE.lower()}.csv"
            detailed_stats_df.to_csv(detailed_output, index=False)
        
        # Verify boundary consistency
        boundary_areas = [stats["boundary_area_sqkm"] for stats in all_stats.values()]
        if len(set(boundary_areas)) == 1:
            logger.info(f"✓ BOUNDARY CONSISTENCY VERIFIED: All scenarios use the same {boundary_areas[0]:,.2f} km² boundary")
        else:
            logger.error(f"✗ BOUNDARY INCONSISTENCY DETECTED: Different boundary areas found: {boundary_areas}")
        
        logger.info("=== Voronoi polygon generation completed for all graph scenarios ===")
        
        return all_stats
