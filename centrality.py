import igraph as ig
import osmnx as ox
import numpy as np
import logging

logger = logging.getLogger(__name__)

def farness_centrality(G, weight=None):
    logger.info(f"Computing farness centrality for graph with {G.vcount()} nodes and {G.ecount()} edges")
    
    # Compute farness and normalized farness for each node
    farness = {}
    norm_farness = {}
    n = G.vcount()
    
    logger.debug(f"Using weight attribute: {weight}")
    
    # Use igraph's shortest path distances
    logger.info("Calculating shortest path distances matrix...")
    distances = G.distances(weights=weight)
    logger.info("Distance matrix calculation completed")
    
    for i in range(n):
        # Sum of distances to all reachable nodes except itself
        total_dist = sum(dist for j, dist in enumerate(distances[i]) if i != j and dist != float('inf'))
        farness[i] = total_dist
        # Normalize by number of reachable nodes minus one (excluding itself)
        reachable = sum(1 for dist in distances[i] if dist != float('inf')) - 1
        norm_farness[i] = total_dist / reachable if reachable > 0 else 0
        
        if i % max(1, n // 10) == 0:  # Log progress every 10%
            logger.debug(f"Processed {i}/{n} nodes ({100*i/n:.1f}%)")

    # Add as vertex attributes
    G.vs["farness"] = [farness.get(i, 0) for i in range(n)]
    G.vs["norm_farness"] = [norm_farness.get(i, 0) for i in range(n)]
    
    logger.info(f"Farness centrality computation completed. "
               f"Avg farness: {np.mean(list(farness.values())):.2f}, "
               f"Max farness: {max(farness.values()):.2f}")

    return G, farness

def download_graph(place):
    logger.info(f"Downloading street network for: {place}")
    try:
        # Download the street network for the given place and convert to igraph
        G_nx = ox.graph_from_place(place, network_type="drive")
        logger.info(f"Downloaded NetworkX graph with {len(G_nx.nodes)} nodes and {len(G_nx.edges)} edges")
        
        # Convert NetworkX to igraph
        logger.info("Converting NetworkX graph to igraph...")
        G = ig.Graph.from_networkx(G_nx)
        logger.info(f"Conversion completed. igraph has {G.vcount()} vertices and {G.ecount()} edges")
        
        return G
    except Exception as e:
        logger.error(f"Failed to download graph for {place}: {e}")
        raise

def get_graph_stats(G):
    logger.info("Computing comprehensive graph statistics...")
    stats = {}
    
    # Basic graph statistics
    stats['num_nodes'] = {'value': G.vcount(), 'unit': 'count'}
    stats['num_edges'] = {'value': G.ecount(), 'unit': 'count'}
    logger.debug(f"Basic stats: {stats['num_nodes']['value']} nodes, {stats['num_edges']['value']} edges")
    
    # Check connectivity
    is_connected = G.is_connected(mode="weak")
    stats['is_connected'] = {'value': is_connected, 'unit': 'boolean'}
    components = G.connected_components(mode="weak")
    stats['num_components'] = {'value': len(components), 'unit': 'count'}
    logger.info(f"Graph connectivity: {is_connected}, Components: {len(components)}")
    
    # Density
    stats['density'] = {'value': G.density(), 'unit': 'ratio'}
    
    # Average degree
    degrees = G.degree()
    stats['avg_degree'] = {'value': np.mean(degrees), 'unit': 'connections'}
    stats['max_degree'] = {'value': max(degrees), 'unit': 'connections'}
    stats['min_degree'] = {'value': min(degrees), 'unit': 'connections'}
    logger.debug(f"Degree stats - Avg: {stats['avg_degree']['value']}, Max: {stats['max_degree']['value']}, Min: {stats['min_degree']['value']}")
    
    # Centrality measures (for largest connected component if graph is disconnected)
    if is_connected:
        largest_cc = G
        logger.info("Using entire graph for centrality calculations (graph is connected)")
    else:
        largest_component = max(components, key=len)
        largest_cc = G.subgraph(largest_component)
        logger.info(f"Using largest component with {largest_cc.vcount()} nodes for centrality calculations")
    
    # Degree centrality
    logger.debug("Computing degree centrality...")
    degree_centrality = largest_cc.degree()
    max_possible_degree = largest_cc.vcount() - 1
    normalized_degree_centrality = [d / max_possible_degree for d in degree_centrality] if max_possible_degree > 0 else [0] * largest_cc.vcount()
    stats['avg_degree_centrality'] = {'value': np.mean(normalized_degree_centrality), 'unit': 'normalized ratio'}
    
    # Closeness centrality
    logger.debug("Computing closeness centrality...")
    try:
        closeness_centrality = largest_cc.closeness(weights='weight', normalized=True)
        stats['avg_closeness_centrality'] = {'value': np.mean(closeness_centrality), 'unit': 'normalized ratio'}
    except Exception as e:
        logger.warning(f"Failed to compute closeness centrality: {e}")
        stats['avg_closeness_centrality'] = {'value': "Could not compute", 'unit': 'n/a'}
    
    # Betweenness centrality (can be slow for large graphs)
    if largest_cc.vcount() < 1000:
        logger.debug("Computing betweenness centrality...")
        try:
            betweenness_centrality = largest_cc.betweenness(weights='weight', normalized=True)
            stats['avg_betweenness_centrality'] = {'value': np.mean(betweenness_centrality), 'unit': 'normalized ratio'}
        except Exception as e:
            logger.warning(f"Failed to compute betweenness centrality: {e}")
            stats['avg_betweenness_centrality'] = {'value': "Could not compute", 'unit': 'n/a'}
    else:
        logger.info(f"Skipping betweenness centrality (graph too large: {largest_cc.vcount()} nodes)")
        stats['avg_betweenness_centrality'] = {'value': "Skipped (graph too large)", 'unit': 'n/a'}
    
    # Eigenvector centrality (only for connected graphs)
    if largest_cc.is_connected(mode="weak"):
        logger.debug("Computing eigenvector centrality...")
        try:
            eigenvector_centrality = largest_cc.eigenvector_centrality(weights='weight')
            stats['avg_eigenvector_centrality'] = {'value': np.mean(eigenvector_centrality), 'unit': 'normalized ratio'}
        except Exception as e:
            logger.warning(f"Failed to compute eigenvector centrality: {e}")
            stats['avg_eigenvector_centrality'] = {'value': "Could not compute", 'unit': 'n/a'}
    else:
        logger.info("Skipping eigenvector centrality (largest component not connected)")
        stats['avg_eigenvector_centrality'] = {'value': "Graph not connected", 'unit': 'n/a'}
    
    # Farness centrality (if available)
    if 'farness' in G.vs.attributes():
        logger.debug("Processing farness centrality statistics...")
        farness_values = G.vs['farness']
        stats['avg_farness'] = {'value': np.mean(farness_values), 'unit': 'meters'}
        stats['max_farness'] = {'value': max(farness_values), 'unit': 'meters'}
        stats['min_farness'] = {'value': min(farness_values), 'unit': 'meters'}
    else:
        logger.debug("No farness centrality data found")
    
    # Normalized farness centrality (if available)
    if 'norm_farness' in G.vs.attributes():
        logger.debug("Processing normalized farness centrality statistics...")
        norm_farness_values = G.vs['norm_farness']
        stats['avg_norm_farness'] = {'value': np.mean(norm_farness_values), 'unit': 'meters/node'}
        stats['max_norm_farness'] = {'value': max(norm_farness_values), 'unit': 'meters/node'}
        stats['min_norm_farness'] = {'value': min(norm_farness_values), 'unit': 'meters/node'}
    
    logger.info("Graph statistics computation completed")
    return stats

def format_graph_stats(stats, title="Graph Statistics"):
    """
    Format graph statistics dictionary into a nicely formatted string.
    
    Args:
        stats: Dictionary of graph statistics with units
        title: Title for the statistics display
    
    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"\n{'='*50}")
    lines.append(f"{title:^50}")
    lines.append(f"{'='*50}")
    
    # Basic Statistics
    lines.append("\nBasic Statistics:")
    lines.append(f"  Nodes: {stats['num_nodes']['value']:,} {stats['num_nodes']['unit']}")
    lines.append(f"  Edges: {stats['num_edges']['value']:,} {stats['num_edges']['unit']}")
    lines.append(f"  Connected: {stats['is_connected']['value']} ({stats['is_connected']['unit']})")
    lines.append(f"  Components: {stats['num_components']['value']} {stats['num_components']['unit']}")
    lines.append(f"  Density: {stats['density']['value']:.6f} ({stats['density']['unit']})")
    
    # Degree Statistics
    lines.append("\nDegree Statistics:")
    lines.append(f"  Average: {stats['avg_degree']['value']:.2f} {stats['avg_degree']['unit']}")
    lines.append(f"  Maximum: {stats['max_degree']['value']} {stats['max_degree']['unit']}")
    lines.append(f"  Minimum: {stats['min_degree']['value']} {stats['min_degree']['unit']}")
    
    # Centrality Measures
    lines.append("\nCentrality Measures:")
    lines.append(f"  Avg Degree Centrality: {stats['avg_degree_centrality']['value']:.6f} ({stats['avg_degree_centrality']['unit']})")
    lines.append(f"  Avg Closeness Centrality: {stats['avg_closeness_centrality']['value']:.6f} ({stats['avg_closeness_centrality']['unit']})")
    
    if isinstance(stats['avg_betweenness_centrality']['value'], str):
        lines.append(f"  Avg Betweenness Centrality: {stats['avg_betweenness_centrality']['value']} ({stats['avg_betweenness_centrality']['unit']})")
    else:
        lines.append(f"  Avg Betweenness Centrality: {stats['avg_betweenness_centrality']['value']:.6f} ({stats['avg_betweenness_centrality']['unit']})")
    
    if isinstance(stats['avg_eigenvector_centrality']['value'], str):
        lines.append(f"  Avg Eigenvector Centrality: {stats['avg_eigenvector_centrality']['value']} ({stats['avg_eigenvector_centrality']['unit']})")
    else:
        lines.append(f"  Avg Eigenvector Centrality: {stats['avg_eigenvector_centrality']['value']:.6f} ({stats['avg_eigenvector_centrality']['unit']})")
    
    # Farness Statistics (if available)
    if 'avg_farness' in stats:
        lines.append("\nFarness Centrality:")
        lines.append(f"  Average: {stats['avg_farness']['value']:,.2f} {stats['avg_farness']['unit']}")
        lines.append(f"  Maximum: {stats['max_farness']['value']:,.2f} {stats['max_farness']['unit']}")
        lines.append(f"  Minimum: {stats['min_farness']['value']:,.2f} {stats['min_farness']['unit']}")
    
    # Normalized Farness Statistics (if available)
    if 'avg_norm_farness' in stats:
        lines.append("\nNormalized Farness Centrality:")
        lines.append(f"  Average: {stats['avg_norm_farness']['value']:.6f} {stats['avg_norm_farness']['unit']}")
        lines.append(f"  Maximum: {stats['max_norm_farness']['value']:.6f} {stats['max_norm_farness']['unit']}")
        lines.append(f"  Minimum: {stats['min_norm_farness']['value']:.6f} {stats['min_norm_farness']['unit']}")
    
    lines.append(f"{'='*50}\n")
    
    return "\n".join(lines)

def compare_graph_stats(stats1, stats2, title1="Graph 1", title2="Graph 2"):
    """
    Compare two graph statistics dictionaries and format the output nicely.
    
    Args:
        stats1: First statistics dictionary with units
        stats2: Second statistics dictionary with units
        title1: Title for the first graph
        title2: Title for the second graph
    
    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"{'GRAPH COMPARISON':^70}")
    lines.append(f"{'='*70}")
    lines.append(f"{title1:<35} vs {title2:>33}")
    lines.append(f"{'-'*70}")
    
    # Helper function to format values
    def format_value(stat_dict, precision=None):
        value = stat_dict['value']
        unit = stat_dict['unit']
        
        if isinstance(value, str):
            return f"{value} ({unit})"
        elif isinstance(value, bool):
            return f"{value} ({unit})"
        elif isinstance(value, (int, np.integer)):
            return f"{value:,} {unit}"
        elif isinstance(value, (float, np.floating)):
            if precision is not None:
                return f"{value:.{precision}f} {unit}"
            else:
                return f"{value:.6f} {unit}"
        else:
            return f"{value} ({unit})"
    
    # Helper function to calculate percentage change
    def calc_change(val1, val2):
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if val1 == 0:
                return "N/A" if val2 == 0 else "∞"
            return f"{((val2 - val1) / val1) * 100:+.2f}%"
        return "N/A"
    
    # Basic Statistics
    lines.append("\nBasic Statistics:")
    basic_stats = ['num_nodes', 'num_edges', 'is_connected', 'num_components', 'density']
    
    for stat in basic_stats:
        if stat in stats1 and stat in stats2:
            val1_str = format_value(stats1[stat], 2 if stat == 'density' else None)
            val2_str = format_value(stats2[stat], 2 if stat == 'density' else None)
            change = calc_change(stats1[stat]['value'], stats2[stat]['value'])
            
            stat_name = stat.replace('_', ' ').title()
            lines.append(f"  {stat_name}:")
            lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")
    
    # Degree Statistics
    lines.append("\nDegree Statistics:")
    degree_stats = ['avg_degree', 'max_degree', 'min_degree']
    
    for stat in degree_stats:
        if stat in stats1 and stat in stats2:
            val1_str = format_value(stats1[stat], 2)
            val2_str = format_value(stats2[stat], 2)
            change = calc_change(stats1[stat]['value'], stats2[stat]['value'])
            
            stat_name = stat.replace('_', ' ').title()
            lines.append(f"  {stat_name}:")
            lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")
    
    # Centrality Measures
    lines.append("\nCentrality Measures:")
    centrality_stats = ['avg_degree_centrality', 'avg_closeness_centrality', 
                       'avg_betweenness_centrality', 'avg_eigenvector_centrality']
    
    for stat in centrality_stats:
        if stat in stats1 and stat in stats2:
            val1_str = format_value(stats1[stat], 6)
            val2_str = format_value(stats2[stat], 6)
            change = calc_change(stats1[stat]['value'], stats2[stat]['value'])
            
            stat_name = stat.replace('_', ' ').replace('avg ', '').title()
            lines.append(f"  {stat_name}:")
            lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")
    
    # Farness Statistics (if available)
    farness_stats = ['avg_farness', 'max_farness', 'min_farness']
    if any(stat in stats1 and stat in stats2 for stat in farness_stats):
        lines.append("\nFarness Centrality:")
        
        for stat in farness_stats:
            if stat in stats1 and stat in stats2:
                val1_str = format_value(stats1[stat], 2)
                val2_str = format_value(stats2[stat], 2)
                change = calc_change(stats1[stat]['value'], stats2[stat]['value'])
                
                stat_name = stat.replace('_', ' ').replace('avg ', '').title()
                lines.append(f"  {stat_name}:")
                lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")
    
    # Normalized Farness Statistics (if available)
    norm_farness_stats = ['avg_norm_farness', 'max_norm_farness', 'min_norm_farness']
    if any(stat in stats1 and stat in stats2 for stat in norm_farness_stats):
        lines.append("\nNormalized Farness Centrality:")
        
        for stat in norm_farness_stats:
            if stat in stats1 and stat in stats2:
                val1_str = format_value(stats1[stat], 6)
                val2_str = format_value(stats2[stat], 6)
                change = calc_change(stats1[stat]['value'], stats2[stat]['value'])
                
                stat_name = stat.replace('_', ' ').replace('avg ', '').replace('norm ', '').title()
                lines.append(f"  {stat_name}:")
                lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")
    
    lines.append(f"{'='*70}\n")
    
    return "\n".join(lines)