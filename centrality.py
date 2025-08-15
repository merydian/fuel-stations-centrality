import networkx as nx
import osmnx as ox
import numpy as np

def farness_centrality(G, weight=None):
    # Compute farness and normalized farness for each node
    farness = {}
    norm_farness = {}
    n = G.number_of_nodes()
    for node in G.nodes:
        # Shortest path lengths from this node to all others
        lengths = nx.single_source_dijkstra_path_length(G, node, weight="weight")
        # Sum of distances to all reachable nodes except itself
        total_dist = sum(dist for target, dist in lengths.items() if target != node)
        farness[node] = total_dist
        # Normalize by number of reachable nodes minus one (excluding itself)
        reachable = len(lengths) - 1
        norm_farness[node] = total_dist / reachable if reachable > 0 else 0

    # Add as node attributes
    nx.set_node_attributes(G, farness, name="farness")
    nx.set_node_attributes(G, norm_farness, name="norm_farness")

    return G, farness

def download_graph(place):
    # Download the street network for the given place
    G = ox.graph_from_place(place, network_type="drive")
    return G

def get_graph_stats(G):
    stats = {}
    
    # Basic graph statistics
    stats['num_nodes'] = {'value': G.number_of_nodes(), 'unit': 'count'}
    stats['num_edges'] = {'value': G.number_of_edges(), 'unit': 'count'}
    
    # Check connectivity based on graph type
    if G.is_directed():
        stats['is_connected'] = {'value': nx.is_weakly_connected(G), 'unit': 'boolean'}
        stats['num_components'] = {'value': nx.number_weakly_connected_components(G), 'unit': 'count'}
    else:
        stats['is_connected'] = {'value': nx.is_connected(G), 'unit': 'boolean'}
        stats['num_components'] = {'value': nx.number_connected_components(G), 'unit': 'count'}
    
    # Density
    stats['density'] = {'value': nx.density(G), 'unit': 'ratio'}
    
    # Average degree
    degrees = dict(G.degree())
    stats['avg_degree'] = {'value': np.mean(list(degrees.values())), 'unit': 'connections'}
    stats['max_degree'] = {'value': max(degrees.values()), 'unit': 'connections'}
    stats['min_degree'] = {'value': min(degrees.values()), 'unit': 'connections'}
    
    # Centrality measures (for largest connected component if graph is disconnected)
    if G.is_directed():
        if nx.is_weakly_connected(G):
            largest_cc = G
        else:
            largest_cc = G.subgraph(max(nx.weakly_connected_components(G), key=len))
    else:
        if nx.is_connected(G):
            largest_cc = G
        else:
            largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
    
    # Degree centrality
    degree_centrality = nx.degree_centrality(largest_cc)
    stats['avg_degree_centrality'] = {'value': np.mean(list(degree_centrality.values())), 'unit': 'normalized ratio'}
    
    # Closeness centrality
    closeness_centrality = nx.closeness_centrality(largest_cc, distance='weight')
    stats['avg_closeness_centrality'] = {'value': np.mean(list(closeness_centrality.values())), 'unit': 'normalized ratio'}
    
    # Betweenness centrality (can be slow for large graphs)
    if largest_cc.number_of_nodes() < 1000:  # Only compute for smaller graphs
        betweenness_centrality = nx.betweenness_centrality(largest_cc, weight='weight')
        stats['avg_betweenness_centrality'] = {'value': np.mean(list(betweenness_centrality.values())), 'unit': 'normalized ratio'}
    else:
        stats['avg_betweenness_centrality'] = {'value': "Skipped (graph too large)", 'unit': 'n/a'}
    
    # Eigenvector centrality (only for connected graphs)
    is_connected_check = nx.is_weakly_connected(largest_cc) if G.is_directed() else nx.is_connected(largest_cc)
    if is_connected_check:
        try:
            eigenvector_centrality = nx.eigenvector_centrality(largest_cc, weight='weight')
            stats['avg_eigenvector_centrality'] = {'value': np.mean(list(eigenvector_centrality.values())), 'unit': 'normalized ratio'}
        except:
            stats['avg_eigenvector_centrality'] = {'value': "Could not compute", 'unit': 'n/a'}
    else:
        stats['avg_eigenvector_centrality'] = {'value': "Graph not connected", 'unit': 'n/a'}
    
    # Farness centrality (if available)
    farness_values = [data.get('farness', 0) for node, data in G.nodes(data=True) if 'farness' in data]
    if farness_values:
        stats['avg_farness'] = {'value': np.mean(farness_values), 'unit': 'meters'}
        stats['max_farness'] = {'value': max(farness_values), 'unit': 'meters'}
        stats['min_farness'] = {'value': min(farness_values), 'unit': 'meters'}
    
    # Normalized farness centrality (if available)
    norm_farness_values = [data.get('norm_farness', 0) for node, data in G.nodes(data=True) if 'norm_farness' in data]
    if norm_farness_values:
        stats['avg_norm_farness'] = {'value': np.mean(norm_farness_values), 'unit': 'meters/node'}
        stats['max_norm_farness'] = {'value': max(norm_farness_values), 'unit': 'meters/node'}
        stats['min_norm_farness'] = {'value': min(norm_farness_values), 'unit': 'meters/node'}
    
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