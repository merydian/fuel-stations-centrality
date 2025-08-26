import logging
import numpy as np
import pyproj
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import transform
from scipy.spatial import Voronoi
from centrality import straightness_centrality, graph_straightness

logger = logging.getLogger(__name__)


def get_graph_stats(G, base_convex_hull=None):
    logger.info("Computing comprehensive graph statistics...")
    from config import Config
    stats = {}

    # Basic graph statistics
    stats["num_nodes"] = {"value": G.vcount(), "unit": "count"}
    stats["num_edges"] = {"value": G.ecount(), "unit": "count"}
    logger.debug(
        f"Basic stats: {stats['num_nodes']['value']} nodes, {stats['num_edges']['value']} edges"
    )

    # Check connectivity
    is_connected = G.is_connected(mode="weak")
    stats["is_connected"] = {"value": is_connected, "unit": "boolean"}
    components = G.connected_components(mode="weak")
    stats["num_components"] = {"value": len(components), "unit": "count"}
    logger.info(f"Graph connectivity: {is_connected}, Components: {len(components)}")

    # Density
    stats["density"] = {"value": G.density(), "unit": "ratio"}

    # Average degree
    degrees = G.degree()
    stats["avg_degree"] = {"value": np.mean(degrees), "unit": "connections"}
    stats["max_degree"] = {"value": max(degrees), "unit": "connections"}
    stats["min_degree"] = {"value": min(degrees), "unit": "connections"}
    logger.debug(
        f"Degree stats - Avg: {stats['avg_degree']['value']}, Max: {stats['max_degree']['value']}, Min: {stats['min_degree']['value']}"
    )

    # Centrality measures (for largest connected component if graph is disconnected)
    if is_connected:
        largest_cc = G
        logger.info(
            "Using entire graph for centrality calculations (graph is connected)"
        )
    else:
        largest_component = max(components, key=len)
        largest_cc = G.subgraph(largest_component)
        logger.info(
            f"Using largest component with {largest_cc.vcount()} nodes for centrality calculations"
        )

    # Degree centrality
    if Config.CALCULATE_DEGREE_CENTRALITY:
        logger.debug("Computing degree centrality...")
        degree_centrality = largest_cc.degree()
        max_possible_degree = largest_cc.vcount() - 1
        normalized_degree_centrality = (
            [d / max_possible_degree for d in degree_centrality]
            if max_possible_degree > 0
            else [0] * largest_cc.vcount()
        )
        stats["avg_degree_centrality"] = {
            "value": np.mean(normalized_degree_centrality),
            "unit": "normalized ratio",
        }
    else:
        logger.info("Skipping degree centrality calculation (disabled in config)")
        stats["avg_degree_centrality"] = {
            "value": "Calculation disabled",
            "unit": "n/a",
        }

    # Closeness centrality
    if Config.CALCULATE_CLOSENESS_CENTRALITY:
        logger.debug("Computing closeness centrality...")
        try:
            # Check if weight attribute exists, use it if available
            weight_attr = "weight" if "weight" in G.es.attributes() else None
            closeness_centrality = largest_cc.closeness(
                weights=weight_attr, normalized=True
            )
            stats["avg_closeness_centrality"] = {
                "value": np.mean(closeness_centrality),
                "unit": "normalized ratio",
            }
        except Exception as e:
            logger.warning(f"Failed to compute closeness centrality: {e}")
            stats["avg_closeness_centrality"] = {
                "value": "Could not compute",
                "unit": "n/a",
            }
    else:
        logger.info("Skipping closeness centrality calculation (disabled in config)")
        stats["avg_closeness_centrality"] = {
            "value": "Calculation disabled",
            "unit": "n/a",
        }

    # Betweenness centrality - compute regardless of size
    if Config.CALCULATE_BETWEENNESS_CENTRALITY:
        logger.debug("Computing betweenness centrality...")
        try:
            # Check if weight attribute exists, use it if available
            weight_attr = "weight" if "weight" in G.es.attributes() else None
            betweenness_centrality = largest_cc.betweenness(weights=weight_attr)
            # Normalize manually since igraph doesn't support normalized parameter consistently
            n = largest_cc.vcount()
            normalization_factor = 2.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
            normalized_betweenness = [
                b * normalization_factor for b in betweenness_centrality
            ]

            stats["avg_betweenness_centrality"] = {
                "value": np.mean(normalized_betweenness),
                "unit": "normalized ratio",
            }
        except Exception as e:
            logger.warning(f"Failed to compute betweenness centrality: {e}")
            stats["avg_betweenness_centrality"] = {
                "value": "Could not compute",
                "unit": "n/a",
            }
    else:
        logger.info("Skipping betweenness centrality calculation (disabled in config)")
        stats["avg_betweenness_centrality"] = {
            "value": "Calculation disabled",
            "unit": "n/a",
        }

    # Eigenvector centrality (only for connected graphs)
    if Config.CALCULATE_EIGENVECTOR_CENTRALITY:
        if largest_cc.is_connected(mode="weak"):
            logger.debug("Computing eigenvector centrality...")
            try:
                # Check if weight attribute exists, use it if available
                weight_attr = "weight" if "weight" in G.es.attributes() else None
                eigenvector_centrality = largest_cc.eigenvector_centrality(
                    weights=weight_attr
                )
                stats["avg_eigenvector_centrality"] = {
                    "value": np.mean(eigenvector_centrality),
                    "unit": "normalized ratio",
                }
            except Exception as e:
                logger.warning(f"Failed to compute eigenvector centrality: {e}")
                stats["avg_eigenvector_centrality"] = {
                    "value": "Could not compute",
                    "unit": "n/a",
                }
        else:
            logger.info("Skipping eigenvector centrality (largest component not connected)")
            stats["avg_eigenvector_centrality"] = {
                "value": "Graph not connected",
                "unit": "n/a",
            }
    else:
        logger.info("Skipping eigenvector centrality calculation (disabled in config)")
        stats["avg_eigenvector_centrality"] = {
            "value": "Calculation disabled",
            "unit": "n/a",
        }

    # Straightness centrality and global straightness - compute regardless of size
    if Config.CALCULATE_STRAIGHTNESS_CENTRALITY or Config.CALCULATE_GLOBAL_STRAIGHTNESS:
        logger.debug("Computing straightness measures...")
        try:
            # Check if nodes have coordinate attributes
            if "x" in largest_cc.vs.attributes() and "y" in largest_cc.vs.attributes():
                # Check if weight attribute exists, use it if available
                weight_attr = "weight" if "weight" in G.es.attributes() else None
                
                if Config.CALCULATE_STRAIGHTNESS_CENTRALITY:
                    straightness_values = straightness_centrality(
                        largest_cc, weight=weight_attr
                    )
                    stats["avg_straightness_centrality"] = {
                        "value": np.mean(straightness_values),
                        "unit": "ratio",
                    }
                    stats["max_straightness_centrality"] = {
                        "value": max(straightness_values),
                        "unit": "ratio",
                    }
                    stats["min_straightness_centrality"] = {
                        "value": min(straightness_values),
                        "unit": "ratio",
                    }
                else:
                    logger.info("Skipping straightness centrality calculation (disabled in config)")
                    stats["avg_straightness_centrality"] = {
                        "value": "Calculation disabled",
                        "unit": "n/a",
                    }

                # Global straightness
                if Config.CALCULATE_GLOBAL_STRAIGHTNESS:
                    global_straightness = graph_straightness(largest_cc, weight=weight_attr)
                    stats["global_straightness"] = {
                        "value": global_straightness,
                        "unit": "ratio",
                    }
                else:
                    logger.info("Skipping global straightness calculation (disabled in config)")
                    stats["global_straightness"] = {
                        "value": "Calculation disabled",
                        "unit": "n/a",
                    }

            else:
                logger.warning("No coordinate data found for straightness centrality")
                stats["avg_straightness_centrality"] = {
                    "value": "No coordinate data",
                    "unit": "n/a",
                }
                stats["global_straightness"] = {
                    "value": "No coordinate data",
                    "unit": "n/a",
                }
        except Exception as e:
            logger.warning(f"Failed to compute straightness measures: {e}")
            stats["avg_straightness_centrality"] = {
                "value": "Could not compute",
                "unit": "n/a",
            }
            stats["global_straightness"] = {"value": "Could not compute", "unit": "n/a"}
    else:
        logger.info("Skipping all straightness calculations (disabled in config)")
        stats["avg_straightness_centrality"] = {
            "value": "Calculation disabled",
            "unit": "n/a",
        }
        stats["global_straightness"] = {
            "value": "Calculation disabled",
            "unit": "n/a",
        }

    # Farness centrality (if available)
    if "farness" in G.vs.attributes():
        logger.debug("Processing farness centrality statistics...")
        farness_values = G.vs["farness"]
        stats["avg_farness"] = {"value": np.mean(farness_values), "unit": "meters"}
        stats["max_farness"] = {"value": max(farness_values), "unit": "meters"}
        stats["min_farness"] = {"value": min(farness_values), "unit": "meters"}
    else:
        logger.debug("No farness centrality data found")

    # Normalized farness centrality (if available)
    if "norm_farness" in G.vs.attributes():
        logger.debug("Processing normalized farness centrality statistics...")
        norm_farness_values = G.vs["norm_farness"]
        stats["avg_norm_farness"] = {
            "value": np.mean(norm_farness_values),
            "unit": "meters/node",
        }
        stats["max_norm_farness"] = {
            "value": max(norm_farness_values),
            "unit": "meters/node",
        }
        stats["min_norm_farness"] = {
            "value": min(norm_farness_values),
            "unit": "meters/node",
        }

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
    lines.append(f"\n{'=' * 50}")
    lines.append(f"{title:^50}")
    lines.append(f"{'=' * 50}")

    # Basic Statistics
    lines.append("\nBasic Statistics:")
    lines.append(
        f"  Nodes: {stats['num_nodes']['value']:,} {stats['num_nodes']['unit']}"
    )
    lines.append(
        f"  Edges: {stats['num_edges']['value']:,} {stats['num_edges']['unit']}"
    )
    lines.append(
        f"  Connected: {stats['is_connected']['value']} ({stats['is_connected']['unit']})"
    )
    lines.append(
        f"  Components: {stats['num_components']['value']} {stats['num_components']['unit']}"
    )
    lines.append(
        f"  Density: {stats['density']['value']:.6f} ({stats['density']['unit']})"
    )

    # Spatial Statistics
    if "convex_hull_area_km2" in stats:
        lines.append("\nSpatial Statistics:")
        if isinstance(stats["convex_hull_area_km2"]["value"], str):
            lines.append(
                f"  Convex Hull Area: {stats['convex_hull_area_km2']['value']} ({stats['convex_hull_area_km2']['unit']})"
            )
        else:
            lines.append(
                f"  Convex Hull Area: {stats['convex_hull_area_km2']['value']:,.2f} {stats['convex_hull_area_km2']['unit']}"
            )
            lines.append(
                f"    ({stats['convex_hull_area_m2']['value']:,.0f} {stats['convex_hull_area_m2']['unit']})"
            )

        # Voronoi statistics
        if "avg_voronoi_area_km2" in stats:
            if isinstance(stats["avg_voronoi_area_km2"]["value"], str):
                lines.append(
                    f"  Avg Voronoi Area: {stats['avg_voronoi_area_km2']['value']} ({stats['avg_voronoi_area_km2']['unit']})"
                )
            else:
                lines.append(
                    f"  Avg Voronoi Area: {stats['avg_voronoi_area_km2']['value']:,.2f} {stats['avg_voronoi_area_km2']['unit']}"
                )
                if "max_voronoi_area_m2" in stats:
                    lines.append(
                        f"    Maximum: {stats['max_voronoi_area_m2']['value']:,.0f} {stats['max_voronoi_area_m2']['unit']}"
                    )
                    lines.append(
                        f"    Minimum: {stats['min_voronoi_area_m2']['value']:,.0f} {stats['min_voronoi_area_m2']['unit']}"
                    )
                if "num_valid_voronoi" in stats:
                    lines.append(
                        f"    Valid polygons: {stats['num_valid_voronoi']['value']} {stats['num_valid_voronoi']['unit']}"
                    )

    # Degree Statistics
    lines.append("\nDegree Statistics:")
    lines.append(
        f"  Average: {stats['avg_degree']['value']:.2f} {stats['avg_degree']['unit']}"
    )
    lines.append(
        f"  Maximum: {stats['max_degree']['value']} {stats['max_degree']['unit']}"
    )
    lines.append(
        f"  Minimum: {stats['min_degree']['value']} {stats['min_degree']['unit']}"
    )

    # Centrality Measures
    lines.append("\nCentrality Measures:")
    lines.append(
        f"  Avg Degree Centrality: {stats['avg_degree_centrality']['value']:.6f} ({stats['avg_degree_centrality']['unit']})"
    )
    lines.append(
        f"  Avg Closeness Centrality: {stats['avg_closeness_centrality']['value']:.6f} ({stats['avg_closeness_centrality']['unit']})"
    )

    if isinstance(stats["avg_betweenness_centrality"]["value"], str):
        lines.append(
            f"  Avg Betweenness Centrality: {stats['avg_betweenness_centrality']['value']} ({stats['avg_betweenness_centrality']['unit']})"
        )
    else:
        lines.append(
            f"  Avg Betweenness Centrality: {stats['avg_betweenness_centrality']['value']:.6f} ({stats['avg_betweenness_centrality']['unit']})"
        )

    if isinstance(stats["avg_eigenvector_centrality"]["value"], str):
        lines.append(
            f"  Avg Eigenvector Centrality: {stats['avg_eigenvector_centrality']['value']} ({stats['avg_eigenvector_centrality']['unit']})"
        )
    else:
        lines.append(
            f"  Avg Eigenvector Centrality: {stats['avg_eigenvector_centrality']['value']:.6f} ({stats['avg_eigenvector_centrality']['unit']})"
        )

    # Straightness Measures
    if "avg_straightness_centrality" in stats:
        lines.append("\nStraightness Measures:")
        if isinstance(stats["avg_straightness_centrality"]["value"], str):
            lines.append(
                f"  Avg Straightness Centrality: {stats['avg_straightness_centrality']['value']} ({stats['avg_straightness_centrality']['unit']})"
            )
        else:
            lines.append(
                f"  Avg Straightness Centrality: {stats['avg_straightness_centrality']['value']:.6f} ({stats['avg_straightness_centrality']['unit']})"
            )
            if "max_straightness_centrality" in stats:
                lines.append(
                    f"    Maximum: {stats['max_straightness_centrality']['value']:.6f} {stats['max_straightness_centrality']['unit']}"
                )
                lines.append(
                    f"    Minimum: {stats['min_straightness_centrality']['value']:.6f} {stats['min_straightness_centrality']['unit']}"
                )

        if "global_straightness" in stats:
            if isinstance(stats["global_straightness"]["value"], str):
                lines.append(
                    f"  Global Straightness: {stats['global_straightness']['value']} ({stats['global_straightness']['unit']})"
                )
            else:
                lines.append(
                    f"  Global Straightness: {stats['global_straightness']['value']:.6f} ({stats['global_straightness']['unit']})"
                )

    # Farness Statistics (if available)
    if "avg_farness" in stats:
        lines.append("\nFarness Centrality:")
        lines.append(
            f"  Average: {stats['avg_farness']['value']:,.2f} {stats['avg_farness']['unit']}"
        )
        lines.append(
            f"  Maximum: {stats['max_farness']['value']:,.2f} {stats['max_farness']['unit']}"
        )
        lines.append(
            f"  Minimum: {stats['min_farness']['value']:,.2f} {stats['min_farness']['unit']}"
        )

    # Normalized Farness Statistics (if available)
    if "avg_norm_farness" in stats:
        lines.append("\nNormalized Farness Centrality:")
        lines.append(
            f"  Average: {stats['avg_norm_farness']['value']:.6f} {stats['avg_norm_farness']['unit']}"
        )
        lines.append(
            f"  Maximum: {stats['max_norm_farness']['value']:.6f} {stats['max_norm_farness']['unit']}"
        )
        lines.append(
            f"  Minimum: {stats['min_norm_farness']['value']:.6f} {stats['min_norm_farness']['unit']}"
        )

    lines.append(f"{'=' * 50}\n")

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
    lines.append(f"\n{'=' * 70}")
    lines.append(f"{'GRAPH COMPARISON':^70}")
    lines.append(f"{'=' * 70}")
    lines.append(f"{title1:<35} vs {title2:>33}")
    lines.append(f"{'-' * 70}")

    # Helper function to format values
    def format_value(stat_dict, precision=None):
        value = stat_dict["value"]
        unit = stat_dict["unit"]

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
    basic_stats = [
        "num_nodes",
        "num_edges",
        "is_connected",
        "num_components",
        "density",
    ]

    for stat in basic_stats:
        if stat in stats1 and stat in stats2:
            val1_str = format_value(stats1[stat], 2 if stat == "density" else None)
            val2_str = format_value(stats2[stat], 2 if stat == "density" else None)
            change = calc_change(stats1[stat]["value"], stats2[stat]["value"])

            stat_name = stat.replace("_", " ").title()
            lines.append(f"  {stat_name}:")
            lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")

    # Spatial Statistics
    if "convex_hull_area_km2" in stats1 and "convex_hull_area_km2" in stats2:
        lines.append("\nSpatial Statistics:")
        val1_str = format_value(stats1["convex_hull_area_km2"], 2)
        val2_str = format_value(stats2["convex_hull_area_km2"], 2)
        change = calc_change(
            stats1["convex_hull_area_km2"]["value"],
            stats2["convex_hull_area_km2"]["value"],
        )

        lines.append("  Convex Hull Area:")
        lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")

        # Voronoi area comparison
        if "avg_voronoi_area_km2" in stats1 and "avg_voronoi_area_km2" in stats2:
            val1_str = format_value(stats1["avg_voronoi_area_km2"], 2)
            val2_str = format_value(stats2["avg_voronoi_area_km2"], 2)
            change = calc_change(
                stats1["avg_voronoi_area_km2"]["value"],
                stats2["avg_voronoi_area_km2"]["value"],
            )

            lines.append("  Avg Voronoi Area:")
            lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")

    # Degree Statistics
    lines.append("\nDegree Statistics:")
    degree_stats = ["avg_degree", "max_degree", "min_degree"]

    for stat in degree_stats:
        if stat in stats1 and stat in stats2:
            val1_str = format_value(stats1[stat], 2)
            val2_str = format_value(stats2[stat], 2)
            change = calc_change(stats1[stat]["value"], stats2[stat]["value"])

            stat_name = stat.replace("_", " ").title()
            lines.append(f"  {stat_name}:")
            lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")

    # Centrality Measures
    lines.append("\nCentrality Measures:")
    centrality_stats = [
        "avg_degree_centrality",
        "avg_closeness_centrality",
        "avg_betweenness_centrality",
        "avg_eigenvector_centrality",
        "avg_straightness_centrality",
    ]

    for stat in centrality_stats:
        if stat in stats1 and stat in stats2:
            val1_str = format_value(stats1[stat], 6)
            val2_str = format_value(stats2[stat], 6)
            change = calc_change(stats1[stat]["value"], stats2[stat]["value"])

            stat_name = stat.replace("_", " ").replace("avg ", "").title()
            lines.append(f"  {stat_name}:")
            lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")

    # Global Straightness
    if "global_straightness" in stats1 and "global_straightness" in stats2:
        val1_str = format_value(stats1["global_straightness"], 6)
        val2_str = format_value(stats2["global_straightness"], 6)
        change = calc_change(
            stats1["global_straightness"]["value"],
            stats2["global_straightness"]["value"],
        )

        lines.append("  Global Straightness:")
        lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")

    # Straightness Statistics (if available)
    straightness_stats = [
        "avg_straightness_centrality",
        "max_straightness_centrality",
        "min_straightness_centrality",
    ]
    if any(stat in stats1 and stat in stats2 for stat in straightness_stats):
        lines.append("\nStraightness Centrality:")

        for stat in straightness_stats:
            if stat in stats1 and stat in stats2:
                val1_str = format_value(stats1[stat], 6)
                val2_str = format_value(stats2[stat], 6)
                change = calc_change(stats1[stat]["value"], stats2[stat]["value"])

                stat_name = stat.replace("_", " ").replace("avg ", "").title()
                lines.append(f"  {stat_name}:")
                lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")

    # Farness Statistics (if available)
    farness_stats = ["avg_farness", "max_farness", "min_farness"]
    if any(stat in stats1 and stat in stats2 for stat in farness_stats):
        lines.append("\nFarness Centrality:")

        for stat in farness_stats:
            if stat in stats1 and stat in stats2:
                val1_str = format_value(stats1[stat], 2)
                val2_str = format_value(stats2[stat], 2)
                change = calc_change(stats1[stat]["value"], stats2[stat]["value"])

                stat_name = stat.replace("_", " ").replace("avg ", "").title()
                lines.append(f"  {stat_name}:")
                lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")

    # Normalized Farness Statistics (if available)
    norm_farness_stats = ["avg_norm_farness", "max_norm_farness", "min_norm_farness"]
    if any(stat in stats1 and stat in stats2 for stat in norm_farness_stats):
        lines.append("\nNormalized Farness Centrality:")

        for stat in norm_farness_stats:
            if stat in stats1 and stat in stats2:
                val1_str = format_value(stats1[stat], 6)
                val2_str = format_value(stats2[stat], 6)
                change = calc_change(stats1[stat]["value"], stats2[stat]["value"])

                stat_name = (
                    stat.replace("_", " ")
                    .replace("avg ", "")
                    .replace("norm ", "")
                    .title()
                )
                lines.append(f"  {stat_name}:")
                lines.append(f"    {val1_str:<30} → {val2_str:<30} ({change})")

    lines.append(f"{'=' * 70}\n")

    return "\n".join(lines)
