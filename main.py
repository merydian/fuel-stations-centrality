"""
Enhanced main module with improved error handling and code structure.
"""

import logging
import time
import sys
import random
from datetime import datetime


from config import Config
from centrality import get_knn_dists
import stats_cpp  # Add C++ stats import
from utils import (
    save_graph_to_geopackage,
    remove_edges_far_from_stations_graph,
    get_gas_stations_from_graph,
    setup_logging,
    log_step_start,
    log_step_end,
    find_stations_in_road_network,
    convert_networkx_to_igraph,
    save_removed_stations_to_geopackage,
    save_stations_to_geopackage,
    make_graph_from_stations,
    process_fuel_stations,
    remove_stations_from_road_network
)
import osmnx as ox
import networkx as nx

logger = logging.getLogger(__name__)


def main():
    """Main function for fuel station centrality analysis."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Configuration validation and setup
    Config.ensure_directories()
    Config.validate_config()

    # Get PBF file stem for output filenames
    pbf_stem = Config.LOCAL_PBF_PATH.stem if hasattr(Config, "LOCAL_PBF_PATH") else "unknown"

    # Log analysis start
    time.time()
    logger.info("=" * 80)
    logger.info(
        f"FUEL STATION CENTRALITY ANALYSIS STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 80)
    logger.info("Analysis Configuration:")
    logger.info(f"  • Location: {Config.PLACE}")
    logger.info(f"  • Max edge distance: {Config.MAX_DISTANCE:,} meters")
    logger.info(f"  • Stations to remove: {Config.N_REMOVE}")
    logger.info(f"  • k-NN parameter: {Config.K_NN}")
    logger.info(f"  • Removal criteria: {Config.REMOVAL_KIND}")
    logger.info(f"  • Max stations limit: {Config.MAX_STATIONS if Config.MAX_STATIONS else 'No limit'}")
    logger.info(f"  • Log level: {Config.LOG_LEVEL}")
    logger.info("")
    logger.info("Centrality Calculations:")
    logger.info(f"  • Degree centrality: {'Enabled' if Config.CALCULATE_DEGREE_CENTRALITY else 'Disabled'}")
    logger.info(f"  • Closeness centrality: {'Enabled' if Config.CALCULATE_CLOSENESS_CENTRALITY else 'Disabled'}")
    logger.info(f"  • Betweenness centrality: {'Enabled' if Config.CALCULATE_BETWEENNESS_CENTRALITY else 'Disabled'}")
    logger.info(f"  • Eigenvector centrality: {'Enabled' if Config.CALCULATE_EIGENVECTOR_CENTRALITY else 'Disabled'}")
    logger.info(f"  • Straightness centrality: {'Enabled' if Config.CALCULATE_STRAIGHTNESS_CENTRALITY else 'Disabled'}")
    logger.info(f"  • Global straightness: {'Enabled' if Config.CALCULATE_GLOBAL_STRAIGHTNESS else 'Disabled'}")
    logger.info("")

    try:
        # Step 0: Download road network
        step_start = log_step_start(
            "0", "Downloading/Loading road network from OpenStreetMap"
        )

        road_filepath = Config.get_road_filepath()
        logger.info(f"Loading road network from {road_filepath}")
        G_road = ox.load_graphml(road_filepath)
        G_road.remove_edges_from(nx.selfloop_edges(G_road))

        logger.info(
            f"✓ Cached road network loaded: {len(G_road.nodes):,} nodes, {len(G_road.edges):,} edges"
        )
        log_step_end(step_start, "0", "Road network acquisition")

        # Step 1: Load fuel stations
        step_start = log_step_start(
            "1", "Extracting fuel stations from road network area"
        )
        stations = get_gas_stations_from_graph(G_road)
        
        # Process and potentially limit stations
        stations = process_fuel_stations(stations)
        
        # Log clustering information if available
        if 'cluster_id' in stations.columns:
            total_original = stations['stations_in_cluster'].sum()
            clustered_stations = len(stations[stations['stations_in_cluster'] > 1])
            reduction_count = total_original - len(stations)
            reduction_percent = (reduction_count / total_original) * 100 if total_original > 0 else 0
            
            logger.info(f"✓ Station clustering applied:")
            logger.info(f"  • Original stations extracted: {total_original:,}")
            logger.info(f"  • Final clustered stations: {len(stations):,}")
            logger.info(f"  • Stations combined: {reduction_count:,}")
            logger.info(f"  • Multi-station clusters: {clustered_stations}")
            logger.info(f"  • Clustering reduction: {reduction_percent:.1f}%")
        else:
            logger.info(f"✓ No clustering applied: {len(stations):,} stations")
            
        # Log station limitation if applied
        if Config.MAX_STATIONS and 'cluster_id' in stations.columns:
            original_after_clustering = len(stations)
            if original_after_clustering > Config.MAX_STATIONS:
                logger.info(f"  • Station limit applied: {original_after_clustering} → {Config.MAX_STATIONS} stations")
            
        log_step_end(step_start, "1", "Fuel station extraction")

        # Step 1.5: Save all gas stations to GeoPackage
        step_start = log_step_start("1.5", "Saving all gas stations to GeoPackage")
        save_stations_to_geopackage(
            stations, out_file=f"all_gas_stations_{Config.get_road_filename()}.gpkg"
        )
        
        log_step_end(step_start, "1.5", "Gas stations save")

        # Step 2: Map stations to road network (needed for both methods)
        step_start = log_step_start("2", "Mapping stations to road network")
        station_to_node_mapping = find_stations_in_road_network(G_road, stations)
        log_step_end(step_start, "2", "Station mapping")

        # Step 3: Create stations connectivity graph for analysis
        step_start = log_step_start(
            "3", "Building station connectivity graph for k-NN analysis"
        )

        logger.info("  Using road network for driving distances...")
        G_stations = make_graph_from_stations(
            stations,
            use_ors=False,
            G_road=G_road,
            station_to_node_mapping=station_to_node_mapping,
        )

        logger.info(
            f"✓ Station graph created: {G_stations.vcount()} nodes, {G_stations.ecount()} edges"
        )
        log_step_end(step_start, "3", "Station graph creation")

        # Step 4: Analyze stations to find removal candidates
        step_start = log_step_start(
            "4", "Computing k-NN distances for station analysis"
        )
        logger.info(
            "  Computing farness centrality and k-NN distances on station graph..."
        )
        G_stations, knn_dist = get_knn_dists(
            G_stations, weight="weight", n=Config.K_NN
        )
        logger.info(f"✓ Station analysis complete: {len(knn_dist)} stations analyzed")
        log_step_end(step_start, "4", "Station analysis")

        # Step 5: Convert road network to igraph for centrality analysis
        step_start = log_step_start(
            "5", "Converting road network for centrality analysis"
        )
        G_road_ig = convert_networkx_to_igraph(G_road)

        # Step 6: Get baseline statistics on full road network
        step_start = log_step_start("6", "Computing baseline road network statistics")
        old_stats = stats_cpp.GraphStatsCalculator.get_graph_stats(G_road_ig)
        logger.info("✓ Baseline road network statistics computed")
        log_step_end(step_start, "6", "Baseline statistics")

        # Step 7: Save baseline road network
        step_start = log_step_start("7", "Saving baseline road network data")
        save_graph_to_geopackage(
            G_road_ig,
            out_file=f"road_network_baseline_{pbf_stem}.gpkg",
        )
        logger.info("✓ Baseline road network saved to GeoPackage")
        log_step_end(step_start, "7", "Baseline save")

        # Step 8: Identify stations for removal based on k-NN analysis
        step_start = log_step_start(
            "8", f"Identifying stations for removal based on {Config.REMOVAL_KIND}"
        )
        logger.info(
            f"  Selecting top {Config.N_REMOVE} stations for removal based on {Config.REMOVAL_KIND} values..."
        )

        # Get stations with highest knn_dist values for removal
        sorted_stations = sorted(knn_dist.items(), key=lambda x: x[1], reverse=True)
        stations_to_remove = [
            station_id for station_id, _ in sorted_stations[: Config.N_REMOVE]
        ]

        logger.info(
            f"✓ Identified {len(stations_to_remove)} stations for removal: {stations_to_remove}"
        )

        # Save smart-removed stations to GeoPackage with k-NN distance data
        save_removed_stations_to_geopackage(
            stations,
            stations_to_remove,
            out_file=f"removed_stations_smart_{pbf_stem}.gpkg",
            removal_type="smart_knn",
            knn_dist=knn_dist,
        )

        log_step_end(step_start, "8", "Station identification")

        # Step 9: Remove stations from road network (smart removal)
        step_start = log_step_start(
            "9", "Removing identified stations from road network"
        )
        G_road_filtered = remove_stations_from_road_network(G_road, station_to_node_mapping, stations_to_remove)
        G_road_ig = convert_networkx_to_igraph(G_road_filtered)
        
        # Clean up edges far from stations before baseline analysis
        logger.info(
            "  Cleaning up edges far from gas stations from baseline road network..."
        )
        print(G_road_ig.es.attributes())
        G_road_ig, edges_removed = remove_edges_far_from_stations_graph(
            G_road_ig, stations, Config.MAX_DISTANCE, station_to_node_mapping
        )
        
        # Check if any edges were removed - exit if none
        if edges_removed == 0:
            logger.error("❌ ANALYSIS TERMINATED: No edges were removed from the road network.")
            logger.error("This indicates that all road network edges are within the specified distance")
            logger.error(f"of gas stations ({Config.MAX_DISTANCE:,}m). The analysis would not be meaningful.")
            logger.error("")
            logger.error("Possible solutions:")
            logger.error(f"  • Reduce MAX_DISTANCE (currently {Config.MAX_DISTANCE:,}m) in config.py")
            logger.error("  • Use a different region with sparser gas station coverage")
            logger.error("  • Increase the road network size to include more remote areas")
            sys.exit(1)

        # No base convex hull needed
        base_convex_hull = None
        log_step_end(step_start, "9", "Smart station removal")

        # Step 10: Create random comparison by removing random stations
        step_start = log_step_start("10", "Creating random comparison road network")
        # Select random stations for removal
        random.seed(Config.RANDOM_SEED)
        all_station_indices = list(station_to_node_mapping.keys())
        random_stations_to_remove = random.sample(
            all_station_indices, min(Config.N_REMOVE, len(all_station_indices))
        )
        assert random_stations_to_remove != stations_to_remove

        # Save random-removed stations to GeoPackage with k-NN distance data
        save_removed_stations_to_geopackage(
            stations,
            random_stations_to_remove,
            out_file=f"removed_stations_random_{pbf_stem}.gpkg",
            removal_type="random",
            knn_dist=knn_dist,
        )

        # Convert graph for random comparison
        G_road_random_filtered = remove_stations_from_road_network(G_road, station_to_node_mapping, random_stations_to_remove)

        G_road_ig_random = convert_networkx_to_igraph(G_road_random_filtered)

        logger.info(
            f"✓ Random-filtered road network: {G_road_ig_random.vcount()} nodes, {G_road_ig_random.ecount()} edges"
        )
        log_step_end(step_start, "10", "Random station removal")

        # Step 11: Compute centrality measures on filtered road networks
        step_start = log_step_start(
            "11", "Computing centrality measures on filtered road networks"
        )

        logger.info("  Computing statistics for smart-filtered road network...")
        smart_stats = stats_cpp.GraphStatsCalculator.get_graph_stats(G_road_ig, base_convex_hull=base_convex_hull)

        logger.info("  Computing statistics for random-filtered road network...")
        random_stats = stats_cpp.GraphStatsCalculator.get_graph_stats(G_road_ig_random, base_convex_hull=base_convex_hull)

        logger.info("✓ Centrality measures computed for both filtered networks")
        log_step_end(step_start, "11", "Filtered network analysis")

        # Step 12: Save filtered road networks
        step_start = log_step_start("12", "Saving filtered road networks")
        save_graph_to_geopackage(
            G_road_ig,
            out_file=f"road_network_smart_filtered_{pbf_stem}.gpkg",
        )
        save_graph_to_geopackage(
            G_road_ig_random,
            out_file=f"road_network_random_filtered_{pbf_stem}.gpkg",
        )
        logger.info("✓ Filtered road networks saved")
        log_step_end(step_start, "12", "Filtered network save")

        # Step 13: Compare and log results
        step_start = log_step_start(
            "13", "Comparing baseline vs filtered road networks"
        )

        logger.info("=" * 60)
        logger.info("                    ANALYSIS COMPLETE")
        logger.info("=" * 60)

        # Helper function to calculate percentage change
        def calc_percentage_change(old_val, new_val):
            """Calculate percentage change between two values."""
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if old_val == 0:
                    return "N/A" if new_val == 0 else "∞"
                return f"{((new_val - old_val) / old_val) * 100:+.2f}%"
            return "N/A"

        # Log baseline road network stats
        logger.info("📊 BASELINE ROAD NETWORK STATISTICS:")
        logger.info(f"   • Nodes: {old_stats['num_nodes'].value_num:,}")
        logger.info(f"   • Edges: {old_stats['num_edges'].value_num:,}")
        logger.info(f"   • Density: {old_stats['density'].value_num:.6f}")
        logger.info(
            f"   • Mean degree centrality: {old_stats.get('avg_degree_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_str if not old_stats.get('avg_degree_centrality', stats_cpp.StatValue('N/A', 'n/a')).is_numeric else old_stats.get('avg_degree_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_num}"
        )
        logger.info(
            f"   • Mean closeness centrality: {old_stats.get('avg_closeness_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_str if not old_stats.get('avg_closeness_centrality', stats_cpp.StatValue('N/A', 'n/a')).is_numeric else old_stats.get('avg_closeness_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_num}"
        )
        logger.info(
            f"   • Mean betweenness centrality: {old_stats.get('avg_betweenness_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_str if not old_stats.get('avg_betweenness_centrality', stats_cpp.StatValue('N/A', 'n/a')).is_numeric else old_stats.get('avg_betweenness_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_num}"
        )
        logger.info(
            f"   • Mean eigenvector centrality: {old_stats.get('avg_eigenvector_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_str if not old_stats.get('avg_eigenvector_centrality', stats_cpp.StatValue('N/A', 'n/a')).is_numeric else old_stats.get('avg_eigenvector_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_num}"
        )
        logger.info(
            f"   • Mean straightness centrality: {old_stats.get('avg_straightness_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_str if not old_stats.get('avg_straightness_centrality', stats_cpp.StatValue('N/A', 'n/a')).is_numeric else old_stats.get('avg_straightness_centrality', stats_cpp.StatValue('N/A', 'n/a')).value_num}"
        )
        logger.info(
            f"   • Global straightness: {old_stats.get('global_straightness', stats_cpp.StatValue('N/A', 'n/a')).value_str if not old_stats.get('global_straightness', stats_cpp.StatValue('N/A', 'n/a')).is_numeric else old_stats.get('global_straightness', stats_cpp.StatValue('N/A', 'n/a')).value_num}"
        )

        # Log smart-filtered road network stats
        logger.info("")
        logger.info(
            f"🎯 SMART-FILTERED ROAD NETWORK STATISTICS (removed {Config.N_REMOVE} high k-NN stations):"
        )
        logger.info(
            f"   • Nodes: {smart_stats['num_nodes'].value_num:,} (Δ: {smart_stats['num_nodes'].value_num - old_stats['num_nodes'].value_num:+,})"
        )
        logger.info(
            f"   • Edges: {smart_stats['num_edges'].value_num:,} (Δ: {smart_stats['num_edges'].value_num - old_stats['num_edges'].value_num:+,})"
        )
        logger.info(
            f"   • Density: {smart_stats['density'].value_num:.6f} (Δ: {smart_stats['density'].value_num - old_stats['density'].value_num:+.6f})"
        )

        # Centrality measures with percentage changes
        smart_deg_cent = smart_stats.get("avg_degree_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if smart_stats.get("avg_degree_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        old_deg_cent = old_stats.get("avg_degree_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if old_stats.get("avg_degree_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        deg_cent_change = calc_percentage_change(old_deg_cent, smart_deg_cent)
        logger.info(
            f"   • Mean degree centrality: {smart_deg_cent} ({deg_cent_change})"
        )

        smart_close_cent = smart_stats.get("avg_closeness_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if smart_stats.get("avg_closeness_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        old_close_cent = old_stats.get("avg_closeness_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if old_stats.get("avg_closeness_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        close_cent_change = calc_percentage_change(old_close_cent, smart_close_cent)
        logger.info(
            f"   • Mean closeness centrality: {smart_close_cent} ({close_cent_change})"
        )

        smart_between_cent = smart_stats.get("avg_betweenness_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if smart_stats.get("avg_betweenness_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        old_between_cent = old_stats.get("avg_betweenness_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if old_stats.get("avg_betweenness_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        between_cent_change = calc_percentage_change(
            old_between_cent, smart_between_cent
        )
        logger.info(
            f"   • Mean betweenness centrality: {smart_between_cent} ({between_cent_change})"
        )

        smart_eigen_cent = smart_stats.get("avg_eigenvector_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if smart_stats.get("avg_eigenvector_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        old_eigen_cent = old_stats.get("avg_eigenvector_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if old_stats.get("avg_eigenvector_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        eigen_cent_change = calc_percentage_change(old_eigen_cent, smart_eigen_cent)
        logger.info(
            f"   • Mean eigenvector centrality: {smart_eigen_cent} ({eigen_cent_change})"
        )

        smart_straight_cent = smart_stats.get("avg_straightness_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if smart_stats.get("avg_straightness_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        old_straight_cent = old_stats.get("avg_straightness_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if old_stats.get("avg_straightness_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        straight_cent_change = calc_percentage_change(
            old_straight_cent, smart_straight_cent
        )
        logger.info(
            f"   • Mean straightness centrality: {smart_straight_cent} ({straight_cent_change})"
        )

        smart_global_straight = smart_stats.get("global_straightness", stats_cpp.StatValue("N/A", "n/a")).value_num if smart_stats.get("global_straightness", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        old_global_straight = old_stats.get("global_straightness", stats_cpp.StatValue("N/A", "n/a")).value_num if old_stats.get("global_straightness", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        global_straight_change = calc_percentage_change(
            old_global_straight, smart_global_straight
        )
        logger.info(
            f"   • Global straightness: {smart_global_straight} ({global_straight_change})"
        )

        # Log random-filtered road network stats
        logger.info("")
        logger.info(
            f"🎲 RANDOM-FILTERED ROAD NETWORK STATISTICS (removed {Config.N_REMOVE} random stations):"
        )
        logger.info(
            f"   • Nodes: {random_stats['num_nodes'].value_num:,} (Δ: {random_stats['num_nodes'].value_num - old_stats['num_nodes'].value_num:+,})"
        )
        logger.info(
            f"   • Edges: {random_stats['num_edges'].value_num:,} (Δ: {random_stats['num_edges'].value_num - old_stats['num_edges'].value_num:+,})"
        )
        logger.info(
            f"   • Density: {random_stats['density'].value_num:.6f} (Δ: {random_stats['density'].value_num - old_stats['density'].value_num:+.6f})"
        )

        # Centrality measures with percentage changes for random
        random_deg_cent = random_stats.get("avg_degree_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if random_stats.get("avg_degree_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        random_deg_cent_change = calc_percentage_change(old_deg_cent, random_deg_cent)
        logger.info(
            f"   • Mean degree centrality: {random_deg_cent} ({random_deg_cent_change})"
        )

        random_close_cent = random_stats.get("avg_closeness_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if random_stats.get("avg_closeness_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        random_close_cent_change = calc_percentage_change(
            old_close_cent, random_close_cent
        )
        logger.info(
            f"   • Mean closeness centrality: {random_close_cent} ({random_close_cent_change})"
        )

        random_between_cent = random_stats.get("avg_betweenness_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if random_stats.get("avg_betweenness_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        random_between_cent_change = calc_percentage_change(
            old_between_cent, random_between_cent
        )
        logger.info(
            f"   • Mean betweenness centrality: {random_between_cent} ({random_between_cent_change})"
        )

        random_eigen_cent = random_stats.get("avg_eigenvector_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if random_stats.get("avg_eigenvector_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        random_eigen_cent_change = calc_percentage_change(
            old_eigen_cent, random_eigen_cent
        )
        logger.info(
            f"   • Mean eigenvector centrality: {random_eigen_cent} ({random_eigen_cent_change})"
        )

        random_straight_cent = random_stats.get("avg_straightness_centrality", stats_cpp.StatValue("N/A", "n/a")).value_num if random_stats.get("avg_straightness_centrality", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        random_straight_cent_change = calc_percentage_change(
            old_straight_cent, random_straight_cent
        )
        logger.info(
            f"   • Mean straightness centrality: {random_straight_cent} ({random_straight_cent_change})"
        )

        random_global_straight = random_stats.get("global_straightness", stats_cpp.StatValue("N/A", "n/a")).value_num if random_stats.get("global_straightness", stats_cpp.StatValue("N/A", "n/a")).is_numeric else "N/A"
        random_global_straight_change = calc_percentage_change(
            old_global_straight, random_global_straight
        )
        logger.info(
            f"   • Global straightness: {random_global_straight} ({random_global_straight_change})"
        )

        logger.info("")
        logger.info("📁 Output files saved:")
        
        # Update file descriptions to mention clustering
        logger.info(
            f"   • road_network_baseline_{pbf_stem}.gpkg - Complete road network with centrality measures"
        )
        logger.info(
            f"   • road_network_smart_filtered_{pbf_stem}.gpkg - Road network after strategic station removal"
        )
        logger.info(
            f"   • road_network_random_filtered_{pbf_stem}.gpkg - Road network after random station removal"
        )
        logger.info(
            f"   • removed_stations_smart_{pbf_stem}.gpkg - Stations removed by smart k-NN analysis"
        )
        logger.info(
            f"   • removed_stations_random_{pbf_stem}.gpkg - Stations removed by random selection"
        )

        log_step_end(step_start, "13", "Results comparison")

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        import traceback

        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
