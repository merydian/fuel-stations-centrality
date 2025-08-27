#include "stats.h"
#include "graph.h"  // Assuming graph class exists
#include "config.h" // Assuming config class exists
#include "centrality.h" // Assuming centrality functions exist
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

GraphStats GraphStatsCalculator::getGraphStats(const Graph& G, const std::vector<std::pair<double, double>>* base_convex_hull) {
    std::cout << "Computing comprehensive graph statistics..." << std::endl;
    
    GraphStats stats;
    
    // Basic graph statistics
    stats["num_nodes"] = StatValue(static_cast<int>(G.getNodeCount()), "count");
    stats["num_edges"] = StatValue(static_cast<int>(G.getEdgeCount()), "count");
    
    std::cout << "Basic stats: " << G.getNodeCount() << " nodes, " << G.getEdgeCount() << " edges" << std::endl;
    
    // Check connectivity
    bool is_connected = G.isConnected();
    stats["is_connected"] = StatValue(is_connected, "boolean");
    
    auto components = G.getConnectedComponents();
    stats["num_components"] = StatValue(static_cast<int>(components.size()), "count");
    
    std::cout << "Graph connectivity: " << is_connected << ", Components: " << components.size() << std::endl;
    
    // Density
    stats["density"] = StatValue(G.getDensity(), "ratio");
    
    // Average degree
    auto degrees = G.getDegrees();
    double avg_degree = std::accumulate(degrees.begin(), degrees.end(), 0.0) / degrees.size();
    auto minmax_degree = std::minmax_element(degrees.begin(), degrees.end());
    
    stats["avg_degree"] = StatValue(avg_degree, "connections");
    stats["max_degree"] = StatValue(static_cast<double>(*minmax_degree.second), "connections");
    stats["min_degree"] = StatValue(static_cast<double>(*minmax_degree.first), "connections");
    
    std::cout << "Degree stats - Avg: " << avg_degree 
              << ", Max: " << *minmax_degree.second 
              << ", Min: " << *minmax_degree.first << std::endl;
    
    // Get largest connected component for centrality calculations
    Graph largest_cc = G;
    if (!is_connected) {
        auto largest_component = *std::max_element(components.begin(), components.end(),
            [](const auto& a, const auto& b) { return a.size() < b.size(); });
        largest_cc = G.getSubgraph(largest_component);
        std::cout << "Using largest component with " << largest_cc.getNodeCount() 
                  << " nodes for centrality calculations" << std::endl;
    } else {
        std::cout << "Using entire graph for centrality calculations (graph is connected)" << std::endl;
    }
    
    // Degree centrality
    if (Config::CALCULATE_DEGREE_CENTRALITY) {
        std::cout << "Computing degree centrality..." << std::endl;
        try {
            auto degree_centrality = calculateDegreeCentrality(largest_cc);
            double avg_degree_centrality = std::accumulate(degree_centrality.begin(), degree_centrality.end(), 0.0) / degree_centrality.size();
            stats["avg_degree_centrality"] = StatValue(avg_degree_centrality, "normalized ratio");
        } catch (const std::exception& e) {
            stats["avg_degree_centrality"] = StatValue("Could not compute", "n/a");
        }
    } else {
        std::cout << "Skipping degree centrality calculation (disabled in config)" << std::endl;
        stats["avg_degree_centrality"] = StatValue("Calculation disabled", "n/a");
    }
    
    // Closeness centrality
    if (Config::CALCULATE_CLOSENESS_CENTRALITY) {
        std::cout << "Computing closeness centrality..." << std::endl;
        try {
            auto closeness_centrality = calculateClosenessCentrality(largest_cc);
            double avg_closeness_centrality = std::accumulate(closeness_centrality.begin(), closeness_centrality.end(), 0.0) / closeness_centrality.size();
            stats["avg_closeness_centrality"] = StatValue(avg_closeness_centrality, "normalized ratio");
        } catch (const std::exception& e) {
            std::cout << "Failed to compute closeness centrality: " << e.what() << std::endl;
            stats["avg_closeness_centrality"] = StatValue("Could not compute", "n/a");
        }
    } else {
        std::cout << "Skipping closeness centrality calculation (disabled in config)" << std::endl;
        stats["avg_closeness_centrality"] = StatValue("Calculation disabled", "n/a");
    }
    
    // Betweenness centrality
    if (Config::CALCULATE_BETWEENNESS_CENTRALITY) {
        std::cout << "Computing betweenness centrality..." << std::endl;
        try {
            auto betweenness_centrality = calculateBetweennessCentrality(largest_cc);
            double avg_betweenness_centrality = std::accumulate(betweenness_centrality.begin(), betweenness_centrality.end(), 0.0) / betweenness_centrality.size();
            stats["avg_betweenness_centrality"] = StatValue(avg_betweenness_centrality, "normalized ratio");
        } catch (const std::exception& e) {
            std::cout << "Failed to compute betweenness centrality: " << e.what() << std::endl;
            stats["avg_betweenness_centrality"] = StatValue("Could not compute", "n/a");
        }
    } else {
        std::cout << "Skipping betweenness centrality calculation (disabled in config)" << std::endl;
        stats["avg_betweenness_centrality"] = StatValue("Calculation disabled", "n/a");
    }
    
    // Eigenvector centrality
    if (Config::CALCULATE_EIGENVECTOR_CENTRALITY) {
        if (largest_cc.isConnected()) {
            std::cout << "Computing eigenvector centrality..." << std::endl;
            try {
                auto eigenvector_centrality = calculateEigenvectorCentrality(largest_cc);
                double avg_eigenvector_centrality = std::accumulate(eigenvector_centrality.begin(), eigenvector_centrality.end(), 0.0) / eigenvector_centrality.size();
                stats["avg_eigenvector_centrality"] = StatValue(avg_eigenvector_centrality, "normalized ratio");
            } catch (const std::exception& e) {
                std::cout << "Failed to compute eigenvector centrality: " << e.what() << std::endl;
                stats["avg_eigenvector_centrality"] = StatValue("Could not compute", "n/a");
            }
        } else {
            std::cout << "Skipping eigenvector centrality (largest component not connected)" << std::endl;
            stats["avg_eigenvector_centrality"] = StatValue("Graph not connected", "n/a");
        }
    } else {
        std::cout << "Skipping eigenvector centrality calculation (disabled in config)" << std::endl;
        stats["avg_eigenvector_centrality"] = StatValue("Calculation disabled", "n/a");
    }
    
    // Straightness centrality and global straightness
    if (Config::CALCULATE_STRAIGHTNESS_CENTRALITY || Config::CALCULATE_GLOBAL_STRAIGHTNESS) {
        std::cout << "Computing straightness measures..." << std::endl;
        try {
            if (largest_cc.hasCoordinates()) {
                if (Config::CALCULATE_STRAIGHTNESS_CENTRALITY) {
                    auto straightness_values = calculateStraightnessCentrality(largest_cc);
                    double avg_straightness = std::accumulate(straightness_values.begin(), straightness_values.end(), 0.0) / straightness_values.size();
                    auto minmax_straightness = std::minmax_element(straightness_values.begin(), straightness_values.end());
                    
                    stats["avg_straightness_centrality"] = StatValue(avg_straightness, "ratio");
                    stats["max_straightness_centrality"] = StatValue(*minmax_straightness.second, "ratio");
                    stats["min_straightness_centrality"] = StatValue(*minmax_straightness.first, "ratio");
                } else {
                    std::cout << "Skipping straightness centrality calculation (disabled in config)" << std::endl;
                    stats["avg_straightness_centrality"] = StatValue("Calculation disabled", "n/a");
                }
                
                if (Config::CALCULATE_GLOBAL_STRAIGHTNESS) {
                    double global_straightness = calculateGlobalStraightness(largest_cc);
                    stats["global_straightness"] = StatValue(global_straightness, "ratio");
                } else {
                    std::cout << "Skipping global straightness calculation (disabled in config)" << std::endl;
                    stats["global_straightness"] = StatValue("Calculation disabled", "n/a");
                }
            } else {
                std::cout << "No coordinate data found for straightness centrality" << std::endl;
                stats["avg_straightness_centrality"] = StatValue("No coordinate data", "n/a");
                stats["global_straightness"] = StatValue("No coordinate data", "n/a");
            }
        } catch (const std::exception& e) {
            std::cout << "Failed to compute straightness measures: " << e.what() << std::endl;
            stats["avg_straightness_centrality"] = StatValue("Could not compute", "n/a");
            stats["global_straightness"] = StatValue("Could not compute", "n/a");
        }
    } else {
        std::cout << "Skipping all straightness calculations (disabled in config)" << std::endl;
        stats["avg_straightness_centrality"] = StatValue("Calculation disabled", "n/a");
        stats["global_straightness"] = StatValue("Calculation disabled", "n/a");
    }
    
    // Farness centrality (if available)
    if (G.hasFarnessData()) {
        std::cout << "Processing farness centrality statistics..." << std::endl;
        auto farness_values = G.getFarnessValues();
        double avg_farness = std::accumulate(farness_values.begin(), farness_values.end(), 0.0) / farness_values.size();
        auto minmax_farness = std::minmax_element(farness_values.begin(), farness_values.end());
        
        stats["avg_farness"] = StatValue(avg_farness, "meters");
        stats["max_farness"] = StatValue(*minmax_farness.second, "meters");
        stats["min_farness"] = StatValue(*minmax_farness.first, "meters");
    } else {
        std::cout << "No farness centrality data found" << std::endl;
    }
    
    // Normalized farness centrality (if available)
    if (G.hasNormalizedFarnessData()) {
        std::cout << "Processing normalized farness centrality statistics..." << std::endl;
        auto norm_farness_values = G.getNormalizedFarnessValues();
        double avg_norm_farness = std::accumulate(norm_farness_values.begin(), norm_farness_values.end(), 0.0) / norm_farness_values.size();
        auto minmax_norm_farness = std::minmax_element(norm_farness_values.begin(), norm_farness_values.end());
        
        stats["avg_norm_farness"] = StatValue(avg_norm_farness, "meters/node");
        stats["max_norm_farness"] = StatValue(*minmax_norm_farness.second, "meters/node");
        stats["min_norm_farness"] = StatValue(*minmax_norm_farness.first, "meters/node");
    }
    
    std::cout << "Graph statistics computation completed" << std::endl;
    return stats;
}

std::string GraphStatsCalculator::formatGraphStats(const GraphStats& stats, const std::string& title) {
    std::ostringstream oss;
    
    oss << "\n" << std::string(50, '=') << "\n";
    oss << std::setw(50) << title << "\n";
    oss << std::string(50, '=') << "\n";
    
    // Basic Statistics
    oss << "\nBasic Statistics:\n";
    oss << "  Nodes: " << formatValue(stats.at("num_nodes")) << "\n";
    oss << "  Edges: " << formatValue(stats.at("num_edges")) << "\n";
    oss << "  Connected: " << formatValue(stats.at("is_connected")) << "\n";
    oss << "  Components: " << formatValue(stats.at("num_components")) << "\n";
    oss << "  Density: " << formatValue(stats.at("density"), 6) << "\n";
    
    // ...existing sections for spatial, degree, centrality statistics...
    
    oss << std::string(50, '=') << "\n\n";
    
    return oss.str();
}

std::string GraphStatsCalculator::compareGraphStats(const GraphStats& stats1, const GraphStats& stats2, 
                                                   const std::string& title1, const std::string& title2) {
    std::ostringstream oss;
    
    oss << "\n" << std::string(70, '=') << "\n";
    oss << std::setw(70) << "GRAPH COMPARISON" << "\n";
    oss << std::string(70, '=') << "\n";
    oss << std::left << std::setw(35) << title1 << " vs " << std::right << std::setw(33) << title2 << "\n";
    oss << std::string(70, '-') << "\n";
    
    // ...existing comparison logic...
    
    oss << std::string(70, '=') << "\n\n";
    
    return oss.str();
}

// Private helper methods
std::vector<double> GraphStatsCalculator::calculateDegreeCentrality(const Graph& G) {
    auto degrees = G.getDegrees();
    int max_possible_degree = G.getNodeCount() - 1;
    
    std::vector<double> normalized_centrality;
    normalized_centrality.reserve(degrees.size());
    
    for (int degree : degrees) {
        double normalized = (max_possible_degree > 0) ? static_cast<double>(degree) / max_possible_degree : 0.0;
        normalized_centrality.push_back(normalized);
    }
    
    return normalized_centrality;
}

std::vector<double> GraphStatsCalculator::calculateClosenessCentrality(const Graph& G) {
    // Implementation depends on Graph class interface
    return G.getClosenessCentrality();
}

std::vector<double> GraphStatsCalculator::calculateBetweennessCentrality(const Graph& G) {
    // Implementation depends on Graph class interface
    return G.getBetweennessCentrality();
}

std::vector<double> GraphStatsCalculator::calculateEigenvectorCentrality(const Graph& G) {
    // Implementation depends on Graph class interface
    return G.getEigenvectorCentrality();
}

std::vector<double> GraphStatsCalculator::calculateStraightnessCentrality(const Graph& G) {
    // Implementation depends on centrality module
    return straightness_centrality(G);
}

double GraphStatsCalculator::calculateGlobalStraightness(const Graph& G) {
    // Implementation depends on centrality module
    return graph_straightness(G);
}

std::string GraphStatsCalculator::formatValue(const StatValue& stat, int precision) {
    std::ostringstream oss;
    
    if (!stat.is_numeric) {
        oss << stat.value_str << " (" << stat.unit << ")";
    } else {
        if (precision >= 0) {
            oss << std::fixed << std::setprecision(precision);
        }
        oss << stat.value_num << " " << stat.unit;
    }
    
    return oss.str();
}

std::string GraphStatsCalculator::calculateChange(double val1, double val2) {
    if (val1 == 0.0) {
        return (val2 == 0.0) ? "N/A" : "∞";
    }
    
    double change = ((val2 - val1) / val1) * 100.0;
    std::ostringstream oss;
    oss << std::showpos << std::fixed << std::setprecision(2) << change << "%";
    return oss.str();
}
