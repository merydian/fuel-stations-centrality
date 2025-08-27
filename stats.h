#ifndef STATS_H
#define STATS_H

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <iostream>

// Forward declarations
class Graph;
class Config;

struct StatValue {
    std::string value_str;
    double value_num;
    bool is_numeric;
    std::string unit;
    
    StatValue(const std::string& val, const std::string& u) 
        : value_str(val), value_num(0.0), is_numeric(false), unit(u) {}
    
    StatValue(double val, const std::string& u) 
        : value_str(""), value_num(val), is_numeric(true), unit(u) {}
        
    StatValue(int val, const std::string& u) 
        : value_str(""), value_num(static_cast<double>(val)), is_numeric(true), unit(u) {}
        
    StatValue(bool val, const std::string& u) 
        : value_str(val ? "true" : "false"), value_num(val ? 1.0 : 0.0), is_numeric(false), unit(u) {}
};

using GraphStats = std::map<std::string, StatValue>;

class GraphStatsCalculator {
public:
    static GraphStats getGraphStats(const Graph& G, const std::vector<std::pair<double, double>>* base_convex_hull = nullptr);
    
    static std::string formatGraphStats(const GraphStats& stats, const std::string& title = "Graph Statistics");
    
    static std::string compareGraphStats(const GraphStats& stats1, const GraphStats& stats2, 
                                       const std::string& title1 = "Graph 1", 
                                       const std::string& title2 = "Graph 2");

private:
    static std::vector<double> calculateDegreeCentrality(const Graph& G);
    static std::vector<double> calculateClosenessCentrality(const Graph& G);
    static std::vector<double> calculateBetweennessCentrality(const Graph& G);
    static std::vector<double> calculateEigenvectorCentrality(const Graph& G);
    static std::vector<double> calculateStraightnessCentrality(const Graph& G);
    static double calculateGlobalStraightness(const Graph& G);
    
    static std::string formatValue(const StatValue& stat, int precision = -1);
    static std::string calculateChange(double val1, double val2);
};

#endif // STATS_H
