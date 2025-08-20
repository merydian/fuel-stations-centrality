#!/usr/bin/env python3
"""
Test script to verify Numba optimization for straightness centrality.
"""

import time
import igraph as ig
import numpy as np
from centrality import straightness_centrality, graph_straightness

def create_test_graph(n=100):
    """Create a test graph with random coordinates"""
    # Create a random graph
    g = ig.Graph.Erdos_Renyi(n=n, p=0.05)
    
    # Add random coordinates
    np.random.seed(42)  # For reproducible results
    x_coords = np.random.uniform(0, 100, n)
    y_coords = np.random.uniform(0, 100, n)
    
    g.vs["x"] = x_coords
    g.vs["y"] = y_coords
    
    # Add random weights
    g.es["weight"] = np.random.uniform(1, 10, g.ecount())
    
    return g

def benchmark_straightness():
    """Benchmark the straightness centrality computation"""
    print("Creating test graph...")
    g = create_test_graph(n=50)  # Small graph for testing
    print(f"Test graph: {g.vcount()} nodes, {g.ecount()} edges")
    
    print("\nTesting straightness_centrality function...")
    start_time = time.time()
    result = straightness_centrality(g, weight="weight")
    end_time = time.time()
    
    print(f"Computation completed in {end_time - start_time:.3f} seconds")
    print(f"Average straightness centrality: {np.mean(result):.6f}")
    print(f"Max straightness centrality: {max(result):.6f}")
    print(f"Min straightness centrality: {min(result):.6f}")
    
    print("\nTesting graph_straightness function...")
    start_time = time.time()
    global_straightness = graph_straightness(g, weight="weight")
    end_time = time.time()
    
    print(f"Global straightness computation completed in {end_time - start_time:.3f} seconds")
    print(f"Global straightness centrality: {global_straightness:.6f}")

if __name__ == "__main__":
    benchmark_straightness()
