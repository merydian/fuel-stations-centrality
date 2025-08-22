#!/usr/bin/env python3
"""
Benchmark script to compare performance with and without Numba optimization.
"""

import time
import igraph as ig
import numpy as np
import sys
import os

# Temporarily disable Numba to test fallback
sys.path.insert(0, os.path.dirname(__file__))


def create_test_graph(n=100):
    """Create a test graph with random coordinates"""
    g = ig.Graph.Erdos_Renyi(n=n, p=0.05)

    np.random.seed(42)
    x_coords = np.random.uniform(0, 100, n)
    y_coords = np.random.uniform(0, 100, n)

    g.vs["x"] = x_coords
    g.vs["y"] = y_coords
    g.es["weight"] = np.random.uniform(1, 10, g.ecount())

    return g


def benchmark_performance():
    """Benchmark performance comparison"""
    print("=== Numba Straightness Centrality Performance Benchmark ===\n")

    # Test different graph sizes
    sizes = [25, 50, 75]

    for size in sizes:
        print(f"Testing with {size} nodes...")
        g = create_test_graph(n=size)
        print(f"Graph: {g.vcount()} nodes, {g.ecount()} edges")

        # Import with Numba enabled
        from centrality import straightness_centrality

        # Test straightness_centrality - first run (includes JIT compilation)
        print("  First run (includes JIT compilation):")
        start_time = time.time()
        result1 = straightness_centrality(g, weight="weight")
        time1 = time.time() - start_time
        print(f"    Time: {time1:.3f} seconds")

        # Test straightness_centrality - second run (JIT already compiled)
        print("  Second run (JIT already compiled):")
        start_time = time.time()
        result2 = straightness_centrality(g, weight="weight")
        time2 = time.time() - start_time
        print(f"    Time: {time2:.3f} seconds")

        # Verify results are identical
        assert np.allclose(result1, result2), "Results should be identical"

        print(f"  Speedup (first vs second run): {time1 / time2:.1f}x")
        print(f"  Average straightness: {np.mean(result2):.4f}")
        print()


if __name__ == "__main__":
    benchmark_performance()
