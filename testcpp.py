# Test in Python after compilation
import stats_cpp

# Create sample stats
stats = stats_cpp.GraphStatsCalculator.create_sample_stats()

# Format and print
formatted = stats_cpp.GraphStatsCalculator.format_graph_stats(stats, "Sample Graph")
print(formatted)

# Access individual stats
print(f"Number of nodes: {stats['num_nodes'].value_num}")
print(f"Is connected: {stats['is_connected'].value_str}")