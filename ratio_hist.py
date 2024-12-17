import numpy as np
import matplotlib.pyplot as plt
from optimized_impl import prune_graph

# Parameters
dimension = 5  # Number of elements in each vertex
num_trials = 100

# Store edge ratios
edge_ratios = []

# Run the experiment for different random seeds and number of vertices
for seed in range(num_trials):
    np.random.seed(seed)
    num_vertices = np.random.randint(50, 200)  # Randomize number of vertices between 50 and 200
    vertices = [np.random.rand(dimension) for _ in range(num_vertices)]

    # Prune the graph and calculate the edge ratio
    pruned_edges = prune_graph(vertices)
    num_pruned_edges = len(pruned_edges)
    num_total_edges = num_vertices * (num_vertices - 1)
    edge_ratio = num_pruned_edges / num_total_edges

    edge_ratios.append(edge_ratio)

# Plot the histogram of edge ratios
plt.figure(figsize=(10, 6))
plt.hist(edge_ratios, bins=20, edgecolor='black', alpha=0.7)
plt.title("Histogram of Edge Ratios in Pruned Graphs with Varying Vertex Counts",fontsize=22)
plt.xlabel("Edge Ratio (Pruned Edges / Total Edges)", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
