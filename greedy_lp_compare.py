import numpy as np
import matplotlib.pyplot as plt
from gurobipy import GRB
import gurobipy as gp
from utils import *
from optimized_impl import prune_graph
from gurobi_global import solve_minimum_edge_cover

# Reusing your prune_graph and solve_minimum_edge_cover functions
# (Assuming both are already defined in the script)

# Function to generate Gaussian-distributed points
def generate_gaussian_points(n, d, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return [np.random.randn(d) for _ in range(n)]

# Main program
if __name__ == "__main__":
    # Parameters
    num_vertices = 100
    dimension = 3
    num_runs = 100

    # Lists to store edge counts across runs
    pruned_edge_counts = []
    min_edge_cover_edge_counts = []

    # Perform multiple runs
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        # Generate vertices
        vertices = generate_gaussian_points(num_vertices, dimension, seed=run)

        # Compute edges using prune_graph
        pruned_edges = prune_graph(vertices)
        pruned_edge_counts.append(len(pruned_edges))

        # Compute edges using solve_minimum_edge_cover
        min_edge_cover_edges, _ = solve_minimum_edge_cover(vertices, directed=True, relaxed=False)
        min_edge_cover_edge_counts.append(len(min_edge_cover_edges))

    # Plot histograms on the same plot
    plt.figure(figsize=(12, 6))

    max_edge_count = max(max(pruned_edge_counts), max(min_edge_cover_edge_counts))
    bin_width = 5  # Adjust this for wider columns
    bins = np.arange(0, max_edge_count + bin_width, bin_width)


    plt.hist(
        pruned_edge_counts,
        bins=bins,
        alpha=0.7,
        label="Greedy Algorithm",
        color="blue",
        edgecolor="black"
    )

    plt.hist(
        min_edge_cover_edge_counts,
        bins=bins,
        alpha=0.7,
        label="LP Algorithm",
        color="green",
        edgecolor="black"
    )

    # Plot details
    plt.title(f"Histogram of Edge Counts Across {num_runs} Runs", fontsize=24)
    plt.xlabel("Number of Edges", fontsize=22)
    plt.ylabel("Frequency", fontsize=23)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
