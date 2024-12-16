import matplotlib.pyplot as plt
import numpy as np
from utils import generate_gaussian_points, plot_3d_graph, edgelist_to_neighborhoods
from gurobi_global import solve_minimum_edge_cover

import networkx as nx

def average_degree_vs_dimension(n_points, dimensions, directed=False, relaxed=False):
     """
     Generate average degree vs dimension plot.
     """
     avg_degrees = []
     for dim in dimensions:
          points = generate_gaussian_points(n_points, dim)
          edges_selected, _ = solve_minimum_edge_cover(points, directed=directed, relaxed=relaxed)
          num_edges = len(edges_selected)
          avg_degree = (2 * num_edges) / n_points
          avg_degrees.append(avg_degree)
          print(f"Dimension: {dim}, Average Degree: {avg_degree:.2f}")
     
     # Plotting
     plt.figure(figsize=(8, 6))
     plt.plot(dimensions, avg_degrees, marker='o')
     plt.title("Average Degree vs Dimension", fontsize=25)
     plt.xlabel("Dimension", fontsize=20)
     plt.ylabel("Average Degree", fontsize=20)
     plt.tick_params(axis='x', labelsize=20)  # Set x-axis tick label size
     plt.tick_params(axis='y', labelsize=20)  # Set y-axis tick label size
     plt.grid(True)
     plt.show()
     




def compute_path_lengths_directed(points, edges):
     """
     Compute the lengths of the shortest paths in the directed graph.

     Args:
          points (numpy.ndarray): The coordinates of the points.
          edges (List[Tuple[int, int]]): The edges of the graph.

     Returns:
          List[float]: A list of shortest path lengths.
     """
     # Build the directed graph
     G = nx.DiGraph()
     
     for i, j in edges:
          dist = np.linalg.norm(points[i] - points[j])
          G.add_edge(i, j, weight=dist)
     
     # Compute shortest paths between all pairs
     path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
     
     # Extract all path lengths into a list
     lengths = []
     for source, target_lengths in path_lengths.items():
          for target, length in target_lengths.items():
               if source != target:  # Exclude self-loops
                    lengths.append(length)
     
     return lengths

def plot_path_length_histogram(path_lengths, title="Path Length Histogram"):
     """
     Plot a histogram of path lengths.

     Args:
          path_lengths (List[float]): The list of path lengths.
          title (str): The title of the plot.
     """
     plt.figure(figsize=(8, 6))
     plt.hist(path_lengths, bins=30, color='blue', alpha=0.7, edgecolor='black')
     plt.title(title, fontsize=25)
     plt.xlabel("Path Length", fontsize=20)
     plt.ylabel("Frequency", fontsize=20)
     plt.tick_params(axis='x', labelsize=15)  # Set x-axis tick label size
     plt.tick_params(axis='y', labelsize=15)  # Set y-axis tick label size
     plt.grid(True)
     plt.show()

# if __name__ == "__main__":
#      n_points = 250
#      dimensions = 3  # Change this for different dimensions
#      points = generate_gaussian_points(n_points, dimensions)

#      # Solve directed minimum edge cover
#      edges, _ = solve_minimum_edge_cover(points, directed=True, relaxed=False)
     
#      # Compute shortest path lengths
#      path_lengths = compute_path_lengths_directed(points, edges)
     
#      # Plot histogram
#      plot_path_length_histogram(
#           path_lengths, 
#           title=f"Path Length Histogram for {dimensions}D Directed Navigable Graph"
#      )

# Main script for degree vs dimension
if __name__ == "__main__":
    n_points = 250
    dimensions = [2,4,6,8,10] # Dimensions from 2 to 20
    average_degree_vs_dimension(n_points, dimensions, directed=False, relaxed=True)


def compute_hop_counts(points, edges, directed=False):
    """
    Compute the number of hops (unweighted shortest paths) in the graph.

    Args:
        points (numpy.ndarray): The coordinates of the points.
        edges (List[Tuple[int, int]]): The edges of the graph.
        directed (bool): Whether the graph is directed.

    Returns:
        List[int]: A list of shortest path hop counts.
    """
    # Build the graph
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edges)
    
    # Compute shortest paths (unweighted)
    path_hops = dict(nx.all_pairs_shortest_path_length(G))
    
    # Extract all hop counts into a list
    hops = []
    for source, target_hops in path_hops.items():
        for target, hop_count in target_hops.items():
            if source != target:  # Exclude self-loops
                hops.append(hop_count)
    
    return hops

def plot_hop_count_histogram(hops, title="Hop Count Histogram"):
    """
    Plot a histogram of hop counts.

    Args:
        hops (List[int]): The list of hop counts.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(hops, bins=range(1, max(hops) + 2), color='green', alpha=0.7, edgecolor='black', align='left')
    plt.title(title, fontsize=25)
    plt.xlabel("Number of Hops", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.tick_params(axis='x', labelsize=15)  # Set x-axis tick label size
    plt.tick_params(axis='y', labelsize=15)  # Set y-axis tick label size
    plt.grid(True)
    plt.show()

# if __name__ == "__main__":
#     # Generate points
#     n_points = 250
#     dimensions = 3  # Change this for different dimensions
#     points = generate_gaussian_points(n_points, dimensions)

#     # Solve directed minimum edge cover
#     edges, _ = solve_minimum_edge_cover(points, directed=True, relaxed=False)
    
#     # Compute shortest path lengths (weighted)
#     path_lengths = compute_path_lengths_directed(points, edges)
#     plot_path_length_histogram(
#         path_lengths, 
#         title=f"Path Length Histogram for {dimensions}D Directed Navigable Graph"
#     )
    
#     # Compute number of hops (unweighted)
#     hops = compute_hop_counts(points, edges, directed=True)
#     plot_hop_count_histogram(
#         hops, 
#         title=f"Hop Count Histogram for {dimensions}D Directed Navigable Graph"
#     )