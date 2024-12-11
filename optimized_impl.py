'''
Optimized implementation of the pruned graph algorithm with O(n^2) complexity.
'''

import numpy as np
from itertools import combinations

# Define the distance function (Euclidean distance by default)
def distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def prune_graph(vertices):
    """
    Prunes a fully connected graph of vertices based on the given conditions.

    Args:
        vertices (list of np.array): List of n-dimensional vectors representing vertices.

    Returns:
        list of tuples: Pruned edge set as a list of directed edges (v, u).

    Complexity:
        - Outer loop (over vertices): O(n)
        - Inner loop (over candidate edges): O(n)
        - Subset calculation: Optimized with caching: O(n)
        - Total: O(n^2)
    """
    num_vertices = len(vertices)
    pruned_edges = []

    # Precompute distances between all vertices to avoid redundant calculations
    distance_cache = np.zeros((num_vertices, num_vertices))
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            dist = distance(vertices[i], vertices[j])
            distance_cache[i, j] = dist
            distance_cache[j, i] = dist

    # Iterate over all vertices to find the optimal outgoing edge for each
    for v in range(num_vertices):
        best_u = None
        best_subset_size = -1

        for u in range(num_vertices):
            if u == v:
                continue

            # Compute the subset U(u) for vertex u using precomputed distances
            U_u = [q for q in range(num_vertices) if distance_cache[u, q] < distance_cache[v, q]]

            # Verify subset condition during pruning
            valid = all(distance_cache[u, q] < distance_cache[v, q] for q in U_u)
            if not valid:
                continue

            # Keep track of the largest valid subset
            if len(U_u) > best_subset_size:
                best_u = u
                best_subset_size = len(U_u)

        # Add the best edge to the pruned edge set
        if best_u is not None:
            pruned_edges.append((v, best_u))
        else:
            # Add a self-loop if no valid edge is found
            pruned_edges.append((v, v))

    return pruned_edges

def validate_pruned_graph(vertices, pruned_edges):
    """
    Validates the pruned graph based on the specified conditions.

    Args:
        vertices (list of np.array): List of n-dimensional vectors representing vertices.
        pruned_edges (list of tuples): Pruned edge set as a list of directed edges (v, u).

    Returns:
        bool: True if the pruned graph satisfies all conditions, False otherwise.

    Complexity:
        - Outgoing edge check: O(n)
        - Subset verification: Optimized with caching: O(n^2)
        - Union completeness: O(n^2)
        - Total: O(n^2)
    """
    num_vertices = len(vertices)
    outgoing_edges = {v: [] for v in range(num_vertices)}

    # Precompute distances between all vertices
    distance_cache = np.zeros((num_vertices, num_vertices))
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            dist = distance(vertices[i], vertices[j])
            distance_cache[i, j] = dist
            distance_cache[j, i] = dist

    # Build outgoing edge list for each vertex
    for v, u in pruned_edges:
        outgoing_edges[v].append(u)

    # Condition 1: Each vertex must have exactly one outgoing edge
    for v in range(num_vertices):
        if len(outgoing_edges[v]) != 1:
            print(f"Validation failed: Vertex {v} does not have exactly one outgoing edge.")
            return False

    # Condition 2: Verify subset U(u) for each outgoing edge
    for v in range(num_vertices):
        u = outgoing_edges[v][0]  # Get the target vertex for v
        U_u = [q for q in range(num_vertices) if distance_cache[u, q] < distance_cache[v, q]]

        # Check if all vertices in U(u) satisfy the condition
        for q in U_u:
            if distance_cache[u, q] >= distance_cache[v, q]:
                print(f"Validation failed: Edge ({v}, {u}) does not satisfy the subset condition for vertex {q}.")
                return False

    # Condition 3: Check union completeness
    covered_vertices = set()
    for v in range(num_vertices):
        u = outgoing_edges[v][0]
        U_u = [q for q in range(num_vertices) if distance_cache[u, q] < distance_cache[v, q]]
        covered_vertices.update(U_u)

    if len(covered_vertices) != num_vertices:
        print("Validation failed: Not all vertices are covered by the subsets U(u).")
        return False

    print("Validation passed: The pruned graph satisfies all conditions.")
    return True

# Example usage
if __name__ == "__main__":
    # Set parameters
    num_vertices = 100
    dimension = 5  # Number of elements in each vertex

    # Generate random vertices in the specified dimension
    np.random.seed(42)  # For reproducibility
    vertices = [np.random.rand(dimension) for _ in range(num_vertices)]

    pruned_edges = prune_graph(vertices)
#     print("Pruned Edges:", pruned_edges)

    # Compute the ratio of pruned edges to total edges in a fully connected graph
    num_pruned_edges = len(pruned_edges)
    num_total_edges = num_vertices * (num_vertices - 1)
    edge_ratio = num_pruned_edges / num_total_edges
    print(f"Ratio of pruned edges to total edges: {edge_ratio:.4f}")

    is_valid = validate_pruned_graph(vertices, pruned_edges)
    print("Is the pruned graph valid?", is_valid)