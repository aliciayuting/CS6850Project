'''
Brute force implementation of the graph pruning algorithm with complexity O(n^3).
'''

import numpy as np
from itertools import combinations

# Define the distance function (Euclidean distance by default)
def distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def prune_graph_brute_force(vertices):
    """
    Brute-force pruning of a fully connected graph of vertices based on the given conditions.

    Args:
        vertices (list of np.array): List of n-dimensional vectors representing vertices.

    Returns:
        list of tuples: Pruned edge set as a list of directed edges (v, u).

    Complexity:
        - Outer loop (over vertices): O(n)
        - Inner loop (over candidate edges): O(n^2)
        - Total: O(n^3)
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

    # Iterate over all vertices to construct edges for each vertex x
    for x in range(num_vertices):
        local_covered = set()
        valid_edges = []

        # Iterate over all possible edges (y -> x)
        for y in range(num_vertices):
            if y == x:
                continue

            # Compute the subset U_y for vertex y
            U_y = {q for q in range(num_vertices) if distance_cache[y, q] < distance_cache[x, q]}

            # Add edge if it contributes new coverage
            uncovered = U_y - local_covered
            if uncovered:
                valid_edges.append((y, x))
                local_covered.update(uncovered)

            # Stop if all vertices are covered for this x
            if len(local_covered) == num_vertices:
                break

        pruned_edges.extend(valid_edges)

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
        - Precomputing distances: O(n^2)
        - Validation: O(n^2)
    """
    num_vertices = len(vertices)

    # Precompute distances between all vertices
    distance_cache = np.zeros((num_vertices, num_vertices))
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            dist = distance(vertices[i], vertices[j])
            distance_cache[i, j] = dist
            distance_cache[j, i] = dist

    # Validate for each vertex x
    for x in range(num_vertices):
        covered_vertices = set()

        # Get incoming edges for vertex x
        incoming_edges = [y for y, target in pruned_edges if target == x]

        for y in incoming_edges:
            U_y = {q for q in range(num_vertices) if distance_cache[y, q] < distance_cache[x, q]}
            covered_vertices.update(U_y)

        # Ensure vertex x itself is included in the coverage
        covered_vertices.add(x)

        # Debug: Log intermediate validation state
        # print(f"Validating vertex {x}:")
        for y in incoming_edges:
            U_y = {q for q in range(num_vertices) if distance_cache[y, q] < distance_cache[x, q]}
        #     print(f"Edge ({y} -> {x}), U_y: {U_y}")
        # print(f"Covered vertices for {x}: {covered_vertices}")

        # Check if the union of subsets U_y covers all vertices
        if len(covered_vertices) != num_vertices:
            print(f"Validation failed: Vertex {x} is not fully covered by its incoming edges.")
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

    # Run the brute-force pruning algorithm
    pruned_edges_brute_force = prune_graph_brute_force(vertices)
    # print("Pruned Edges (Brute Force):", pruned_edges_brute_force)

    # Compute the ratio of pruned edges to total edges in a fully connected graph (Brute Force)
    num_pruned_edges_brute = len(pruned_edges_brute_force)
    num_total_edges = num_vertices * (num_vertices - 1)
    edge_ratio_brute = num_pruned_edges_brute / num_total_edges
    print(f"Ratio of pruned edges to total edges (Brute Force): {edge_ratio_brute:.4f}")

    is_valid_brute = validate_pruned_graph(vertices, pruned_edges_brute_force)
    print("Is the pruned graph valid (Brute Force)?", is_valid_brute)
