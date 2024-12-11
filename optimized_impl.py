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
        - Precomputing distances: O(n^2)
        - Outer loop (over vertices): O(n)
        - Subset calculation: Optimized with caching: O(n^2)
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

    # Iterate over all vertices to construct edges for each vertex x
    for x in range(num_vertices):
        local_covered = set()
        valid_edges = []

        # Sort other vertices by their contribution to coverage
        candidates = sorted(
            [(y, {q for q in range(num_vertices) if distance_cache[y, q] < distance_cache[x, q]})
             for y in range(num_vertices) if y != x],
            key=lambda item: -len(item[1])  # Prioritize candidates with larger subsets
        )

        for y, U_y in candidates:
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
        covered_vertices = set([x])  # Include x itself in the covered vertices

        # Get incoming edges for vertex x
        incoming_edges = [y for y, target in pruned_edges if target == x]

        for y in incoming_edges:
            U_y = {q for q in range(num_vertices) if distance_cache[y, q] < distance_cache[x, q]}
            covered_vertices.update(U_y)

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

    pruned_edges = prune_graph(vertices)
    print("Pruned Edges:", pruned_edges)

    # Compute the ratio of pruned edges to total edges in a fully connected graph
    num_pruned_edges = len(pruned_edges)
    num_total_edges = num_vertices * (num_vertices - 1)
    edge_ratio = num_pruned_edges / num_total_edges
    print(f"Ratio of pruned edges to total edges: {edge_ratio:.4f}")

    is_valid = validate_pruned_graph(vertices, pruned_edges)
    print("Is the pruned graph valid?", is_valid)
