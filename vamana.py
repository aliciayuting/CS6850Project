import numpy as np
import math
import random
from collections import defaultdict

# rn this is a naive single-thread python port of the c++ logic above. 
# big disclaimers: concurrency is stripped out, so performance might be bad for big point sets. 
# we only do euclidean distance. 
# graph will be a dict: {index: set_of_neighbors}, with up to R neighbors per node after pruning.

def build_knn_graph(points, r=16, alpha=1.2, num_passes=2, single_batch=0, beam_size=16,
                    base=2.0, max_fraction=0.02, random_order=True):
    """
    points: np.ndarray of shape (n, d)
    r: max number of neighbors per node
    alpha: alpha param for robust pruning
    num_passes: how many passes to do
    single_batch: if nonzero, random start edges are used w/ the given degree
    beam_size: search beam for building
    base, max_fraction, random_order: controlling batch sizes & insertion order
    returns adjacency list as a dict {i: set_of_neighbors}
    """

    n = len(points)
    dim = points.shape[1] if n > 0 else 0
    graph = {i: set() for i in range(n)}

    def dist(a, b):
        # euclidean distance
        return np.linalg.norm(points[a] - points[b])

    def robust_prune(p, candidates, add=True):
        # p: index of current node
        # candidates: either indices or (index, distance) pairs
        # if just indices, we compute distance on the fly
        # returns (new_neighbors, distance_comps)
        distance_comps = 0
        cand_pairs = []
        for c in candidates:
            distance_comps += 1
            cand_pairs.append((c, dist(p, c)))
        if add:
            # add existing neighbors of p
            for neigh in graph[p]:
                distance_comps += 1
                cand_pairs.append((neigh, dist(p, neigh)))
        # sort by distance
        cand_pairs.sort(key=lambda x: (x[1], x[0]))
        # unique them
        unique_pairs = []
        seen = set()
        for cidx, cd in cand_pairs:
            if cidx not in seen:
                seen.add(cidx)
                unique_pairs.append((cidx, cd))
        new_nbhs = []
        candidate_idx = 0
        while len(new_nbhs) < r and candidate_idx < len(unique_pairs):
            cidx, cd = unique_pairs[candidate_idx]
            candidate_idx += 1
            if cidx == p or cidx == -1:
                continue
            new_nbhs.append(cidx)
            # alpha filter
            for i in range(candidate_idx, len(unique_pairs)):
                nxt_idx, nxt_dist = unique_pairs[i]
                if nxt_idx != -1:
                    distance_comps += 1
                    dist_starprime = dist(cidx, nxt_idx)
                    if alpha * dist_starprime <= nxt_dist:
                        unique_pairs[i] = (-1, nxt_dist)
        return set(new_nbhs), distance_comps

    def add_neighbors_without_repeats(existing_neighbors, new_cands):
        s = set(existing_neighbors)
        result = []
        for c in new_cands:
            if c not in s:
                result.append(c)
        return result

    # maybe add single_batch random edges
    if single_batch != 0:
        degree = single_batch
        print("using single batch w/ random start edges, deg =", degree)
        for i in range(n):
            for _ in range(degree):
                rand_neigh = random.randrange(n)
                graph[i].add(rand_neigh)

    inserts = list(range(n))
    if random_order:
        random.shuffle(inserts)

    def beam_search(q_idx, start):
        # a silly linear beam search for demonstration
        # real beam search is fancier. 
        visited = set()
        frontier = [start]
        visited.add(start)
        while frontier:
            next_frontier = []
            # pick up to beam_size best
            frontier_dist = [(f, dist(q_idx, f)) for f in frontier]
            frontier_dist.sort(key=lambda x: x[1])
            frontier_dist = frontier_dist[:beam_size]
            new_front = []
            for node_idx, node_d in frontier_dist:
                # examine neighbors
                for ngh in graph[node_idx]:
                    if ngh not in visited:
                        visited.add(ngh)
                        new_front.append(ngh)
            frontier = new_front
        return visited, 0  # ignoring distance comps for brevity

    # build index
    print("building graph, n =", n)
    start_point = 0  # let's just fix a single start node
    for pass_i in range(num_passes):
        # last pass uses alpha, earlier uses alpha=1.0
        pass_alpha = alpha if pass_i == num_passes - 1 else 1.0
        print("pass", pass_i + 1, "alpha =", pass_alpha)
        count = 0
        inc = 0
        m = len(inserts)
        progress_inc = 0.1
        frac = 0.0
        max_batch_size = min(int(max_fraction * n), 1000000)
        if max_batch_size <= 0:
            max_batch_size = n
        i_ptr = 0
        while i_ptr < m:
            if single_batch == 0:
                # figure out batch
                batch_floor = int(pow(base, inc)) - 1 if pow(base, inc) - 1 < m else i_ptr
                batch_ceil = int(pow(base, inc + 1)) - 1 if pow(base, inc + 1) - 1 < m else min(i_ptr + max_batch_size, m)
                if batch_floor < 0: 
                    batch_floor = 0
                if batch_floor < i_ptr:
                    batch_floor = i_ptr
                if batch_ceil < batch_floor:
                    batch_ceil = batch_floor
                i_ptr = batch_ceil
            else:
                # single batch mode lumps everything
                batch_floor = 0
                batch_ceil = m
                i_ptr = m

            new_out = [[] for _ in range(batch_ceil - batch_floor)]
            # do beam search + robust prune
            for j in range(batch_floor, batch_ceil):
                idx = inserts[j]
                sp = j if single_batch else start_point
                visited, _ = beam_search(idx, sp)
                new_neighbors, _ = robust_prune(idx, visited, add=False)
                new_out[j - batch_floor] = new_neighbors

            for j in range(batch_floor, batch_ceil):
                idx = inserts[j]
                graph[idx] = new_out[j - batch_floor]

            # add bidirectional edges
            edges_to_add = defaultdict(list)
            for j in range(batch_floor, batch_ceil):
                idx = inserts[j]
                for ngh in new_out[j - batch_floor]:
                    edges_to_add[ngh].append(idx)
            # for each node in edges_to_add, either add them or do robust prune if it gets too big
            for node_idx, cands in edges_to_add.items():
                newsize = len(cands) + len(graph[node_idx])
                if newsize <= r:
                    # just add them
                    c = add_neighbors_without_repeats(graph[node_idx], cands)
                    graph[node_idx].update(c)
                else:
                    # robust prune
                    all_cands = list(cands)
                    all_cands.extend(graph[node_idx])
                    pruned, _ = robust_prune(node_idx, all_cands, add=False)
                    graph[node_idx] = pruned

            inc += 1

    # convert sets -> sorted lists
    final_graph = {}
    for i in range(n):
        # optional: sort by distance if you want a knn style ordering
        final_graph[i] = sorted(list(graph[i]), key=lambda x: dist(i, x))
    return final_graph

# usage:
# points = np.random.rand(1000, 16)
# g = build_knn_graph(points, r=16, alpha=1.2, num_passes=2, single_batch=0)
# g is {i: [neighbors...], ...}
if __name__ == "__main__":
    from utils import generate_gaussian_points
    
    points = generate_gaussian_points(1000, 16, 0)
    g = build_knn_graph(points, r=16, alpha=2, num_passes=1, single_batch=0)
    
    print(np.mean([len(v) for v in g.values()]))