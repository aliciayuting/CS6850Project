import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def hitting_sets(points):
    n = len(points)
    
    # --- 1) Precompute all pairwise distances ---
    print("Computing pairwise distances...")
    dist = cdist(points, points, metric='euclidean')
    print("done.")
    
    # --- compute the set cover problem for each point ---
    sets = []
    # naive approach
    for x in tqdm(range(n), desc="Computing hitting set sets"):
        S_x = []
        for q in range(n):
            x_dist = dist[x][q]
            S_x_q = [] # points you could connect to that would cover q
            for y in range(n):
                y_dist = dist[y][q]
                if y_dist < x_dist:
                    S_x_q.append(y)
            S_x.append(S_x_q)
        sets.append(S_x)
    return sets

def solve_minimum_edge_cover(points, directed=False, relaxed=False):
    """
    points: list of coordinate tuples, e.g. [(x1,y1), (x2,y2), ...] in 2D
            or higher dimension as needed
    """
    n = len(points)
    
    sets = hitting_sets(points)

    # --- 2) Build the Gurobi model ---
    model = gp.Model("MinimumEdgeCover")

    if relaxed:
        var_type = GRB.CONTINUOUS
    else:
        var_type = GRB.BINARY
    
    z = {}
    for i in range(n):
        for j in range(i+1, n) if not directed else range(n):
            z[(i,j)] = model.addVar(vtype=var_type, 
                                   lb=0, ub=1, 
                                   name=f"z_{i}_{j}")

    # --- 3) Add coverage constraints ---
    # For each pair (x, q), x != q, we need SUM_{y in S_{x,q}} z_{x,y} >= 1,
    # where S_{x,q} = { y != x : dist(x,y) < dist(x,q) }.
    for x, S_x in enumerate(sets):
        for q, S_xq in enumerate(S_x):
            if x == q:
                continue
            
            # If S_{x,q} is empty, then there's no "closer" y to x than q,
            # which could make the model infeasible. Usually, if d(x,y) < d(x,q)
            # never holds, it's an impossible pair to cover. 
            if len(S_xq) == 0:
                print(f"Warning: x={x}, q={q} has no solutions.")
            else:
                if not directed:
                    model.addConstr(
                        gp.quicksum(z[(min(x, y), max(x, y))] for y in S_xq) >= 1,
                        name=f"cover_x{ x }_q{ q }"
                    )
                else:
                    model.addConstr(
                        gp.quicksum(z[(x, y)] for y in S_xq) >= 1,
                        name=f"cover_x{ x }_q{ q }"
                    )
                    

    # --- 4) Objective: minimize sum of z_{ij} ---
    model.setObjective(
        gp.quicksum(z[(i,j)] for i in range(n) for j in list(range(i+1, n) if not directed else range(n))),
        GRB.MINIMIZE
    )

    # --- 5) Solve ---
    model.optimize()

    # --- 6) Extract solution ---
    # For the LP relaxation, z may be fractional. For an ILP, z are {0,1}.
    edges_selected = []
    for i in range(n):
        for j in range(i+1, n) if not directed else range(n):
            if model.status == GRB.OPTIMAL:
                val = z[(i,j)].X
                # if val > some small threshold, interpret as selected
                if val > 1e-5:
                    edges_selected.append((i, j))
    
    objective_value = model.objVal
    
    return edges_selected, objective_value


if __name__ == "__main__":
    from utils import generate_gaussian_points, plot_3d_graph, edgelist_to_neighborhoods, plot_2d_graph
    n, d = 250, 3
    points = generate_gaussian_points(n, d)
    
    edges, objective = solve_minimum_edge_cover(points, directed=False, relaxed=False)
    
    # print(edges)
    
    neighborhoods = edgelist_to_neighborhoods(edges, undirected=True)
    # print(neighborhoods)
    
    # import matplotlib.pyplot as plt
    
    fig = plot_3d_graph(points, neighborhoods)
    # fig = plot_2d_graph(points, neighborhoods, edge_kwargs=dict(linewidth=0.5, color='black'))
    
    # fig.savefig("plots/2d_minimum_cover_250.png", dpi=300)
    html_str = fig.to_html(include_plotlyjs='cdn')
    
    with open("plots/3d_minimum_cover_250.html", "w") as f:
        f.write(html_str)

    
    fig.show()
    
    # # --- 1) Precompute all pairwise distances ---
    # print("Computing pairwise distances...")
    # dist = cdist(points, points, metric='euclidean')
    # print("done.")
    
    # # --- compute the set cover problem for each point ---
    # sets = hitting_sets(points)
    
    # components = [sorted(list(c)) for c in components]
    # print(components)
    
    # x, q = 0, 38
    # print(x, q)
    
    # S_xq = sets[x][q]
    # print(S_xq)
    
    # print([y for y in S_xq if y in neighborhoods[x]])
    
    # print(dist[x][q], dist[37][q])