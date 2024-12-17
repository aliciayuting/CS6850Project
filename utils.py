# %%
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html

def plot_2d_graph(points, neighborhoods, point_kwargs={}, edge_kwargs={}):
    """Plots a graph over 2d points. Accepts kwargs to be sent to the plotting
    functions.

    Args:
        points (numpy.ndarray): 2d numpy array of the vectors of the points
        neighborhoods (List[List]): A list of lists describing neighborhoods

    Returns:
        A figure containing the visualization
    """
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], **point_kwargs)

    # Plot edges
    for i, edge_list in enumerate(neighborhoods):
        point1 = points[i]
        for neighbor in edge_list:
            point2 = points[neighbor]
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], **edge_kwargs)

    # Remove ticks, gridlines, and borders
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    
    fig.tight_layout()

    return fig

def plot_3d_graph(points, neighborhoods, point_kwargs={}, edge_kwargs={}):
    """Plots a graph over 3d points in 3d space, creating an interactive plotly
    visualization. Works even in jupyter notebooks. Accepts kwargs to be sent 
    to the plotting functions.

    Args:
        points (numpy.ndarray): 2d numpy array of the vectors of the points
        neighborhoods (List[List]): A list of lists describing neighborhoods

    Returns:
        A figure containing the visualization
    """
    fig = go.Figure()

    # add points with hover labels for indices
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        text=[f"Index: {i}" for i in range(points.shape[0])],  # add hover text
        hoverinfo="text",  # show only hover text
        **point_kwargs
    ))

    # add edges
    print("--------- ",len(points), len(neighborhoods))
    for i, neighbors in enumerate(neighborhoods):
        if i >= len(points):
            print(f"Skipping invalid point index: {i}")
            continue
        p1 = points[i]
        for n in neighbors:
            p2 = points[n]
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode='lines',
                hoverinfo='none',  # no hover for edges
                **edge_kwargs
            ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )

    return fig


def generate_gaussian_points(n, d, seed=0):
    """Generates n d-dimensional points from a gaussian distribution.

    Args:
        n (int): The number of points to generate
        d (int): The dimensionality of the points
        seed (int, optional): The seed for the RNG. Defaults to 0.

    Returns:
        numpy.ndarray: A numpy array of shape (n, d) containing the points
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))

def edgelist_to_neighborhoods(edges, undirected=False):
    """Converts a list of edges to a list of neighborhoods.

    Args:
        edges (List[Tuple[int, int]]): A list of edges
        undirected (bool, optional): Whether the graph is undirected. Defaults to False.
        
    Returns:
        List[List[int]]: A list of neighborhoods
    """
    max_node = max(max(e) for e in edges)
    neighborhoods = [[] for _ in range(max_node + 1)]
    
    for a, b in edges:
        neighborhoods[a].append(b)
        if undirected:
            neighborhoods[b].append(a)
            
    # remove duplicates
    for i, neighbors in enumerate(neighborhoods):
        neighborhoods[i] = sorted(set(neighbors))
        
    return neighborhoods

def neighborhoods_to_edgelist(neighborhoods):
    """Converts a list of neighborhoods to a list of edges.

    Args:
        neighborhoods (List[List[int]]): A list of neighborhoods

    Returns:
        List[Tuple[int, int]]: A list of edges
    """
    edges = []
    for i, neighbors in enumerate(neighborhoods):
        for n in neighbors:
            edges.append((i, n))
    return edges

# %%
if __name__ == '__main__':
    # k-nn graph example
    rng = np.random.default_rng()
    X = rng.normal(size=(50, 3))
    edges = [np.argsort([np.linalg.norm(X[x] - X[i]) for i in range(50)])[1:6] for x in range(50)]
    
    fig = plot_3d_graph(X, edges)
    fig.show()
    
    
    