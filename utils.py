# %%
import matplotlib.pyplot as plt
import numpy as np

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

    return fig
# %%
if __name__ == '__main__':
    # k-nn graph example
    rng = np.random.default_rng()
    X = rng.normal(size=(50, 2))
    edges = [np.argsort([np.linalg.norm(X[x] - X[i]) for i in range(50)])[1:6] for x in range(50)]
    
    fig = plot_2d_graph(X, edges)
    plt.show()
    