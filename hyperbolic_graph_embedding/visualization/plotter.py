import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
import geoopt
from matplotlib.patches import Circle

class Plotter:
    """
    An advanced visualization tool for 2D embeddings in different geometries.
    Supports both Euclidean and hyperbolic spaces, with proper edge rendering.
    """
    def __init__(self, title="Embedding Visualization"):
        self.title = title

    def plot_embeddings(self, embeddings, graph=None, labels=None, is_hyperbolic=False, 
                        node_labels=None, node_types=None, show_geodesic_grid=True,
                        manifold: geoopt.PoincareBall = None):
        """
        Plot embeddings with proper geometry and graph structure.
        
        Args:
            embeddings: Tensor or array of shape [N, 2] with 2D embedding coordinates.
            graph: NetworkX graph object showing connections between nodes.
            labels: Optional node coloring.
            is_hyperbolic: Whether embeddings are in hyperbolic space.
            node_labels: Optional dict mapping node indices to labels.
            node_types: Optional list of node types for color coding.
            show_geodesic_grid: Whether to show a geodesic grid (for hyperbolic space).
            manifold: If is_hyperbolic is True, a geoopt.PoincareBall instance must be provided.
        """
        # Convert tensor to numpy if necessary.
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        x, y = embeddings[:, 0], embeddings[:, 1]
        
        if is_hyperbolic:
            # Draw the Poincaré disk boundary.
            boundary = Circle((0, 0), 1, fill=False, edgecolor='black', linestyle='-', linewidth=1.5)
            ax.add_patch(boundary)
            
            # Optionally, draw a geodesic grid (if you wish to add extra gridlines).
            if show_geodesic_grid:
                self._draw_geodesic_grid(ax)
                
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
        else:
            # For Euclidean space, set limits based on data range.
            margin = 0.1 * max(np.ptp(x), np.ptp(y))
            ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
            ax.set_ylim(np.min(y) - margin, np.max(y) + margin)
        
        # Color mapping based on node types if provided.
        colors = None
        if node_types:
            color_map = {
                'variable': 'skyblue',
                'constant': 'lightgreen',
                'function': 'coral',
                'operation': 'gold',
                'other': 'lightgray'
            }
            colors = [color_map.get(t, 'lightgray') for t in node_types]
        
        # Draw graph edges if provided.
        if graph is not None:
            node_list = list(graph.nodes())
            node_to_idx = {node: i for i, node in enumerate(node_list)}
            
            for u, v in graph.edges():
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                p1 = embeddings[u_idx]
                p2 = embeddings[v_idx]
                
                if is_hyperbolic:
                    if manifold is None:
                        raise ValueError("For hyperbolic plotting, a PoincareBall manifold instance must be provided.")
                    self._draw_hyperbolic_geodesic(ax, p1, p2, manifold, color='gray', linewidth=1.0, arrow=graph.is_directed())
                else:
                    if graph.is_directed():
                        ax.annotate("", xy=p2, xytext=p1,
                                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))
                    else:
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=1.0, zorder=1)
        
        # Draw nodes.
        if labels is not None:
            scatter = ax.scatter(x, y, c=labels, cmap="viridis", s=100, zorder=2)
            plt.colorbar(scatter)
        else:
            ax.scatter(x, y, color=colors if colors else 'blue', s=100, zorder=2, alpha=0.7)
        
        # Draw node labels if provided.
        if node_labels:
            for i, (x_i, y_i) in enumerate(zip(x, y)):
                if i in node_labels:
                    ax.annotate(node_labels[i], (x_i, y_i), 
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', fontsize=9, fontweight='bold')
        
        ax.set_aspect('equal')
        plt.title(self.title + (" (Hyperbolic Space)" if is_hyperbolic else " (Euclidean Space)"))
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(False)
        plt.tight_layout()
        plt.show()
    
    def _draw_geodesic_grid(self, ax, num_lines=6, line_style='--', line_color='lightgray', line_width=0.5):
        """Draw a simple geodesic grid (horizontal/vertical lines and circles) in the Poincaré disk."""
        # Draw axes through origin.
        ax.axhline(y=0, color=line_color, linestyle=line_style, linewidth=line_width, alpha=0.7)
        ax.axvline(x=0, color=line_color, linestyle=line_style, linewidth=line_width, alpha=0.7)
        # Optionally, add a few concentric circles.
        for r in np.linspace(0.2, 0.8, num_lines):
            circle = Circle((0, 0), r, fill=False, color=line_color, linestyle=line_style, linewidth=line_width, alpha=0.7)
            ax.add_patch(circle)
    
    def _draw_hyperbolic_geodesic(self, ax, p1, p2, manifold: geoopt.PoincareBall, color='gray', linewidth=1.0, arrow=False):
        """
        Draw the geodesic (shortest path) between two points in the Poincaré disk using
        the geodesic interpolation formula:
            gamma(t) = exp_{p1}(t * log_{p1}(p2))
        Args:
            ax: Matplotlib axis.
            p1, p2: Endpoints as numpy arrays of shape (2,).
            manifold: An instance of geoopt.PoincareBall.
            arrow: If True, draw an arrow along the geodesic.
        """
        # Convert points to torch tensors.
        p1_tensor = torch.tensor(p1, dtype=torch.float32)
        p2_tensor = torch.tensor(p2, dtype=torch.float32)
        
        # Compute tangent vector from p1 to p2.
        v = manifold.log_map(p2_tensor, p1_tensor)
        
        # Generate interpolation points along the geodesic.
        ts = torch.linspace(0, 1, steps=100)
        geodesic_points = [manifold.exp_map(t * v, p1_tensor).detach().numpy() for t in ts]
        geodesic_points = np.array(geodesic_points)
        
        # Plot the geodesic arc.
        ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], color=color, linewidth=linewidth, zorder=1)
        
        # Optionally, add an arrow at the midpoint.
        if arrow:
            mid_idx = len(geodesic_points) // 2
            start = geodesic_points[mid_idx]
            end = geodesic_points[mid_idx+1]
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="->", color=color, lw=linewidth))
