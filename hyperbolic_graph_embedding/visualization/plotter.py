import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
import geoopt
from matplotlib.patches import Circle
import os

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
        Plot embeddings interactively.
        
        (This method remains for on-screen visualization.)
        """
        # Convert tensor to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        x, y = embeddings[:, 0], embeddings[:, 1]
        
        if is_hyperbolic:
            # Draw the Poincaré disk boundary
            boundary = Circle((0, 0), 1, fill=False, edgecolor='black', linestyle='-', linewidth=1.5)
            ax.add_patch(boundary)
            if show_geodesic_grid:
                self._draw_geodesic_grid(ax)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
        else:
            margin = 0.1 * max(np.ptp(x), np.ptp(y))
            ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
            ax.set_ylim(np.min(y) - margin, np.max(y) + margin)
        
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
        
        if labels is not None:
            scatter = ax.scatter(x, y, c=labels, cmap="viridis", s=100, zorder=2)
            plt.colorbar(scatter)
        else:
            ax.scatter(x, y, color=colors if colors else 'blue', s=100, zorder=2, alpha=0.7)
        
        if node_labels:
            for i, (x_i, y_i) in enumerate(zip(x, y)):
                if i in node_labels:
                    ax.annotate(node_labels[i], (x_i, y_i), 
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', fontsize=9, fontweight='bold')
        
        ax.set_aspect('equal')
        ax.set_title(self.title + (" (Hyperbolic Space)" if is_hyperbolic else " (Euclidean Space)"))
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(False)
        plt.tight_layout()
        plt.show()

    def save_frame(self, embeddings, epoch: int, output_dir: str, 
                   graph=None, labels=None, is_hyperbolic=False, 
                   node_labels=None, node_types=None, show_geodesic_grid=True,
                   manifold: geoopt.PoincareBall = None):
        """
        Plot embeddings and save the frame to a file for later animation.
        
        Args:
            embeddings: Tensor or array of shape [N, 2].
            epoch: Current epoch number (used in filename).
            output_dir: Directory to save the frame image.
            (Other parameters are similar to plot_embeddings.)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        x, y = embeddings[:, 0], embeddings[:, 1]
        
        if is_hyperbolic:
            boundary = Circle((0, 0), 1, fill=False, edgecolor='black', linestyle='-', linewidth=1.5)
            ax.add_patch(boundary)
            if show_geodesic_grid:
                self._draw_geodesic_grid(ax)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
        else:
            margin = 0.1 * max(np.ptp(x), np.ptp(y))
            ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
            ax.set_ylim(np.min(y) - margin, np.max(y) + margin)
        
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
        
        if labels is not None:
            scatter = ax.scatter(x, y, c=labels, cmap="viridis", s=100, zorder=2)
            plt.colorbar(scatter)
        else:
            ax.scatter(x, y, color=colors if colors else 'blue', s=100, zorder=2, alpha=0.7)
        
        if node_labels:
            for i, (x_i, y_i) in enumerate(zip(x, y)):
                if i in node_labels:
                    ax.annotate(node_labels[i], (x_i, y_i), 
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', fontsize=9, fontweight='bold')
        
        ax.set_aspect('equal')
        ax.set_title(self.title + (" (Hyperbolic Space)" if is_hyperbolic else " (Euclidean Space)") + f" Epoch {epoch}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(False)
        plt.tight_layout()
        
        # Save the figure to the specified output directory
        filename = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(filename)
        plt.close(fig)
    
    def _draw_geodesic_grid(self, ax, num_lines=6, line_style='--', line_color='lightgray', line_width=0.5):
        """Draw a simple geodesic grid in the Poincaré disk."""
        ax.axhline(y=0, color=line_color, linestyle=line_style, linewidth=line_width, alpha=0.7)
        ax.axvline(x=0, color=line_color, linestyle=line_style, linewidth=line_width, alpha=0.7)
        for r in np.linspace(0.2, 0.8, num_lines):
            circle = Circle((0, 0), r, fill=False, color=line_color, linestyle=line_style, linewidth=line_width, alpha=0.7)
            ax.add_patch(circle)
    
    def _draw_hyperbolic_geodesic(self, ax, p1, p2, manifold: geoopt.PoincareBall, color='gray', linewidth=1.0, arrow=False):
        """
        Draw the hyperbolic geodesic between two points in the Poincaré disk.
        Uses the geodesic interpolation:
            gamma(t) = exp_{p1}(t * log_{p1}(p2))
        """
        p1_tensor = torch.tensor(p1, dtype=torch.float32)
        p2_tensor = torch.tensor(p2, dtype=torch.float32)
        # Compute the tangent vector at p1 pointing toward p2
        v = manifold.log_map(p2_tensor, p1_tensor)
        ts = torch.linspace(0, 1, steps=100)
        geodesic_points = [manifold.exp_map(t * v, p1_tensor).detach().numpy() for t in ts]
        geodesic_points = np.array(geodesic_points)
        ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], color=color, linewidth=linewidth, zorder=1)
        if arrow:
            mid = geodesic_points[len(geodesic_points) // 2]
            mid_next = geodesic_points[len(geodesic_points) // 2 + 1]
            ax.annotate("", xy=mid_next, xytext=mid,
                        arrowprops=dict(arrowstyle="->", color=color, lw=linewidth))

    def create_animation(self, frames_dir, output_path, fps=5, duration=0.2, loop=0):
        """
        Create an animation from saved frames.
        
        Args:
            frames_dir: Directory containing the saved frames
            output_path: Path to save the animation (GIF or MP4)
            fps: Frames per second for the animation
            duration: Duration of each frame in seconds (only for GIF)
            loop: Number of times to loop the GIF (0 = loop forever)
        
        Returns:
            Path to the created animation
        """
        import imageio
        import glob
        from PIL import Image
        
        # Get all image files in the directory
        image_files = sorted(glob.glob(f"{frames_dir}/*.png"))
        
        if not image_files:
            print(f"No image files found in {frames_dir}")
            return None
        
        print(f"Found {len(image_files)} frames")
        
        # Determine output format based on file extension
        _, ext = os.path.splitext(output_path)
        
        # Create a GIF
        if ext.lower() == '.gif':
            print(f"Creating GIF animation at {output_path}")
            with imageio.get_writer(output_path, mode='I', duration=duration, loop=loop) as writer:
                for filename in image_files:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    
        # Create an MP4 video
        elif ext.lower() == '.mp4':
            print(f"Creating MP4 animation at {output_path}")
            
            # For MP4, we often need to ensure consistent image dimensions
            images = [imageio.imread(f) for f in image_files]
            
            # Use imageio's FFMPEG writer for MP4
            writer = imageio.get_writer(output_path, fps=fps)
            for img in images:
                writer.append_data(img)
            writer.close()
            
        else:
            print(f"Unsupported output format: {ext}")
            return None
        
        print(f"Animation saved to {output_path}")
        return output_path
