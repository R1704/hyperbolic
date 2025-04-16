import os
import math
import random
import numpy as np
from rich import print
import networkx as nx
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import geoopt  # make sure to install with: pip install geoopt

# Add the parent directory to sys.path to import from hyperbolic_graph_embedding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyperbolic_graph_embedding.data.tree import ExpressionGenerator, ExpressionVisualizer, generate_and_visualize

# --------------------------
# 1. Hyperbolic helper functions
# --------------------------
def hyperbolic_g(x):
    """
    A similarity mapping: for hyperbolic distance x,
    returns a value close to 1 for small x and approaching 0 for large x.
    """
    return 2 * torch.exp(-x) / (1 + torch.exp(-x))

# We'll create a geoopt manifold instance for the Poincare Ball
manifold = geoopt.PoincareBall(c=1.0)

def loss_hyperbolic(z, I, lambda_reg=1.0, border_penalty=5.0):
    """
    z: (N x 2) tensor representing points in the Poincaré disc.
    I: (N x N) target similarity matrix (derived from the tree)
    lambda_reg: regularization weight for the border penalty.
    border_penalty: additional penalty for points near the border (>0.9)
    """
    # Calculate distances and similarity matrix
    D_hyp = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))  # shape (N, N)
    G = hyperbolic_g(D_hyp)
    loss_data = torch.sum((I - G) ** 2)
    
    # Standard regularization to avoid points at the boundary
    norm_sq = torch.sum(z ** 2, dim=1)
    loss_reg = -lambda_reg * torch.sum(torch.log(1 - norm_sq))
    
    # Additional penalty for points too close to the border
    border_mask = norm_sq > 0.81  # Consider points with |z|² > 0.81 (|z| > 0.9) too close to border
    if border_mask.any():
        border_loss = border_penalty * torch.sum((norm_sq[border_mask] - 0.81) / (1 - norm_sq[border_mask]))
    else:
        border_loss = 0.0
        
    return loss_data + loss_reg + border_loss

# --------------------------
# 2. Expression Tree Generation and Target Distances
# --------------------------
def generate_expression_tree_and_distances(max_depth=6, variables=None):
    """
    Generate a random symbolic expression tree and compute pairwise distances.
    Returns:
        G: a NetworkX directed graph representing the expression tree.
        target_dists: (N x N) tensor of graph (shortest path) distances.
    """
    print("Generating a symbolic expression tree...")
    expr, tree = generate_and_visualize(max_depth=max_depth, variables=variables, 
                                        save_path="expression_tree.png", show=False)
    
    # Convert the directed tree to undirected for distance calculation
    G_undirected = tree.to_undirected()
    
    N = len(tree.nodes())
    target_dists = np.zeros((N, N))
    
    # Calculate shortest path distances between all nodes
    for i in range(N):
        node_i = f"node_{i}"
        if node_i in G_undirected:
            lengths = nx.single_source_shortest_path_length(G_undirected, node_i)
            for j in range(N):
                node_j = f"node_{j}"
                if node_j in lengths:
                    target_dists[i, j] = lengths[node_j]
                else:
                    # Handle disconnected nodes (should be rare in a tree)
                    target_dists[i, j] = N  # Use a large value for disconnected nodes
    
    return tree, torch.tensor(target_dists, dtype=torch.float32)

# --------------------------
# 3. Initial Embeddings in the Poincaré Disc
# --------------------------
def hierarchical_tree_init(tree, root=None, radius=0.9):
    """
    Initialize tree nodes in the Poincaré disc based on tree hierarchy.
    
    Args:
        tree: A NetworkX directed graph (expression tree)
        root: The root node (if None, finds the root based on in-degree)
        radius: Maximum radius (should be < 1.0)
    
    Returns:
        A tensor of shape [N, 2] with initial positions
    """
    # For directed trees, the root typically has in-degree 0
    if root is None:
        # Find nodes with in-degree 0 (likely the root)
        root_candidates = [node for node, in_deg in tree.in_degree() if in_deg == 0]
        if root_candidates:
            root = root_candidates[0]
        else:
            # Fallback to a central node if no clear root
            centrality = nx.betweenness_centrality(tree.to_undirected())
            root = max(centrality, key=centrality.get)
    
    # Convert to undirected for BFS traversal
    undirected_tree = tree.to_undirected()
    
    # Get distances from root using BFS
    distances = nx.single_source_shortest_path_length(undirected_tree, root)
    max_depth = max(distances.values()) if distances else 0
    
    # Number of nodes
    N = len(tree.nodes())
    
    # Create mapping of node name to sequential index for the embedding tensor
    node_to_idx = {node: i for i, node in enumerate(tree.nodes())}
    
    # Initialize embeddings
    embeddings = torch.zeros((N, 2))
    
    # Place root at a small random position near origin
    root_idx = node_to_idx[root]
    embeddings[root_idx] = torch.tensor([0.01, 0.01]) * (torch.rand(2) - 0.5)
    
    # Group nodes by their depth
    nodes_by_depth = {}
    for node, depth in distances.items():
        if depth not in nodes_by_depth:
            nodes_by_depth[depth] = []
        nodes_by_depth[depth].append(node)
    
    # Place nodes layer by layer
    for depth in range(1, max_depth + 1):
        if depth not in nodes_by_depth:
            continue
            
        layer_nodes = nodes_by_depth[depth]
        
        # Calculate radius for this layer (increases with depth)
        layer_radius = radius * (1 - torch.exp(torch.tensor(-depth / max_depth * 3.0))).item()
        
        # Position each node in this layer
        for i, node in enumerate(layer_nodes):
            idx = node_to_idx[node]
            
            # Find parent in the directed tree
            parents = list(tree.predecessors(node))
            if parents:
                parent = parents[0]
                parent_idx = node_to_idx[parent]
                parent_pos = embeddings[parent_idx]
            else:
                # Fallback if no parent is found
                parent_pos = torch.zeros(2)
            
            # Calculate angle based on position in layer and parent position
            angle = 2 * math.pi * (i / len(layer_nodes) + 0.1 * torch.rand(1).item())
            
            # Adjust angle based on parent position
            if torch.norm(parent_pos) > 0.01:
                parent_angle = torch.atan2(parent_pos[1], parent_pos[0]).item()
                angle = 0.7 * parent_angle + 0.3 * angle
                
            # Set position
            x = layer_radius * math.cos(angle)
            y = layer_radius * math.sin(angle)
            embeddings[idx] = torch.tensor([x, y])
    
    # Final safety check to ensure all points are within the disc
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    mask = norms >= 0.99
    if mask.any():
        embeddings = torch.where(mask, embeddings / norms * 0.99, embeddings)
    
    return embeddings

# --------------------------
# 4. Hyperbolic Kuramoto Module
# --------------------------
class HyperbolicKuramoto(nn.Module):
    def __init__(self, N):
        """
        N: number of nodes (oscillators).
        Learnable parameters:
          - K: coupling matrix of shape (N x N)
          - omega: a scalar natural frequency.
        """
        super(HyperbolicKuramoto, self).__init__()
        self.N = N
        # Use geoopt's parameter to register K and omega
        self.K = nn.Parameter(torch.randn(N, N) * 0.1)
        self.omega = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, z, dt, steps):
        """
        z: initial embeddings, tensor of shape (N, 2) (points in R^2 representing the disc).
        dt: time step for Euler integration.
        steps: number of integration steps.
        Returns: final embeddings z_final (N x 2), updated by the dynamics.
        """
        N = self.N
        # Convert z to a complex representation for the update:
        # Represent a 2D point (x, y) as complex number: x + i y.
        z_complex = torch.complex(z[:, 0], z[:, 1])  # shape: (N,)
        
        # Convert self.K to complex type to enable matmul with complex tensors
        K_complex = torch.complex(self.K, torch.zeros_like(self.K))
        
        # Integrate Euler steps.
        for _ in range(steps):
            # Calculate the coupling term: sum_{k} K_{ik} * conj(z_k)
            conj_z = torch.conj(z_complex)
            S = torch.matmul(K_complex, conj_z)  # (N,)
            term1 = - (1 / (2 * N)) * S * (z_complex ** 2)
            term2 = self.omega * z_complex
            T = torch.matmul(K_complex, z_complex)  # (N,)
            term3 = (1 / (2 * N)) * T
            dz = term1 + term2 + term3
            z_complex = z_complex + dt * dz
            
            # If any |z| > 0.999, project it back to the disc.
            abs_z = torch.abs(z_complex)
            mask = abs_z >= 0.999
            if mask.any():
                # Fix: For complex numbers, the division already handles the scaling properly
                z_complex = torch.where(mask, z_complex / abs_z * 0.999, z_complex)
        
        # Convert final complex numbers back to 2D coordinates.
        z_final = torch.stack([torch.real(z_complex), torch.imag(z_complex)], dim=1)
        
        # Additionally, normalize to be inside the disc
        norms = torch.norm(z_final, p=2, dim=1, keepdim=True)
        mask = norms >= 0.999
        if mask.any():
            # Use a non-inplace operation
            z_final = torch.where(mask, z_final / norms * 0.999, z_final)
        return z_final


# --------------------------
# 5. Training Loop for Hyperbolic Kuramoto Expression Tree Embedding
# --------------------------
def train_expression_tree_embedding(tree, graph_dists, z0=None, max_epochs=10000):
    # Hyperparameters for the dynamics and training
    dt = 0.05           # time step
    steps = 100         # number of integration steps per epoch
    target_loss = 0.01   # stop training when loss reaches this threshold
    patience = 500      # early stopping patience
    lr = 1e-3
    lambda_reg = 1.0    # weight for regularization
    border_penalty = 5.0  # penalty for border proximity

    # Number of nodes
    N = len(tree.nodes())
    
    # Initialize embeddings if not provided
    if z0 is None:
        z0 = hierarchical_tree_init(tree, radius=0.8)
    
    # For target similarity, apply hyperbolic_g to graph distances
    I_target = hyperbolic_g(graph_dists)
    
    # Check if CUDA is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instantiate the model and move to device
    model = HyperbolicKuramoto(N).to(device)
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr)

    # Move tensors to device
    z0_fixed = z0.clone().detach().to(device)
    I_target = I_target.to(device)

    loss_history = []
    best_loss = float('inf')
    no_improve_count = 0

    epoch = 0
    while True:
        optimizer.zero_grad()
        z_final = model(z0_fixed, dt, steps)
        loss = loss_hyperbolic(z_final, I_target, lambda_reg=lambda_reg, border_penalty=border_penalty)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if (epoch+1) % 200 == 0:
            print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
        
        # Check stopping criteria
        if loss.item() < target_loss:
            print(f"Target loss reached: {loss.item():.4f} < {target_loss}")
            break
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs - No improvement for {patience} epochs")
            break
            
        if epoch >= max_epochs - 1:
            print(f"Maximum number of epochs ({max_epochs}) reached")
            break
            
        epoch += 1

    print(f"Training completed after {epoch+1} epochs with final loss: {loss.item():.4f}")
    return z_final.detach().cpu(), loss_history

# --------------------------
# 6. Visualization Functions
# --------------------------
def draw_geodesic_grid(ax, num_circles=5):
    """Draw hyperbolic geodesic grid (concentric circles)"""
    for r in np.linspace(0.2, 0.8, num_circles):
        circle = plt.Circle((0, 0), r, fill=False, color='lightgray', linestyle='--', alpha=0.4)
        ax.add_patch(circle)

def visualize_expression_tree_embedding(tree, z_final_np, loss_history):
    """
    Visualize the original expression tree and its embedding in the Poincaré disc.
    
    Args:
        tree: The expression tree as a NetworkX DiGraph
        z_final_np: Final node embeddings as numpy array
        loss_history: List of loss values during training
    """
    fig, ax = plt.subplots(1, 3, figsize=(24, 7))

    # (a) Original tree visualization with specialized layout
    visualizer = ExpressionVisualizer()
    
    # Extract node attributes for visualization
    pos = nx.nx_agraph.graphviz_layout(tree, prog='dot')
    labels = {node: data.get('label', str(node)) for node, data in tree.nodes(data=True)}
    node_types = [data.get('type', 'other') for _, data in tree.nodes(data=True)]
    
    # Define colors for different node types
    color_map = {
        'variable': 'skyblue',
        'constant': 'lightgreen',
        'function': 'coral',
        'operation': 'gold',
        'other': 'lightgray'
    }
    
    node_colors = [color_map.get(t, 'lightgray') for t in node_types]
    
    # Draw the original tree
    nx.draw(tree, pos, ax=ax[0], with_labels=True, labels=labels, 
            node_color=node_colors, node_size=1800, font_size=10, 
            arrows=True, arrowstyle='->', arrowsize=15)
    
    ax[0].set_title("Original Expression Tree", fontsize=14)

    # (b) Embedding plot in the Poincaré disc
    draw_geodesic_grid(ax[1])
    ax[1].set_aspect("equal")
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
    ax[1].add_artist(circle)
    ax[1].scatter(z_final_np[:, 0], z_final_np[:, 1], color='blue', s=80)

    # Create a map from node ID strings (node_0, node_1, etc.) to indices
    node_to_idx = {node: i for i, node in enumerate(tree.nodes())}
    
    # Draw geodesics between connected nodes in the embedding
    for edge in tree.edges():
        source, target = edge
        # Map node names to indices
        i = node_to_idx[source]
        j = node_to_idx[target]
        
        # Get coordinates
        p1 = z_final_np[i]
        p2 = z_final_np[j]
        
        # Convert to tensors for geodesic calculation
        p1_tensor = torch.tensor(p1, dtype=torch.float32)
        p2_tensor = torch.tensor(p2, dtype=torch.float32)
        
        # Check if points are very close to boundary and adjust
        for pt in [p1_tensor, p2_tensor]:
            norm_sq = torch.sum(pt ** 2)
            if norm_sq > 0.98:
                pt.data = pt.data * (0.98 / torch.sqrt(norm_sq))
        
        # Use more points for geodesic when nodes are near boundary
        num_points = 150 if (torch.sum(p1_tensor**2) > 0.9 or torch.sum(p2_tensor**2) > 0.9) else 100
        t = torch.linspace(0, 1, num_points)[:, None]
        
        try:
            # Calculate geodesic points
            geodesic_points = manifold.geodesic(t, p1_tensor, p2_tensor)
            geodesic_points = geodesic_points.detach().numpy()
            
            # Plot the geodesic curve
            ax[1].plot(geodesic_points[:, 0], geodesic_points[:, 1], 
                      color='gray', linewidth=1.0, alpha=0.7, zorder=1)
        except Exception as e:
            print(f"Warning: Geodesic calculation failed for edge {source}-{target}. Using straight line.")
            ax[1].plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linestyle='--', linewidth=0.5)

    # Add node labels and color coding to embedding plot
    for i, node in enumerate(tree.nodes()):
        node_type = tree.nodes[node].get('type', 'other')
        label = tree.nodes[node].get('label', str(i))
        color = color_map.get(node_type, 'white')
        
        ax[1].text(z_final_np[i, 0], z_final_np[i, 1], label, 
                   fontsize=9, ha='center', va='center', 
                   bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))

    ax[1].set_title("Hyperbolic Kuramoto Expression Tree Embedding", fontsize=14)
    ax[1].set_xlabel("Dimension 1")
    ax[1].set_ylabel("Dimension 2")
    ax[1].set_xlim([-1.1, 1.1])
    ax[1].set_ylim([-1.1, 1.1])

    # (c) Loss curve
    ax[2].plot(loss_history, color='red')
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Loss")
    ax[2].set_title("Training Loss", fontsize=14)
    ax[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("expression_embeddings.png", dpi=300)
    print(f"Visualization saved to expression_embeddings.png")
    plt.show()

# --------------------------
# 7. Main execution
# --------------------------
if __name__ == "__main__":
    # Generate a random expression tree
    max_depth = 4  # Adjust depth as needed
    variables = ['x', 'y', 'z']  # Variables to use in expression
    
    # Generate tree and calculate distances
    tree, graph_dists = generate_expression_tree_and_distances(max_depth=max_depth, variables=variables)
    
    # Print some information about the tree
    print(f"Expression tree has {len(tree.nodes())} nodes and {len(tree.edges())} edges")
    print("\nNode attributes:")
    for node, attrs in list(tree.nodes(data=True))[:5]:  # Show first 5 nodes
        print(f"{node}: {attrs}")
    
    # Initialize embeddings
    z0 = hierarchical_tree_init(tree, radius=0.8)
    
    # Train the embedding
    z_final, loss_history = train_expression_tree_embedding(tree, graph_dists, z0)
    
    # Visualize results
    visualize_expression_tree_embedding(tree, z_final.numpy(), loss_history)