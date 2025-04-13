import os
import math
import random
import numpy as np
import networkx as nx
import networkx.generators.trees as trees
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import geoopt  # make sure to install with: pip install geoopt

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
# 2. Tree Generation and Target Distances
# --------------------------
def generate_tree_and_distances(N, seed=42):
    """
    Generate a random tree with N nodes and compute pairwise shortest path distances.
    Returns:
        G: a NetworkX tree graph.
        target_dists: (N x N) tensor of graph (shortest path) distances.
    """
    G = trees.random_labeled_rooted_tree(N, seed=seed)
    target_dists = np.zeros((N, N))
    for i in range(N):
        lengths = nx.single_source_shortest_path_length(G, i)
        for j in range(N):
            target_dists[i, j] = lengths[j]
    return G, torch.tensor(target_dists, dtype=torch.float32)

N = 50  # Number of nodes in the tree
tree, graph_dists = generate_tree_and_distances(N)

# For a target similarity, we can apply the same hyperbolic_g to the graph distances.
# Optionally, one may normalize distances; here we use the raw ones.
I_target = hyperbolic_g(graph_dists)
# I_target now is a (N x N) tensor with values in (0, 1].

# --------------------------
# 3. Initial Embeddings in the Poincaré Disc
# --------------------------
def hierarchical_tree_init(tree, root=None, radius=0.9):
    """
    Hierarchical initialization of tree nodes in the Poincaré disc.
    Places nodes according to their depth in the tree, with the root near the center.
    
    Args:
        tree: A NetworkX tree graph
        root: The root node (if None, selects a central node)
        radius: Maximum radius (should be < 1.0)
    
    Returns:
        A tensor of shape [N, 2] with initial positions
    """
    N = len(tree)
    if root is None:
        # Use a central node as root (e.g., by betweenness centrality)
        centrality = nx.betweenness_centrality(tree)
        root = max(centrality, key=centrality.get)
    
    # Get distances from root
    distances = nx.single_source_shortest_path_length(tree, root)
    max_depth = max(distances.values())
    
    # Create mapping of node ID to sequential index
    node_to_idx = {node: i for i, node in enumerate(tree.nodes())}
    
    # Initialize embeddings
    embeddings = torch.zeros((N, 2))
    
    # Place root at a small random position near origin
    root_idx = node_to_idx[root]
    embeddings[root_idx] = torch.tensor([0.01, 0.01]) * (torch.rand(2) - 0.5)
    
    # Place nodes layer by layer
    for depth in range(1, max_depth + 1):
        # Get nodes at current depth
        layer_nodes = [n for n, d in distances.items() if d == depth]
        
        # Calculate radius for this layer (increases with depth)
        # Use a non-linear scaling to better utilize the space
        layer_radius = radius * (1 - torch.exp(torch.tensor(-depth / max_depth * 3.0))).item()
        
        # Position each node in this layer
        for i, node in enumerate(layer_nodes):
            idx = node_to_idx[node]
            
            # Compute parents of this node (should be only one in a tree)
            parents = [p for p in tree.neighbors(node) if distances[p] < depth]
            if not parents:
                # Fallback if no parent is found
                parent_pos = torch.zeros(2)
            else:
                parent = parents[0]
                parent_idx = node_to_idx[parent]
                parent_pos = embeddings[parent_idx]
            
            # Place the node at a position that:
            # 1. Is at the layer's radius from origin
            # 2. Has some relationship to parent's position
            # 3. Has some randomness to spread siblings
            angle = 2 * math.pi * (i / len(layer_nodes) + 0.1 * torch.rand(1).item())
            
            # Add slight perturbation based on parent position
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

# Replace the random initialization with hierarchical initialization
z0 = hierarchical_tree_init(tree, radius=0.8)  # shape: (N, 2)

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
        
        # Additionally, one may re-normalize to be inside the disc (if needed).
        # Here we simply clip the norm at 0.999.
        norms = torch.norm(z_final, p=2, dim=1, keepdim=True)
        mask = norms >= 0.999
        if mask.any():
            # Use a non-inplace operation
            z_final = torch.where(mask, z_final / norms * 0.999, z_final)
        return z_final


# --------------------------
# 5. Training Loop for Hyperbolic Kuramoto Tree Embedding
# --------------------------
# Hyperparameters for the dynamics and training
dt = 0.05           # time step
steps = 100         # number of integration steps per epoch
max_epochs = 20_000  # maximum number of epochs to run
target_loss = 0.01   # stop training when loss reaches this threshold
patience = 500      # early stopping patience (stop if loss doesn't improve for this many epochs)
lr = 1e-3
lambda_reg = 1.0    # weight for regular regularization
border_penalty = 5.0  # additional penalty for border proximity

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate the model and move it to the appropriate device
model = HyperbolicKuramoto(N).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Initialize z0 on the Poincaré Ball manifold and move to device
z0_fixed = z0.clone().detach().to(device)  # fixed initial state
I_target = I_target.to(device)  # Move target similarity matrix to device as well

loss_history = []
best_loss = float('inf')
no_improve_count = 0

epoch = 0
while True:
    optimizer.zero_grad()
    # Obtain final embeddings by integrating the dynamics
    z_final = model(z0_fixed, dt, steps)  # shape: (N, 2)
    # Compute the loss with additional border penalty
    loss = loss_hyperbolic(z_final, I_target, lambda_reg=lambda_reg, border_penalty=border_penalty)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    # Print progress periodically
    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
    
    # Check stopping criteria
    if loss.item() < target_loss:
        print(f"Target loss reached: {loss.item():.4f} < {target_loss}")
        break
        
    # Early stopping logic
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

# --------------------------
# 6. Visualization of Embedding and Loss
# --------------------------
# Plot the embedding on the Poincaré disc.
z_final_np = z_final.detach().cpu().numpy()
fig, ax = plt.subplots(1, 3, figsize=(21, 6))

# (a) Original tree visualization
pos = nx.kamada_kawai_layout(tree)  # or nx.spring_layout(tree)
nx.draw(tree, pos, ax=ax[0], with_labels=True, node_color='lightblue', 
        node_size=500, font_size=10, font_weight='bold')
ax[0].set_title("Original Tree Structure")

def draw_geodesic_grid(ax, num_circles=5):
    # Similar to the implementation in plotter.py
    for r in np.linspace(0.2, 0.8, num_circles):
        circle = plt.Circle((0, 0), r, fill=False, color='lightgray', linestyle='--', alpha=0.4)
        ax.add_patch(circle)
# Call this before plotting points
draw_geodesic_grid(ax[1])

# (b) Embedding plot.
ax[1].set_aspect("equal")
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
ax[1].add_artist(circle)
ax[1].scatter(z_final_np[:, 0], z_final_np[:, 1], color='blue', s=80)

# Draw geodesics between connected nodes in the hyperbolic embedding
for edge in tree.edges():
    i, j = edge
    # Get the coordinates of the two connected nodes
    p1 = z_final_np[i]
    p2 = z_final_np[j]
    
    # Convert points to tensors for geoopt operations
    p1_tensor = torch.tensor(p1, dtype=torch.float32)
    p2_tensor = torch.tensor(p2, dtype=torch.float32)
    
    # Check if points are very close to boundary and adjust slightly if needed
    for pt in [p1_tensor, p2_tensor]:
        norm_sq = torch.sum(pt ** 2)
        if norm_sq > 0.98:  # very close to boundary
            pt.data = pt.data * (0.98 / torch.sqrt(norm_sq))
    
    # Use more points for geodesic calculation when nodes are near the boundary
    num_points = 150 if (torch.sum(p1_tensor**2) > 0.9 or torch.sum(p2_tensor**2) > 0.9) else 100
    t = torch.linspace(0, 1, num_points)[:, None]
    
    try:
        # Use the stable geodesic function
        geodesic_points = manifold.geodesic(t, p1_tensor, p2_tensor)
        geodesic_points = geodesic_points.detach().cpu().numpy()
        
        # Plot the geodesic curve
        ax[1].plot(geodesic_points[:, 0], geodesic_points[:, 1], 
                  color='gray', linewidth=1.0, alpha=0.7, zorder=1)
    except Exception as e:
        # Fallback to straight line if geodesic calculation fails
        print(f"Warning: Geodesic calculation failed for edge {i}-{j}. Using straight line.")
        ax[1].plot([p1[0], p2[0]], [p2[1], p2[1]], color='red', linestyle='--', linewidth=0.5, alpha=0.5)

# Add node labels to the embedding plot
for i in range(N):
    ax[1].text(z_final_np[i, 0], z_final_np[i, 1], str(i), 
               fontsize=9, ha='center', va='center', 
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

ax[1].set_title("Hyperbolic Kuramoto Tree Embedding")
ax[1].set_xlabel("Dimension 1")
ax[1].set_ylabel("Dimension 2")
ax[1].set_xlim([-1.1, 1.1])
ax[1].set_ylim([-1.1, 1.1])

# (c) Loss curve.
ax[2].plot(loss_history, color='red')
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Loss")
ax[2].set_title("Training Loss")
plt.tight_layout()
plt.savefig("embeds.png")
plt.show()