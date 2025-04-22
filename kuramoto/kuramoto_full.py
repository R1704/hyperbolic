import os
import math
import random
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rich import print

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import geoopt  # pip install geoopt
import imageio.v2 as imageio # For GIF generation
import sklearn.metrics as skm

# allow importing your expression tree utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyperbolic_graph_embedding.data.tree import ExpressionGenerator, ExpressionVisualizer, generate_and_visualize

# --------------------------
# 1. Hyperbolic Helper & Loss Functions
# --------------------------

manifold = geoopt.PoincareBall(c=1.0)

def hyperbolic_g(x):
    """Similarity function based on hyperbolic distance."""
    return 2 * torch.exp(-x) / (1 + torch.exp(-x))

def mse_similarity_loss(z, I_target):
    """MSE on similarity g(d_hyp) vs. I_target."""
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    G = hyperbolic_g(D)
    mse = F.mse_loss(G, I_target, reduction='sum')
    return mse

def stress_loss(z, D_target):
    """Stress (MDS) loss: sum_{i<j}(d_hyp(i,j) - D_target(i,j))^2."""
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    # Ensure D_target is on the same device
    D_target = D_target.to(z.device) 
    diff = D - D_target
    # Sum over all pairs (implicitly includes i<j twice), divide by 2
    stress = torch.sum(diff.pow(2)) / 2 
    return stress

def smooth_binary_loss(z, edge_index, epsilon, tau=50.0):
    """Soft threshold loss using sigmoid."""
    N = z.size(0)
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    # build masks
    edge_mask = torch.zeros_like(D, dtype=torch.bool, device=z.device)
    edge_mask[edge_index[0], edge_index[1]] = True
    edge_mask[edge_index[1], edge_index[0]] = True # Ensure symmetry
    non_edge_mask = ~edge_mask & ~torch.eye(N, dtype=torch.bool, device=z.device)
    
    # Calculate positive and negative losses
    pos = torch.sigmoid(tau * (D - epsilon))[edge_mask].sum()
    neg = torch.sigmoid(tau * (epsilon - D))[non_edge_mask].sum()
    return pos + neg

def info_nce_loss(z, edge_index, tau=0.1):
    """Contrastive InfoNCE loss with stabilization."""
    N = z.size(0)
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    logits = -D / tau
    # Clamp logits to prevent extreme values leading to NaN/Inf in exp
    logits = torch.clamp(logits, min=-30, max=30) 
    exp_logits = torch.exp(logits)
    
    loss = 0.0
    count = 0
    
    # Ensure edge_index is on the correct device
    edge_index = edge_index.to(z.device) 
    
    for i, j in zip(edge_index[0], edge_index[1]):
        numerator = exp_logits[i, j]
        # Denominator: sum over k != i. Add epsilon for stability.
        # Ensure exp_logits[i,i] is subtracted correctly
        denom = exp_logits[i].sum() - exp_logits[i, i] + 1e-8 
        
        # Add epsilon inside log as well
        log_prob = -torch.log(numerator / denom + 1e-8)
        
        # Check for NaN/Inf before adding to loss
        if not torch.isnan(log_prob) and not torch.isinf(log_prob):
            loss += log_prob
            count += 1
            
    # Return average loss as a tensor
    return loss / count if count > 0 else torch.tensor(0.0, device=z.device, requires_grad=True)

def triplet_loss(z, edge_index, margin=1.0):
    """Triplet loss with random negative sampling."""
    N = z.size(0)
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    
    # Ensure edge_index is on the correct device
    edge_index = edge_index.to(z.device)
    
    # Build adjacency set using integer indices
    edges = set((int(i), int(j)) for i, j in zip(edge_index[0], edge_index[1]))
    # Add symmetric edges for lookup
    edges.update((int(j), int(i)) for i, j in zip(edge_index[0], edge_index[1])) 
    
    loss = 0.0
    count = 0
    
    # Iterate through unique edges (anchor, positive)
    unique_edges = set((min(int(i),int(j)), max(int(i),int(j))) for i,j in zip(edge_index[0], edge_index[1]))

    for i, j in unique_edges:
        # Sample a random k such that (i, k) is not an edge and k != i, k != j
        negs = [k for k in range(N) if k != i and k != j and (i, k) not in edges]
        if not negs: continue
        
        k = random.choice(negs)
        
        # Calculate triplet loss: max(0, d(anchor, positive) - d(anchor, negative) + margin)
        triplet_loss_val = F.relu(D[i, j] - D[i, k] + margin)
        
        # Check for NaN/Inf before adding
        if not torch.isnan(triplet_loss_val) and not torch.isinf(triplet_loss_val):
            loss += triplet_loss_val
            count += 1
            
    # Return average loss as a tensor
    return loss / count if count > 0 else torch.tensor(0.0, device=z.device, requires_grad=True)

def nt_xent_loss(z, edge_index, temperature=0.1, normalize=False):
    """NT-Xent loss with balanced attractive and repulsive forces to prevent collapse.
    
    Args:
        z: Node embeddings tensor of shape [N, D]
        edge_index: Edge tensor of shape [2, E]
        temperature: Temperature scaling parameter (lower = sharper contrasts)
        normalize: Whether to L2 normalize embeddings (typically not needed in hyperbolic space)
    
    Returns:
        Loss value as a tensor
    """
    N = z.size(0)
    
    # In hyperbolic space, normalization can push points to the boundary
    # Only normalize if explicitly requested
    if normalize:
        z = F.normalize(z, p=2, dim=1)
    
    # Compute hyperbolic distance matrix
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    
    # Convert distance to similarity with appropriate scaling
    sim = -D / temperature
    # Use tighter clamping to prevent extreme values
    sim = torch.clamp(sim, min=-20.0, max=20.0)
    
    # Ensure edge_index is on the correct device
    edge_index = edge_index.to(z.device)
    
    # Create adjacency matrix and masks
    adj_mask = torch.zeros((N, N), dtype=torch.bool, device=z.device)
    adj_mask[edge_index[0], edge_index[1]] = True
    adj_mask[edge_index[1], edge_index[0]] = True  # Ensure symmetry
    
    # Remove self-loops
    self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
    adj_mask.masked_fill_(self_mask, False)
    
    # Separate tracking of attractive and repulsive terms to balance them
    attractive_loss = 0.0
    repulsive_loss = 0.0
    valid_nodes = 0
    
    # Process each node that has at least one connection
    for i in range(N):
        # Find positive pairs (connected nodes)
        pos_indices = torch.where(adj_mask[i])[0]
        if len(pos_indices) == 0:
            continue
            
        # Find negative pairs (unconnected nodes, excluding self)
        neg_mask = ~adj_mask[i] & ~self_mask[i]
        neg_indices = torch.where(neg_mask)[0]
        if len(neg_indices) == 0:
            continue
            
        valid_nodes += 1
        
        # Compute positives term: pull connected nodes closer
        pos_exp_sum = torch.sum(torch.exp(sim[i, pos_indices]))
        
        # Compute negatives term: push unconnected nodes apart
        all_exp_sum = torch.sum(torch.exp(sim[i])) - torch.exp(sim[i, i])
        
        # InfoNCE for this node (with numerical stability)
        node_loss = -torch.log((pos_exp_sum / (all_exp_sum + 1e-8)) + 1e-8)
        
        if not torch.isnan(node_loss) and not torch.isinf(node_loss):
            attractive_loss += node_loss
        
        # Add explicit repulsion term to prevent collapse
        # This encourages unconnected nodes to maintain distance
        dist_neg = D[i, neg_indices]
        # Use margin-based repulsion: we want negatives to be at least some margin away
        margin = 2.0  # Target minimum distance for unconnected nodes
        repel_term = torch.mean(F.relu(margin - dist_neg))
        
        if not torch.isnan(repel_term) and not torch.isinf(repel_term):
            repulsive_loss += repel_term
    
    # Balance attractive and repulsive forces - prevent collapse or explosion
    if valid_nodes > 0:
        loss = (attractive_loss + 0.5 * repulsive_loss) / valid_nodes
        
        # Add small penalty for points too close to boundary
        norms = z.norm(dim=1)
        boundary_penalty = 0.1 * torch.mean(norms**4)  # Quartic penalty grows sharply near boundary
        loss = loss + boundary_penalty
        
        return loss
    else:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    



def hybrid_loss(z, D_target, edge_index, epoch=0, max_epochs=8000, alpha=0.5, temperature=0.1):
    """Hybrid loss combining stress and NT-Xent losses with adaptive weighting.
    
    Args:
        z: Node embeddings tensor of shape [N, D]
        D_target: Target distance matrix from the original graph
        edge_index: Edge tensor of shape [2, E]
        epoch: Current training epoch (for dynamic weight adjustment)
        max_epochs: Maximum training epochs (for dynamic weight adjustment)
        alpha: Base weight coefficient for stress vs NT-Xent balance
        temperature: Temperature parameter for NT-Xent loss
        
    Returns:
        Combined loss value as a tensor with gradient
    """
    # Calculate individual loss components
    stress = stress_loss(z, D_target)
    nt_xent = nt_xent_loss(z, edge_index, temperature=temperature)
    
    # Dynamic weighting: gradually shift focus from stability (NT-Xent) to accuracy (stress)
    # Start with more NT-Xent weight and gradually increase stress weight
    progress = min(1.0, epoch / (max_epochs * 0.7))  # Transition in first 70% of training
    stress_weight = alpha + (1 - alpha) * progress
    nt_xent_weight = 1.0 - 0.5 * progress
    
    # Weight depth-based adjustment: prioritize closer nodes in stress calculation
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    # Stronger weight for preserving shorter distances (hierarchical relationships)
    weights = 1.0 / (D_target + 1.0)  
    depth_weighted_stress = torch.sum(weights * (D - D_target).pow(2)) / 2
    
    # Calculate final weighted loss
    weighted_loss = (
        stress_weight * depth_weighted_stress + 
        nt_xent_weight * nt_xent
    )
    
    # Add boundary penalty
    norms = z.norm(dim=1)
    # Softer penalty in early epochs, stronger later
    boundary_penalty_strength = 0.05 + 0.1 * progress
    boundary_penalty = boundary_penalty_strength * torch.mean(norms**4)
    
    return weighted_loss + boundary_penalty

    
# --------------------------
# 2. Evaluation Metrics
# --------------------------

def threshold_accuracy(z, edge_index, eps):
    """Binary accuracy, precision, recall, F1, ROC AUC at threshold eps."""
    N = z.size(0)
    # Ensure edge_index is on CPU for numpy operations
    edge_index_cpu = edge_index.cpu()
    z_cpu = z.cpu() # Ensure z is on CPU for distance calculation if needed by manifold
    
    # Calculate distances on CPU
    with torch.no_grad():
        D = manifold.dist(z_cpu.unsqueeze(1), z_cpu.unsqueeze(0)).numpy()
    
    # Create binary adjacency matrix (1 for edges, 0 for non-edges)
    labels = np.zeros((N, N), dtype=int)
    for i, j in zip(edge_index_cpu[0], edge_index_cpu[1]):
        i, j = int(i), int(j)
        if i < N and j < N: # Bounds check
            labels[i, j] = 1
            labels[j, i] = 1
    
    # Exclude self-loops for evaluation
    mask = ~np.eye(N, dtype=bool)
    labels_flat = labels[mask]
    
    # Predictions based on distance threshold
    preds = (D < eps).astype(int)
    preds_flat = preds[mask]
    
    # Calculate metrics
    acc = (preds_flat == labels_flat).mean()
    prec = skm.precision_score(labels_flat, preds_flat, zero_division=0)
    rec = skm.recall_score(labels_flat, preds_flat, zero_division=0)
    f1 = skm.f1_score(labels_flat, preds_flat, zero_division=0)
    
    # For ROC AUC, use scores (negative distance)
    scores = -D[mask]
    try:
        # Ensure there are both positive and negative samples for AUC
        if len(np.unique(labels_flat)) > 1:
            auc = skm.roc_auc_score(labels_flat, scores)
        else:
            auc = 0.5 # Undefined if only one class present
    except ValueError:
        auc = 0.5  # Default value if calculation fails
    
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)

def stress_metric(z, D_target):
    """Normalized stress metric."""
    # Ensure tensors are on the same device
    D_target = D_target.to(z.device)
    with torch.no_grad():
        D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
        num = torch.sum((D - D_target).pow(2))
        den = torch.sum(D_target.pow(2)) + 1e-8 # Add epsilon for stability
    return (num / den).item()

def mean_relative_distortion(z, D_target):
    """Mean relative distortion metric."""
    # Ensure tensors are on the same device
    D_target = D_target.to(z.device)
    with torch.no_grad():
        D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
        # Add epsilon to avoid division by zero for zero target distances
        rel = torch.abs(D / (D_target + 1e-8) - 1.0)
        # Exclude pairs where D_target is zero if necessary (or handle appropriately)
        mask = D_target > 1e-7 
        mean_rel = torch.mean(rel[mask]) if mask.any() else 0.0
    return float(mean_rel)

# --------------------------
# 3. Möbius Centering
# --------------------------

def mobius_centering(z, root_idx):
    """Apply Möbius transformation to center the root node at the origin."""
    if z.shape[0] <= root_idx:
         print(f"Warning: root_idx {root_idx} out of bounds for z shape {z.shape}. Using index 0.")
         root_idx = 0
         
    z_c = torch.complex(z[:, 0], z[:, 1])
    a = z_c[root_idx]
    a_conj = torch.conj(a)
    # Transformation: (z - a) / (1 - ā z)
    z_new = (z_c - a) / (1 - a_conj * z_c + 1e-8) # Epsilon for stability
    return torch.stack([torch.real(z_new), torch.imag(z_new)], dim=1)

# --------------------------
# 4. Expression Tree + Init
# --------------------------

def generate_expression_tree_and_distances(max_depth=6, variables=None):
    """Generates expression tree and computes graph distances."""
    print("Generating expression tree…")
    expr, tree = generate_and_visualize(max_depth=max_depth, variables=variables,
                                        save_path="expression_tree.png", show=False)
    G = tree.to_undirected()
    N = len(tree.nodes())
    # Ensure nodes are correctly indexed if they are strings like "node_i"
    node_list = list(tree.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    D = np.full((N, N), float(N)) # Initialize with large distance
    np.fill_diagonal(D, 0) # Distance to self is 0

    for i in range(N):
        source_node = node_list[i]
        try:
            lengths = nx.single_source_shortest_path_length(G, source_node)
            for target_node, length in lengths.items():
                if target_node in node_to_idx:
                    j = node_to_idx[target_node]
                    D[i, j] = length
        except nx.NetworkXError:
             print(f"Warning: Node {source_node} not found in undirected graph G.")

    return tree, torch.tensor(D, dtype=torch.float32)

def hierarchical_tree_init(tree, root=None, radius=0.9):
    """Initializes node positions hierarchically in the Poincaré disk."""
    if root is None:
        # Find root (node with in-degree 0)
        roots = [n for n, d in tree.in_degree() if d == 0]
        if roots:
            root = roots[0]
        else: # Fallback if no clear root (e.g., cycle or disconnected)
             print("Warning: No root node found (in-degree 0). Using node with highest betweenness centrality.")
             # Calculate centrality on the undirected version for robustness
             G_undirected = tree.to_undirected()
             if not nx.is_connected(G_undirected):
                 print("Warning: Graph is disconnected. Centrality might be misleading.")
                 # Pick the root from the largest connected component
                 largest_cc = max(nx.connected_components(G_undirected), key=len)
                 subgraph = G_undirected.subgraph(largest_cc)
                 root = max(nx.betweenness_centrality(subgraph).items(), key=lambda x: x[1])[0]
             else:
                 root = max(nx.betweenness_centrality(G_undirected).items(), key=lambda x: x[1])[0]

    G = tree.to_undirected()
    try:
        dist = nx.single_source_shortest_path_length(G, root)
        maxd = max(dist.values()) if dist else 0
    except nx.NetworkXError:
        print(f"Error: Root node '{root}' not found in graph for distance calculation. Using default init.")
        # Fallback to random initialization if root is invalid
        N = len(tree.nodes())
        return manifold.random_uniform((N, 2), max_norm=radius * 0.5) 

    N = len(tree.nodes())
    node_list = list(tree.nodes())
    idx = {n: i for i, n in enumerate(node_list)}
    z = torch.zeros(N, 2)

    # Place root near center with slight perturbation
    if root in idx:
        z[idx[root]] = 0.01 * (torch.rand(2) - 0.5)
    else:
        print(f"Warning: Root node '{root}' not in node index map.")
        # Place the first node near center as fallback
        z[0] = 0.01 * (torch.rand(2) - 0.5) 

    # Group nodes by depth
    bydepth = {}
    for n, d in dist.items():
        bydepth.setdefault(d, []).append(n)

    # Place nodes layer by layer
    for d, nodes in bydepth.items():
        if d == 0: continue
        # Calculate radius for this layer
        layer_radius = radius * (1 - math.exp(-d / maxd * 3.0)) if maxd > 0 else radius * 0.5

        for i, n in enumerate(nodes):
            if n not in idx: continue # Skip if node somehow not indexed
            pi = idx[n]
            
            # Find parent position
            parents = list(tree.predecessors(n))
            parent_pos = torch.zeros(2)
            if parents and parents[0] in idx:
                parent_pos = z[idx[parents[0]]]
            
            # Calculate angle
            angle = 2 * math.pi * (i / len(nodes) + 0.1 * random.random())
            # Adjust angle based on parent angle
            if parent_pos.norm() > 0.01:
                parent_ang = math.atan2(parent_pos[1], parent_pos[0])
                angle = 0.7 * parent_ang + 0.3 * angle
                
            z[pi] = torch.tensor([layer_radius * math.cos(angle), layer_radius * math.sin(angle)])

    # Clamp points to be strictly inside the disk
    norms = z.norm(dim=1, keepdim=True)
    z = torch.where(norms >= 0.995, z / norms * 0.995, z) # Use 0.995 for safety margin
    return z

# --------------------------
# 5. Kuramoto Module (with Node-Specific Frequencies)
# --------------------------

class HyperbolicKuramoto(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        # Learnable coupling matrix K
        self.K = nn.Parameter(torch.randn(N, N) * 0.1) 
        # Learnable node-specific frequencies omega_i (vector of size N)
        self.omega = nn.Parameter(torch.randn(N) * 0.1) 

    def forward(self, z, dt, steps):
        # Ensure z is on the correct device
        z = z.to(self.K.device)
        
        z_c = torch.complex(z[:, 0], z[:, 1])
        Kc = torch.complex(self.K, torch.zeros_like(self.K))
        # Ensure omega is complex for broadcasting, shape (N,)
        omega_c = torch.complex(self.omega, torch.zeros_like(self.omega)) 
        
        for _ in range(steps):
            cz = torch.conj(z_c)
            S = Kc @ cz # shape (N,)
            T = Kc @ z_c # shape (N,)
            
            # Apply omega_i element-wise
            term1 = -(1 / (2 * self.N)) * S * (z_c ** 2)
            term2 = omega_c * z_c # Element-wise multiplication
            term3 = (1 / (2 * self.N)) * T
            
            dz = term1 + term2 + term3
            z_c = z_c + dt * dz
            
            # Projection back to disk
            absz = torch.abs(z_c)
            mask = absz >= 0.999
            if mask.any():
                # Ensure division happens correctly for complex numbers
                # Add epsilon to avoid division by zero if absz is exactly 0 (unlikely but safe)
                z_c = torch.where(mask, z_c / (absz + 1e-8) * 0.999, z_c) 
        
        z_f = torch.stack([torch.real(z_c), torch.imag(z_c)], -1)
        # Final check (redundant if projection inside loop works, but safe)
        norms = z_f.norm(dim=1, keepdim=True)
        return torch.where(norms >= 0.999, z_f / (norms + 1e-8) * 0.999, z_f)

# --------------------------
# 6. Training Loop (with Universal Regularization & GIF History)
# --------------------------

def train(tree, D_target, loss_type='mse', epochs=10_000, lambda_reg=1.0, lr=1e-3, 
          create_gif=False, gif_interval=50):
    """Trains the Hyperbolic Kuramoto model."""
    N = len(tree.nodes())
    z0 = hierarchical_tree_init(tree, radius=0.8)
    I_target = hyperbolic_g(D_target) # Precompute target similarity if needed by loss
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HyperbolicKuramoto(N).to(device)
    # Use RiemannianAdam for potentially better optimization on manifold parameters
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr) 
    
    # Move initial state and targets to device
    z0 = z0.to(device)
    I_target = I_target.to(device)
    D_target = D_target.to(device)
    
    # Create node mapping and edge index
    node_list = list(tree.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edge_list = []
    for u, v in tree.to_undirected().edges():
         if u in node_to_idx and v in node_to_idx: # Ensure nodes exist
             edge_list.append([node_to_idx[u], node_to_idx[v]])
    edge_index = torch.tensor(edge_list).t().to(device)
    
    loss_hist = []
    embeddings_history = [] # Store embeddings for GIF

    # Extract labels and colors for GIF function if needed
    labels = {node: data.get('label', str(node)) for node, data in tree.nodes(data=True)}
    color_map = {
        'variable': '#4AA8FF', 'constant': '#50C878', 'function': '#FF6347', 
        'operation': '#FFD700', 'other': '#A9A9A9'
    }
    node_types = [tree.nodes[node].get('type', 'other') for node in node_list]
    node_colors = [color_map.get(t, '#A9A9A9') for t in node_types]

    for ep in range(epochs):
        optimizer.zero_grad()
        # Run the dynamics simulation
        zf = model(z0, dt=0.05, steps=100) 
        
        # Calculate the primary loss based on type
        if loss_type == 'mse':
            loss = mse_similarity_loss(zf, I_target)
        elif loss_type == 'stress':
            loss = stress_loss(zf, D_target)
        elif loss_type == 'binary':
            loss = smooth_binary_loss(zf, edge_index, epsilon=1.0)
        elif loss_type == 'contrastive':
            loss = info_nce_loss(zf, edge_index, tau=0.1)
        elif loss_type == 'triplet':
            loss = triplet_loss(zf, edge_index, margin=1.0)
        elif loss_type == 'nt_xent':
            loss = nt_xent_loss(zf, edge_index, temperature=0.1)
        elif loss_type == 'hybrid':
            # Use the new hybrid loss, pass current epoch for adaptive weighting
            loss = hybrid_loss(zf, D_target, edge_index, epoch=ep, max_epochs=epochs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        # Add universal boundary regularization (already included in hybrid loss)
        if loss_type != 'hybrid':
            norm_sq = zf.pow(2).sum(-1)
            reg_loss = -lambda_reg * torch.log(1 - norm_sq + 1e-8).sum() 
            total_loss = loss + reg_loss
        else:
            # For hybrid loss, we already included the boundary regularization
            total_loss = loss
            reg_loss = torch.tensor(0.0)  # Placeholder for logging

        # Add universal boundary regularization
        norm_sq = zf.pow(2).sum(-1)
        # Use a small epsilon to prevent log(0)
        reg_loss = -lambda_reg * torch.log(1 - norm_sq + 1e-8).sum() 
        
        total_loss = loss + reg_loss # Combine primary loss and regularization

        # Backpropagate the total loss
        total_loss.backward() 
        
        # Gradient clipping (optional but can help stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        loss_hist.append(total_loss.item()) # Log the total loss
        
        # Logging
        if ep % 500 == 0 or ep == epochs - 1:
            print(f"[{loss_type}] Epoch {ep}: Base Loss={loss.item():.4f}, Reg Loss={reg_loss.item():.4f}, Total Loss={total_loss.item():.4f}") 
            
        # Store embedding state for GIF
        if create_gif and (ep % gif_interval == 0 or ep == epochs - 1):
            # Center before saving for consistent visualization
            root_candidates = [node for node, in_deg in tree.in_degree() if in_deg == 0]
            root = root_candidates[0] if root_candidates else node_list[0]
            root_idx = node_to_idx.get(root, 0) # Get index safely
            
            # Detach and move to CPU before centering and converting to numpy
            z_final_cpu = zf.detach().cpu()
            z_centered_np = mobius_centering(z_final_cpu, root_idx).numpy()
            embeddings_history.append(z_centered_np)
            
        # Early stopping condition (optional)
        if total_loss.item() < 1e-3: # Adjusted threshold
             print(f"Early stopping at epoch {ep} with total loss {total_loss.item():.4f}")
             # Ensure the last frame is added if stopping early
             if create_gif and ep % gif_interval != 0:
                 z_final_cpu = zf.detach().cpu()
                 z_centered_np = mobius_centering(z_final_cpu, node_to_idx.get(root, 0)).numpy()
                 embeddings_history.append(z_centered_np)
             break 
             
    # After the loop, create GIF if requested
    if create_gif and embeddings_history:
        # Pass necessary info to GIF creator
        create_embedding_gif(
            tree=tree, 
            embeddings_history=embeddings_history, 
            filename=f"training_dynamics_{loss_type}.gif", 
            node_to_idx=node_to_idx, 
            labels=labels, 
            node_colors=node_colors,
            interval=gif_interval # Pass interval for correct epoch numbering
        )

    return model, zf.detach().cpu(), loss_hist

# --------------------------
# 7. Visualization & GIF Creation
# --------------------------

def visualize(tree, z_final, loss_hist, D_target=None, loss_type='unknown'):
    """Visualizes the original tree, embedding, and loss curve."""
    fig, ax = plt.subplots(1, 3, figsize=(24, 7)) # Wider figure
    
    # --- Get node attributes ---
    node_list = list(tree.nodes())
    labels = {node: data.get('label', str(node)) for node, data in tree.nodes(data=True)}
    node_types = [tree.nodes[node].get('type', 'other') for node in node_list]
    color_map = {
        'variable': '#4AA8FF', 'constant': '#50C878', 'function': '#FF6347', 
        'operation': '#FFD700', 'other': '#A9A9A9'
    }
    node_colors = [color_map.get(t, '#A9A9A9') for t in node_types]
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # --- (a) Original tree visualization ---
    try:
        # Use graphviz layout for better hierarchy
        pos = nx.nx_agraph.graphviz_layout(tree, prog='dot') 
        nx.draw(tree, pos, ax=ax[0], with_labels=True, labels=labels, 
                node_color=node_colors, node_size=800, font_size=10,
                font_weight='bold', arrows=True, arrowstyle='->', arrowsize=15,
                edge_color='gray')
        ax[0].set_title("Original Expression Tree")
    except ImportError:
        print("Warning: pygraphviz not found. Using default NetworkX layout for original tree.")
        pos = nx.spring_layout(tree) # Fallback layout
        nx.draw(tree, pos, ax=ax[0], with_labels=True, labels=labels, 
                node_color=node_colors, node_size=800, font_size=10,
                font_weight='bold', arrows=True, arrowstyle='->', arrowsize=15,
                edge_color='gray')
        ax[0].set_title("Original Expression Tree (Spring Layout)")
    ax[0].axis('off') # Hide axes for tree plot

    # --- (b) Plot embedding in the Poincaré disc ---
    # Center the embedding
    root_candidates = [node for node, in_deg in tree.in_degree() if in_deg == 0]
    root = root_candidates[0] if root_candidates else node_list[0]
    root_idx = node_to_idx.get(root, 0)
    
    z_centered = mobius_centering(z_final, root_idx)
    z_np = z_centered.numpy()
    
    # Draw unit circle and geodesic grid
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
    ax[1].add_artist(circle)
    for r in np.linspace(0.2, 0.8, 4):
        ax[1].add_patch(plt.Circle((0, 0), r, fill=False, color='lightgray', linestyle='--', alpha=0.4))
        
    # Draw geodesics between connected nodes
    for u, v in tree.to_undirected().edges():
        if u not in node_to_idx or v not in node_to_idx: continue # Skip if node missing
        i, j = node_to_idx[u], node_to_idx[v]
        p1, p2 = z_np[i], z_np[j]
        
        # Convert to tensor for geodesic calculation
        p1_tensor = torch.tensor(p1, dtype=torch.float32)
        p2_tensor = torch.tensor(p2, dtype=torch.float32)
        
        # Check norms and adjust points slightly inward if exactly on boundary
        norm1 = torch.norm(p1_tensor)
        norm2 = torch.norm(p2_tensor)
        if norm1 >= 1.0: p1_tensor *= 0.999 / norm1
        if norm2 >= 1.0: p2_tensor *= 0.999 / norm2
            
        num_points = 100
        if norm1 > 0.95 or norm2 > 0.95: num_points = 150 # More points near boundary
            
        try:
            t = torch.linspace(0, 1, num_points)[:, None]
            # Ensure geodesic calculation uses points on the manifold
            geodesic = manifold.geodesic(t, p1_tensor, p2_tensor) 
            geodesic_np = geodesic.detach().numpy()
            ax[1].plot(geodesic_np[:, 0], geodesic_np[:, 1], 
                       color='darkgray', linewidth=0.8, alpha=0.7, zorder=1)
        except Exception as e:
            # Fallback to straight line if geodesic fails
            # print(f"Warning: Geodesic failed for edge ({u},{v}). Using straight line. Error: {e}")
            ax[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       color='lightcoral', linestyle=':', linewidth=0.6, alpha=0.6)
    
    # Plot nodes
    ax[1].scatter(z_np[:, 0], z_np[:, 1], c=node_colors, s=150, edgecolor='black', linewidth=0.5, zorder=10, alpha=0.9)
    # Add text labels
    for i, node in enumerate(node_list):
        label = labels.get(node, '?')
        ax[1].text(z_np[i, 0], z_np[i, 1], label, 
                   fontsize=9, ha='center', va='center', color='black',
                   bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2'),
                   zorder=20) 
    
    ax[1].set_aspect('equal')
    ax[1].set_title(f"Hyperbolic Embedding ({loss_type.upper()})")
    ax[1].set_xlim([-1.1, 1.1]); ax[1].set_ylim([-1.1, 1.1])
    ax[1].axis('off') # Hide axes for embedding plot

    # --- (c) Loss curve ---
    ax[2].plot(loss_hist, color='crimson', linewidth=1.5)
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Total Loss")
    ax[2].set_title("Training Loss Curve")
    ax[2].grid(True, linestyle='--', alpha=0.5)
    ax[2].set_yscale('log') # Use log scale for loss if it varies widely
    
    plt.tight_layout(pad=1.5)
    # Save figure with loss type in filename
    save_filename = f"embedding_result_{loss_type}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight') 
    print(f"Visualization saved to {save_filename}")
    plt.show()
    
    # --- Calculate and print metrics ---
    print(f"\n--- Evaluation Metrics ({loss_type.upper()}) ---")
    if D_target is not None:
        # Ensure D_target is on CPU for metric calculation if z_final is
        D_target_cpu = D_target.cpu() 
        print(f"Stress metric: {stress_metric(z_final, D_target_cpu):.4f}")
        print(f"Mean relative distortion: {mean_relative_distortion(z_final, D_target_cpu):.4f}")
    else:
        print("D_target not provided, skipping stress and distortion metrics.")
    
    # Calculate accuracy metrics
    # Recreate edge_index on CPU for metric function
    edge_index_cpu = torch.tensor([[node_to_idx[u], node_to_idx[v]] 
                                   for u, v in tree.to_undirected().edges() 
                                   if u in node_to_idx and v in node_to_idx]).t()
    
    if edge_index_cpu.numel() > 0: # Check if there are edges
        for eps in [0.5, 1.0, 1.5, 2.0]: # Added more thresholds
            metrics = threshold_accuracy(z_final, edge_index_cpu, eps)
            print(f"\nLink Prediction Metrics @ eps={eps}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
    else:
        print("No edges found to calculate link prediction metrics.")
    print("-" * (28 + len(loss_type)))


def create_embedding_gif(tree, embeddings_history, filename, node_to_idx, labels, node_colors, interval):
    """Creates a GIF from the embedding history."""
    print(f"Creating GIF: {filename}...")
    frames = []
    temp_dir = "temp_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    node_list = list(tree.nodes()) # Ensure consistent node order

    for idx, z_np in enumerate(embeddings_history):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw unit circle and grid
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, ls='--', alpha=0.7)
        ax.add_artist(circle)
        for r in np.linspace(0.2, 0.8, 4):
             ax.add_patch(plt.Circle((0, 0), r, fill=False, color='lightgray', ls='--', alpha=0.4))

        # Draw simple lines for edges (faster for GIF)
        for u, v in tree.to_undirected().edges():
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                # Check bounds for safety
                if i < len(z_np) and j < len(z_np):
                    ax.plot([z_np[i, 0], z_np[j, 0]], [z_np[i, 1], z_np[j, 1]], 
                            color='darkgray', lw=0.6, alpha=0.7) 

        # Draw nodes
        ax.scatter(z_np[:, 0], z_np[:, 1], c=node_colors, s=120, edgecolor='black', lw=0.5, zorder=10, alpha=0.9)
        # Add text labels
        for i, node in enumerate(node_list):
             if i < len(z_np): # Check bounds
                 label = labels.get(node, '?')
                 ax.text(z_np[i, 0], z_np[i, 1], label, fontsize=9, ha='center', va='center', 
                         bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2'), zorder=20)

        ax.set_aspect('equal')
        ax.set_title(f"Epoch {idx * interval}") # Display correct epoch number
        ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1])
        ax.axis('off') # Clean frame

        frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        # Append frame data directly
        try:
             frames.append(imageio.imread(frame_path))
        except FileNotFoundError:
             print(f"Warning: Frame file not found: {frame_path}")


    # Save GIF if frames were generated
    if frames:
        try:
            imageio.mimsave(filename, frames, duration=0.15, loop=0) # Adjust duration, loop=0 for infinite loop
            print(f"GIF saved to {filename}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    else:
        print("No frames generated for GIF.")
        
    # Clean up temporary frames
    try:
        for frame_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, frame_file))
        os.rmdir(temp_dir)
    except OSError as e:
        print(f"Error cleaning up temp frames: {e}")

# --------------------------
# 8. Main Execution
# --------------------------

if __name__ == "__main__":
    # --- Configuration ---
    MAX_DEPTH = 5       # Depth of the expression tree
    VARIABLES = ['x', 'y'] # Variables in the expression
    EPOCHS = 5_000       # Max training epochs per loss type
    LEARNING_RATE = 5e-4 # Adjusted learning rate
    LAMBDA_REG = 0.5     # Regularization strength (tune this)
    CREATE_GIF = True   # Set to True to generate GIFs
    GIF_INTERVAL = 100   # Save frame every 100 epochs for GIF
    
    # --- Generate Tree ---
    tree, graph_dists = generate_expression_tree_and_distances(
        max_depth=MAX_DEPTH, variables=VARIABLES
    )
    print(f"Generated Tree - Nodes: {len(tree.nodes())}, Edges: {len(tree.edges())}")
    
    # --- Train and Visualize for each loss type ---
    # loss_types_to_run = ['mse', 'stress', 'binary', 'contrastive', 'triplet', 'nt_xent']
    loss_types_to_run = ['hybrid']
    
    
    for lt in loss_types_to_run:
        print(f"\n{'='*15} Training with {lt.upper()} Loss {'='*15}")
        try:
            model, zf, lh = train(
                tree, 
                graph_dists, 
                loss_type=lt, 
                epochs=EPOCHS, 
                lambda_reg=LAMBDA_REG, 
                lr=LEARNING_RATE,
                create_gif=CREATE_GIF,
                gif_interval=GIF_INTERVAL
            )
            
            print(f"\n--- Final Results for {lt.upper()} Loss ---")
            visualize(
                tree, 
                zf, 
                lh, 
                graph_dists, 
                loss_type=lt
            )
        except Exception as e:
            print(f"\n!!!!!! ERROR during training/visualization for {lt.upper()} loss !!!!!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            print(f"!!!!!! Skipping {lt.upper()} loss due to error. !!!!!!")

    print("\nAll training runs completed.")