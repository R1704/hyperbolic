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

# allow importing your expression tree utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyperbolic_graph_embedding.data.tree import ExpressionGenerator, ExpressionVisualizer, generate_and_visualize

# --------------------------
# 1. Hyperbolic Helper & Loss Functions
# --------------------------

manifold = geoopt.PoincareBall(c=1.0)

def hyperbolic_g(x):
    return 2 * torch.exp(-x) / (1 + torch.exp(-x))

def mse_similarity_loss(z, I_target, lambda_reg=1.0):
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    G = hyperbolic_g(D)
    mse = F.mse_loss(G, I_target, reduction='sum')
    norm_sq = z.pow(2).sum(-1)
    reg = -lambda_reg * torch.log(1 - norm_sq).sum()
    return mse + reg

def stress_loss(z, D_target, lambda_reg=0.0):
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    diff = D - D_target
    s = diff.pow(2).sum() / 2
    if lambda_reg>0:
        norm_sq = z.pow(2).sum(-1)
        s += -lambda_reg * torch.log(1 - norm_sq).sum()
    return s

def smooth_binary_loss(z, edge_index, epsilon, tau=50.0):
    N = z.size(0)
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    # build masks
    edge_mask = torch.zeros_like(D, dtype=torch.bool)
    edge_mask[edge_index[0], edge_index[1]] = True
    edge_mask[edge_index[1], edge_index[0]] = True
    non_edge_mask = ~edge_mask & ~torch.eye(N, dtype=torch.bool, device=z.device)
    pos = torch.sigmoid(tau*(D - epsilon))[edge_mask].sum()
    neg = torch.sigmoid(tau*(epsilon - D))[non_edge_mask].sum()
    return pos + neg

def info_nce_loss(z, edge_index, tau=0.1):
    N = z.size(0)
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    logits = -D / tau
    exp_logits = torch.exp(logits)
    loss = 0.0; count=0
    for i,j in zip(edge_index[0], edge_index[1]):
        numerator = exp_logits[i,j]
        denom = exp_logits[i].sum() - torch.exp(logits[i,i])
        loss += -torch.log(numerator/denom)
        count+=1
    return loss/count if count>0 else loss

def triplet_loss(z, edge_index, margin=1.0):
    N = z.size(0)
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    edges = set((int(i),int(j)) for i,j in zip(edge_index[0], edge_index[1]))
    loss=0.0; count=0
    for i,j in edges:
        negs = [k for k in range(N) if (i,k) not in edges and k!=i]
        if not negs: continue
        k = random.choice(negs)
        loss += F.relu(D[i,j] - D[i,k] + margin)
        count+=1
    return loss/count if count>0 else loss

# --------------------------
# 2. Evaluation Metrics
# --------------------------

import sklearn.metrics as skm

def threshold_accuracy(z, edge_index, eps):
    N = z.size(0)
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0)).cpu().numpy()
    
    # Create binary adjacency matrix (1 for edges, 0 for non-edges)
    labels = np.zeros((N, N), dtype=int)
    for i, j in zip(edge_index[0].cpu(), edge_index[1].cpu()):
        i, j = int(i), int(j)
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
    
    # For ROC AUC, we need scores (higher = more likely to be an edge)
    scores = -D[mask]  # Negative distance (closer = higher score)
    try:
        auc = skm.roc_auc_score(labels_flat, scores)
    except:
        auc = 0.5  # Default value if calculation fails
    
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)

def stress_metric(z, D_target):
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    num = torch.sum((D-D_target).pow(2))
    den = torch.sum(D_target.pow(2))
    return (num/den).item()

def mean_relative_distortion(z, D_target):
    D = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    rel = torch.abs(D/(D_target+1e-8) - 1.0)
    return float(rel.mean())

# --------------------------
# 3. Möbius Centering
# --------------------------

def mobius_centering(z, root_idx):
    z_c = torch.complex(z[:,0], z[:,1])
    a   = z_c[root_idx]
    ā  = torch.conj(a)
    z_new = (z_c - a) / (1 - ā*z_c + 1e-8)
    return torch.stack([torch.real(z_new), torch.imag(z_new)], dim=1)

# --------------------------
# 4. Expression Tree + Init
# --------------------------

def generate_expression_tree_and_distances(max_depth=6, variables=None):
    print("Generating expression tree…")
    expr, tree = generate_and_visualize(max_depth=max_depth, variables=variables,
                                        save_path="expression_tree.png", show=False)
    G = tree.to_undirected()
    N = len(tree.nodes())
    D = np.zeros((N,N))
    for i in range(N):
        lengths = nx.single_source_shortest_path_length(G, f"node_{i}")
        for j in range(N):
            D[i,j] = lengths.get(f"node_{j}", N)
    return tree, torch.tensor(D, dtype=torch.float32)

def hierarchical_tree_init(tree, root=None, radius=0.9):
    if root is None:
        roots = [n for n,d in tree.in_degree() if d==0]
        root = roots[0] if roots else max(nx.betweenness_centrality(tree).items(), key=lambda x:x[1])[0]
    G=tree.to_undirected()
    dist = nx.single_source_shortest_path_length(G, root)
    maxd = max(dist.values())
    N = len(tree.nodes())
    idx = {n:i for i,n in enumerate(tree.nodes())}
    z = torch.zeros(N,2)
    # root at center perturb
    z[idx[root]] = 0.01*(torch.rand(2)-0.5)
    bydepth = {}
    for n,d in dist.items(): bydepth.setdefault(d,[]).append(n)
    for d,nodes in bydepth.items():
        if d==0: continue
        r = radius*(1-math.exp(-d/maxd*3))
        for i,n in enumerate(nodes):
            pi = idx[n]
            parents = list(tree.predecessors(n))
            parent_pos = z[idx[parents[0]]] if parents else torch.zeros(2)
            angle = 2*math.pi*(i/len(nodes)+0.1*random.random())
            if parent_pos.norm()>0.01:
                parent_ang = math.atan2(parent_pos[1],parent_pos[0])
                angle = 0.7*parent_ang + 0.3*angle
            z[pi] = torch.tensor([r*math.cos(angle), r*math.sin(angle)])
    # clamp
    norms = z.norm(dim=1,keepdim=True)
    z = torch.where(norms>=0.99, z/norms*0.99, z)
    return z

# --------------------------
# 5. Kuramoto Module
# --------------------------

class HyperbolicKuramoto(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.K = nn.Parameter(torch.randn(N,N)*0.1)
        self.omega = nn.Parameter(torch.tensor(0.1))
    def forward(self, z, dt, steps):
        z_c = torch.complex(z[:,0], z[:,1])
        Kc  = torch.complex(self.K, torch.zeros_like(self.K))
        for _ in range(steps):
            cz = torch.conj(z_c)
            S  = Kc @ cz
            t1 = -(1/(2*self.N))*S*z_c**2
            t2 = self.omega*z_c
            t3 = (1/(2*self.N))*(Kc@z_c)
            z_c = z_c + dt*(t1+t2+t3)
            absz = torch.abs(z_c)
            mask = absz>=0.999
            if mask.any():
                z_c = torch.where(mask, z_c/absz*0.999, z_c)
        z_f = torch.stack([torch.real(z_c), torch.imag(z_c)],-1)
        norms = z_f.norm(dim=1,keepdim=True)
        return torch.where(norms>=0.999, z_f/norms*0.999, z_f)

# --------------------------
# 6. Training Loop
# --------------------------

def train(tree, D_target, loss_type='mse', epochs=10_000):
    N = len(tree.nodes())
    z0 = hierarchical_tree_init(tree, radius=0.8)
    I_target = hyperbolic_g(D_target)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HyperbolicKuramoto(N).to(device)
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=1e-3)
    z0 = z0.to(device); I_target=I_target.to(device); D_target=D_target.to(device)
    
    # Create a mapping from node names to indices
    node_to_idx = {node: i for i, node in enumerate(tree.nodes())}
    
    # Convert edge list to tensor of integer indices
    edge_list = []
    for u, v in tree.to_undirected().edges():
        edge_list.append([node_to_idx[u], node_to_idx[v]])
    
    edge_index = torch.tensor(edge_list).t().to(device)
    
    loss_hist=[]
    for ep in range(epochs):
        optimizer.zero_grad()
        zf = model(z0, dt=0.05, steps=100)
        if loss_type=='mse':
            loss = mse_similarity_loss(zf, I_target)
        elif loss_type=='stress':
            loss = stress_loss(zf, D_target)
        elif loss_type=='binary':
            loss = smooth_binary_loss(zf, edge_index, epsilon=0.5)
        elif loss_type=='contrastive':
            loss = info_nce_loss(zf, edge_index)
        elif loss_type=='triplet':
            loss = triplet_loss(zf, edge_index)
        else:
            raise ValueError(loss_type)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        if ep%500==0:
            print(f"[{loss_type}] Epoch {ep}: loss={loss.item():.4f}")
        # early stop
        if loss.item()<1e-2: break
    return model, zf.detach().cpu(), loss_hist

# --------------------------
# 7. Visualization
# --------------------------

def visualize(tree, z_final, loss_hist, D_target=None, loss_type='unknown'):
    fig, ax = plt.subplots(1, 3, figsize=(22, 7))  # Changed to 3 subplots
    
    # Get node attributes for better visualization
    labels = {node: data.get('label', str(node)) for node, data in tree.nodes(data=True)}
    node_types = [data.get('type', 'other') for _, data in tree.nodes(data=True)]
    
    # Define colors for different node types with stronger saturation
    color_map = {
        'variable': '#4AA8FF',  # Stronger blue
        'constant': '#50C878',  # Stronger green
        'function': '#FF6347',  # Stronger coral/red
        'operation': '#FFD700',  # Stronger gold/yellow
        'other': '#A9A9A9'      # Stronger gray
    }
    
    node_colors = [color_map.get(t, '#A9A9A9') for t in node_types]
    
    # (a) Original tree visualization
    pos = nx.nx_agraph.graphviz_layout(tree, prog='dot')  # Hierarchical layout
    nx.draw(tree, pos, ax=ax[0], with_labels=True, labels=labels, 
            node_color=node_colors, node_size=700, font_size=9,
            font_weight='bold', arrows=True, arrowstyle='->', arrowsize=15)
    ax[0].set_title("Original Expression Tree")
    
    # (b) Plot embedding in the Poincaré disc
    # Center the embedding with root at origin
    root_candidates = [node for node, in_deg in tree.in_degree() if in_deg == 0]
    root = root_candidates[0] if root_candidates else list(tree.nodes())[0]
    root_idx = list(tree.nodes()).index(root)
    
    z_centered = mobius_centering(z_final, root_idx)
    z_np = z_centered.numpy()
    
    # Draw unit circle and geodesic grid
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
    ax[1].add_artist(circle)
    
    # Draw geodesic grid (concentric circles)
    for r in np.linspace(0.2, 0.8, 4):
        circle = plt.Circle((0, 0), r, fill=False, color='lightgray', linestyle='--', alpha=0.4)
        ax[1].add_patch(circle)
        
    # Create a mapping from node names to indices for the embedding array
    node_to_idx = {node: i for i, node in enumerate(tree.nodes())}
    
    # Draw geodesics between connected nodes
    for u, v in tree.to_undirected().edges():
        i, j = node_to_idx[u], node_to_idx[v]
        
        # Get coordinates
        p1 = z_np[i]
        p2 = z_np[j]
        
        # Convert to tensor for geodesic calculation
        p1_tensor = torch.tensor(p1, dtype=torch.float32)
        p2_tensor = torch.tensor(p2, dtype=torch.float32)
        
        # Use more points for geodesic when nodes are near boundary
        norm1, norm2 = np.linalg.norm(p1), np.linalg.norm(p2)
        num_points = 100
        if norm1 > 0.9 or norm2 > 0.9:
            num_points = 150
            
        # Calculate geodesic
        try:
            t = torch.linspace(0, 1, num_points)[:, None]
            geodesic = manifold.geodesic(t, p1_tensor, p2_tensor)
            geodesic_np = geodesic.numpy()
            
            # Plot the geodesic curve
            ax[1].plot(geodesic_np[:, 0], geodesic_np[:, 1], 
                       color='gray', linewidth=0.7, alpha=0.8, zorder=1)
        except Exception as e:
            # Fallback to straight line
            ax[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot nodes with improved readability
    for i, node in enumerate(tree.nodes()):
        label = labels.get(node, str(i))
        node_type = tree.nodes[node].get('type', 'other')
        color = color_map.get(node_type, '#A9A9A9')
        
        # Stronger colored nodes with black edge for contrast
        ax[1].scatter(z_np[i, 0], z_np[i, 1], color=color, s=120, edgecolor='black', linewidth=0.5, zorder=10)
        
        # Text directly on node without box border
        ax[1].text(z_np[i, 0], z_np[i, 1], label, 
                   fontsize=10, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'),
                   zorder=20)  # Higher zorder to be on top
    
    ax[1].set_aspect('equal')
    ax[1].set_title("Hyperbolic Embedding (Centered)")
    ax[1].set_xlim([-1.1, 1.1])
    ax[1].set_ylim([-1.1, 1.1])
    
    # (c) Loss curve
    ax[2].plot(loss_hist, color='red')
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Loss")
    ax[2].set_title("Training Loss")
    ax[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"embedding_result_{loss_type}.png", dpi=300)  # Add loss_type to filename    plt.show()
    
    # Calculate and print metrics
    edge_index = torch.zeros((2, len(tree.edges())), dtype=torch.long)
    for i, (u, v) in enumerate(tree.to_undirected().edges()):
        edge_index[0, i] = node_to_idx[u]
        edge_index[1, i] = node_to_idx[v]
    
    print("\nEvaluation Metrics:")
    if D_target is not None:
        print(f"Stress metric: {stress_metric(z_final, D_target):.4f}")
        print(f"Mean relative distortion: {mean_relative_distortion(z_final, D_target):.4f}")
    else:
        print("D_target not provided, skipping stress and distortion metrics")
    
    # Calculate accuracy metrics at different thresholds
    for eps in [0.5, 1.0, 1.5]:
        metrics = threshold_accuracy(z_final, edge_index, eps)
        print(f"\nAccuracy metrics @ eps={eps}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

if __name__=="__main__":
    tree, graph_dists = generate_expression_tree_and_distances(max_depth=6, variables=['x','y'])
    print(f"Nodes: {len(tree.nodes())}, Edges: {len(tree.edges())}")
    
    for lt in ['mse','stress','binary','contrastive','triplet']:
        model, zf, lh = train(tree, graph_dists, loss_type=lt, epochs=5000)
        print(f"\n=== Results for {lt} loss ===")
        visualize(tree, zf, lh, graph_dists, loss_type=lt)  # Pass loss_type