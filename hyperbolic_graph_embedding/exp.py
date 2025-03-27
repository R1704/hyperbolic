# import torch

# def g(x):
#     # x: a tensor of distances
#     return 2 * torch.exp(-x) / (1 + torch.exp(-x))

# def pairwise_euclidean_distances(u):
#     # u is an (N x d) tensor of embeddings
#     # Using broadcasting for pairwise distance computation
#     diff = u.unsqueeze(1) - u.unsqueeze(0)
#     return torch.norm(diff, dim=2)

# def loss_euclidean(u, I, lambda_reg):
#     # Compute pairwise distances
#     D = pairwise_euclidean_distances(u)
#     # Apply function g elementwise
#     G = g(D)
#     # Compute the squared error loss
#     loss_data = torch.sum((I - G) ** 2)
#     # Regularization term: sum of squared norms
#     loss_reg = lambda_reg * torch.sum(u ** 2)
#     return loss_data + loss_reg

# # Example usage:
# N, d = 100, 2  # e.g., 100 nodes in 2D space
# u = torch.randn(N, d, requires_grad=True)
# I = ...  # an (N x N) binary tensor representing the adjacency matrix
# lambda_reg = 0.1
# optimizer = torch.optim.Adam([u], lr=0.01)

# for epoch in range(1000):
#     optimizer.zero_grad()
#     loss = loss_euclidean(u, I, lambda_reg)
#     loss.backward()
#     optimizer.step()
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")


import torch
import geoopt
from utils import create_adj_matrix
from torch_geometric.datasets import Planetoid, TUDataset
import networkx as nx


# Define the Poincaré ball (curvature c=1, by default)
manifold = geoopt.PoincareBall(c=1.0)

def hyperbolic_g(x):
    return 2 * torch.exp(-x) / (1 + torch.exp(-x))

def loss_hyperbolic(z, I, lambda_reg):
    # z is a (N x 2) tensor on the Poincaré ball.
    # Compute pairwise hyperbolic distances efficiently
    # geoopt provides a vectorized 'dist' method.
    D_hyp = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))
    G = hyperbolic_g(D_hyp)
    loss_data = torch.sum((I - G) ** 2)
    # Regularization term: log penalty to avoid points near the boundary
    norm_sq = torch.sum(z**2, dim=1)
    loss_reg = -lambda_reg * torch.sum(torch.log(1 - norm_sq))
    return loss_data + loss_reg

# Initialize embeddings within the Poincaré disc
dataset = TUDataset(root='/tmp/TUDataset', name='MUTAG')
num_nodes = dataset[0].num_nodes
print(f"Cora dataset has {num_nodes} nodes")

# Now initialize z with the correct number of nodes
z = geoopt.ManifoldParameter(torch.randn(num_nodes, 2) * 0.1, manifold=manifold)

# Get edge_index and create adjacency matrix
edge_index = dataset[0].edge_index
I = torch.tensor(create_adj_matrix(edge_index, num_nodes), dtype=torch.float32)


# G = nx.Graph()

lambda_reg = 0.1
optimizer = geoopt.optim.RiemannianSGD([z], lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    loss = loss_hyperbolic(z, I, lambda_reg)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Assume we're working with the hyperbolic embedding 'z'
positions = []

# Training loop with recording every 'record_interval' epochs
record_interval = 50
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_hyperbolic(z, I, lambda_reg)
    loss.backward()
    optimizer.step()
    
    if epoch % record_interval == 0:
        # Record a copy of the current positions
        positions.append(z.detach().cpu().numpy())

# Create an animation of the positions over epochs
fig, ax = plt.subplots()
scat = ax.scatter([], [])

def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Hyperbolic Embeddings Over Time")
    return scat,

def update(frame):
    current_pos = positions[frame]
    scat.set_offsets(current_pos)
    ax.set_title(f"Epoch: {frame * record_interval}")
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(positions),
                              init_func=init, blit=True)
plt.show()
