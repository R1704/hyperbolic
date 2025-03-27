# math_utils/metrics.py

import torch
import torch.nn.functional as F

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def custom_loss(embeddings, adjacency_matrix, distance_fn, lambda_reg=0.1):
    """
    Compute a loss that encourages neighboring nodes (per the adjacency matrix)
    to be close in the embedding space.
    """
    # Compute pairwise distances in a vectorized way.
    diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
    distances = distance_fn(diff, dim=2)
    mapped = 2 * torch.exp(-distances) / (1 + torch.exp(-distances))
    loss_data = F.mse_loss(mapped, adjacency_matrix)
    loss_reg = lambda_reg * torch.sum(embeddings ** 2)
    return loss_data + loss_reg
