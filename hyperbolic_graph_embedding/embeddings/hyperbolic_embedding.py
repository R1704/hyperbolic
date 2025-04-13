import torch
import torch.nn.functional as F
import geoopt

from hyperbolic_graph_embedding.math_utils.utils import g
from hyperbolic_graph_embedding.manifolds.poincare_manifold import PoincareManifold
from hyperbolic_graph_embedding.embeddings.base_embedding import BaseEmbedding

class HyperbolicEmbedding(BaseEmbedding):
    """
    Implements a hyperbolic embedding model.
    
    The loss function is defined as:
      J_hyp(z_1,...,z_N) = sum_{i,j} (I_{ij} - g(d_hyp(z_i,z_j)))^2 - lambda * sum_i log(1 - ||z_i||^2),
    where d_hyp is the hyperbolic distance in the Poincaré ball.
    """
    def __init__(self, num_nodes: int, embedding_dim: int, manifold: PoincareManifold, lambda_reg: float = 0.1):
        super(HyperbolicEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.lambda_reg = lambda_reg
        self.manifold = manifold  # Instance of PoincareManifold
        # Initialize hyperbolic embeddings as a Geoopt ManifoldParameter.
        self.embeddings = geoopt.ManifoldParameter(
            self.manifold.manifold.random((num_nodes, embedding_dim), mean=0.0, std=0.1),
            manifold=self.manifold.manifold
        )

    def forward(self, data=None):
        """
        Returns the hyperbolic embeddings.
        """
        return self.embeddings
    
    def loss(self, embeddings, adjacency_matrix):
        """
        Calculate the hyperbolic embedding loss following the formula:
        J_hyp(z) = sum_{i,j} (I_{ij} - g(d_hyp(z_i,z_j)))^2 - lambda * sum_i log(1 - ||z_i||^2)
        """
        # Vectorized calculation of pairwise hyperbolic distances
        # This is more efficient than the loop implementation
        distances = self.manifold.dist(
            embeddings.unsqueeze(1),  # Shape: [N, 1, dim]
            embeddings.unsqueeze(0)   # Shape: [1, N, dim]
        )  # Result shape: [N, N]
        
        # Apply the mapping function g to convert distances to similarities
        mapped = self.distance_to_probability(distances)
        
        # Calculate data loss (match theoretical form more closely)
        # Note: Using sum instead of mean to match the formula exactly
        loss_data = torch.sum((adjacency_matrix - mapped) ** 2)
        
        # Regularization to keep points away from the boundary
        norm_sq = torch.sum(embeddings**2, dim=1)  # Squared norms
        boundary_dist = 1 - norm_sq  # Distance from boundary
        
        # Add small epsilon for numerical stability
        loss_reg = -self.lambda_reg * torch.sum(
            torch.log(torch.clamp(boundary_dist, min=1e-8))
        )
        
        return loss_data + loss_reg

    # And update your distance_to_probability function:
    def distance_to_probability(self, distances):
        # Convert distances to similarities/probabilities
        # Limit max distance to avoid numerical instability
        # distances = torch.clamp(distances, 0, 15)
        return 2 * torch.exp(-distances) / (1 + torch.exp(-distances))
    
    def optimize(self, loss):
        """
        Perform an optimization step (the user should call backward() and then optimizer.step()).
        """
        loss.backward()