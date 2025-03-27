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
            torch.randn(num_nodes, embedding_dim), manifold=self.manifold.manifold
        )
    
    def forward(self, data=None):
        """
        Returns the hyperbolic embeddings.
        """
        return self.embeddings
    
    def loss(self, embeddings, adjacency_matrix):
        """
        Compute the hyperbolic loss.
        
        Args:
            embeddings: Tensor of shape [N, d] on the Poincaré ball.
            adjacency_matrix: Tensor of shape [N, N] with binary values.
        
        Returns:
            A scalar loss value.
        """
        # Compute pairwise hyperbolic distances using Geoopt's method.
        # Geoopt's dist method accepts two tensors of shape [N, d]
        distances = self.manifold.manifold.dist(embeddings, embeddings)  # shape [N, N]
        # Map the distances using g(x)
        mapped = g(distances)
        data_loss = F.mse_loss(mapped, adjacency_matrix)
        
        # Regularization term: prevent embeddings from moving too close to the boundary.
        # Compute squared norms for each embedding.
        norm_sq = torch.sum(embeddings ** 2, dim=1)
        # Add a small epsilon to avoid log(0)
        reg_loss = - self.lambda_reg * torch.sum(torch.log(1 - norm_sq + 1e-5))
        return data_loss + reg_loss
    
    def optimize(self, loss):
        """
        Perform an optimization step (the user should call backward() and then optimizer.step()).
        """
        loss.backward()