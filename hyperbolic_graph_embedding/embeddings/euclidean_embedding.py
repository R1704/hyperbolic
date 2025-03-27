import torch
import torch.functional as F
from base_embedding import BaseEmbedding


class EuclideanEmbedding(BaseEmbedding):
    """
    Euclidean embedding model.
    The loss function is defined as:
      J(u_1,...,u_N) = Σ_{i,j} (I_ij - g(d(u_i,u_j)))^2 + λ * Σ_i ||u_i||^2,
    where g(x)=2*exp(-x)/(1+exp(-x)) and d(u_i,u_j) is the Euclidean distance.
    """
    def __init__(self, num_nodes: int, embedding_dim: int, lambda_reg: float = 0.1):
        super(EuclideanEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.lambda_reg = lambda_reg
        # Initialize embeddings with a Gaussian distribution (zero mean)
        self.embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim))
    
    def forward(self, data=None):
        """
        For simplicity, ignore extra data and return the embedding matrix.
        """
        return self.embeddings
    
    def loss(self, embeddings, adjacency_matrix):
        """
        Compute the Euclidean loss.
        
        Args:
            embeddings: Tensor of shape [N, d].
            adjacency_matrix: Tensor of shape [N, N] with binary entries.
        
        Returns:
            Scalar loss value.
        """
        # Compute pairwise Euclidean distances
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # shape [N, N, d]
        distances = torch.norm(diff, dim=2)  # shape [N, N]
        # Map distances using g(x)
        mapped = g(distances)
        data_loss = F.mse_loss(mapped, adjacency_matrix)
        reg_loss = self.lambda_reg * torch.sum(embeddings ** 2)
        return data_loss + reg_loss