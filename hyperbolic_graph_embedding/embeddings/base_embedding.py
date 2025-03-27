# embeddings/base_embedding.py

import torch.nn as nn
from abc import ABC, abstractmethod

class BaseEmbedding(nn.Module, ABC):
    """
    Abstract base class for graph embedding models.
    """
    def __init__(self):
        super(BaseEmbedding, self).__init__()
    
    @abstractmethod
    def forward(self, data):
        """
        Compute embeddings from input data.
        """
        pass

    @abstractmethod
    def loss(self, embeddings, data):
        """
        Compute the loss given embeddings and data.
        """
        pass

    @abstractmethod
    def optimize(self, loss):
        """
        Perform an optimization step given the loss.
        """
        pass
