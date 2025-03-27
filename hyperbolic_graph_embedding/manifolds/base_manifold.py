# manifolds/base_manifold.py

from abc import ABC, abstractmethod

class BaseManifold(ABC):
    """
    Abstract base class for manifold operations.
    """
    @abstractmethod
    def exp_map(self, x, v):
        """
        Exponential map at point x for tangent vector v.
        """
        pass
    
    @abstractmethod
    def log_map(self, x, y):
        """
        Logarithm map at point x for point y.
        """
        pass
    
    @abstractmethod
    def proj(self, x):
        """
        Project point x onto the manifold.
        """
        pass
