from base_manifold import BaseManifold
import geoopt

class PoincareManifold(BaseManifold):
    """
    Wrapper for the Poincaré Ball model using Geoopt.
    """
    def __init__(self, dim: int, c: float = 1.0):
        # c > 0: curvature parameter; Geoopt expects c and dim.
        self.manifold = geoopt.PoincareBall(c=c, dim=dim)
    
    def exp_map(self, x, v):
        """
        Exponential map: maps a tangent vector v at point x to the manifold.
        """
        return self.manifold.expmap(v, base_point=x)
    
    def log_map(self, x, y):
        """
        Logarithm map: returns the tangent vector at x that points towards y.
        """
        return self.manifold.logmap(y, base_point=x)
    
    def proj(self, x):
        """
        Projection: ensure that point x lies within the Poincaré ball.
        """
        return self.manifold.projx(x)
