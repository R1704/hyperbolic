from hyperbolic_graph_embedding.manifolds.base_manifold import BaseManifold
import geoopt

class PoincareManifold(BaseManifold):
    """
    Wrapper for the Poincaré Ball model using Geoopt.
    """
    def __init__(self, c: float = 1.0):
        # c > 0: curvature parameter; Geoopt expects c and dim.
        self.manifold = geoopt.PoincareBall(c=c)
    
    def exp_map(self, x, v):
        """
        Exponential map: maps a tangent vector v at point x to the manifold.
        """
        return self.manifold.expmap(v, x)
    
    def log_map(self, x, y):
        """
        Logarithm map: returns the tangent vector at x that points towards y.
        """
        return self.manifold.logmap(y, x)
    
    def proj(self, x):
        """
        Projection: ensure that point x lies within the Poincaré ball.
        """
        return self.manifold.projx(x)
    
    def dist(self, x, y):
        """
        Distance: returns the distance between points x and y.
        """
        return self.manifold.dist(x, y)
