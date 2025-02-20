import numpy as np 


# Define some helper functions for the Poincare disk model.
def get_unit_circle(n_segments=100, scale=1.0):
    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=True)
    unit_circle = np.column_stack((np.cos(theta), np.sin(theta))).astype(np.float32) * scale
    unit_circle = np.concatenate([unit_circle, unit_circle[0:1]], axis=0)
    return unit_circle

def mobius_transform(z, a):
    """Apply a Mobius transformation to map z to the origin."""
    return (z - a) / (1 - np.conj(a) * z)

def inverse_mobius_transform(z, a):
    """Apply a Mobius transformation to map the origin to z."""
    return (a + z) / (1 + np.conj(a) * z)

def get_arc(z, n_points=25_000):
    """Compute the geodesic circle through the origin and z."""
    t = np.linspace(0, 1, num=n_points)
    return t * z

def hyperbolic_isometry(z, t):
    # a rotation by t radians.
    return np.exp(1j * t) * z

def get_geodesic(z1, z2):
    """
    We first apply a Mobius transformation to map z1 to the origin.
    Then we compute the geodesic circle through the origin and z2, which is on the real line.
    Finally, we apply the inverse Mobius transformation to map the geodesic back to the original
    disk
    """

    z2_transformed = mobius_transform(z2, z1)
    geodesic_transformed = get_arc(z2_transformed)
    geodesic_original = inverse_mobius_transform(geodesic_transformed, z1)
    return geodesic_original

def distance(z1, z2):
    return np.arccosh(1 + 2 * abs(z1 - z2)**2 / ((1 - abs(z1)**2) * (1 - abs(z2)**2)))

def circle_inversion(z, c, R):
    return c + (R**2) / (np.conj(z - c))
