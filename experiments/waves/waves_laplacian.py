import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parameters 
dt = 0.01  # time step
c = 5.0    # wave speed
damping = 0.999  # damping coefficient
N = 100    # number of grid points

# Create grid
x = np.linspace(-2 * np.pi, 2 * np.pi, N)
y = np.linspace(-2 * np.pi, 2 * np.pi, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Initialize wave states
Z = np.exp(-(X**2 + Y**2))  # Gaussian initial condition
Z_prev = Z.copy()

# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

def plot_surface(X, Y, Z, ax):
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Wave Equation Simulation (Laplacian Method)')
    ax.set_zlim(-2, 2)
    return ax

def laplacian(Z):
    """Compute the Laplacian using finite differences - optimized version."""
    return (-4*Z + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)) / (dx * dy)

def update(frame):
    global Z, Z_prev, ax
    
    # Compute Laplacian
    nabla2_Z = laplacian(Z)
    
    # Update wave equation using finite differences
    # Wave equation: ∂²Z/∂t² = c²∇²Z
    Z_next = (2 * Z - Z_prev + (c * dt)**2 * nabla2_Z)
    
    # Apply damping
    Z_next = Z_next * damping
    
    # Update states
    Z_prev = Z.copy()
    Z = Z_next.copy()
    
    # Update plot
    ax.clear()
    ax = plot_surface(X, Y, Z, ax)
    return ax

# Create animation
animation = FuncAnimation(fig, update, frames=500, interval=20)

# Show the plot
plt.show()