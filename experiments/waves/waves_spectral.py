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
X, Y = np.meshgrid(x, y)

# Create wavenumber grid
kx = 2 * np.pi * np.fft.fftfreq(N, d=(x[1]-x[0]))
ky = 2 * np.pi * np.fft.fftfreq(N, d=(y[1]-y[0]))
KX, KY = np.meshgrid(kx, ky)
K_squared = KX**2 + KY**2

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
    ax.set_title('Wave Equation Simulation (Spectral Method)')
    ax.set_zlim(-2, 2)
    return ax

def update(frame):
    global Z, Z_prev, ax
    
    # Transform to frequency domain
    Z_hat = np.fft.fft2(Z)
    Z_prev_hat = np.fft.fft2(Z_prev)
    
    # Compute second time derivative in frequency domain
    # Wave equation: ∂²Z/∂t² = c²(∂²Z/∂x² + ∂²Z/∂y²)
    # In Fourier space: -ω² = -c²(kx² + ky²)
    Z_next_hat = (2 * Z_hat 
                  - Z_prev_hat 
                  - (c * dt)**2 * K_squared * Z_hat)
    
    # Transform back to spatial domain
    Z_next = np.real(np.fft.ifft2(Z_next_hat))
    
    # Apply damping
    Z_next = Z_next * damping
    
    # Update states
    Z_prev = Z.copy()
    Z = Z_next.copy()
    
    # Update plot
    ax.clear()
    ax = plot_surface(X, Y, Z, ax)
    print(X, Y, Z)
    return ax

# Create animation
animation = FuncAnimation(fig, update, frames=500, interval=20)

# Show the plot
plt.show()