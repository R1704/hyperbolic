import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters for the simulation
N = 50          # Number of oscillators
dim = 2         # Dimension = 2 for visualization on the unit circle
num_steps = 200 # Number of update steps for the simulation
eta = 0.05      # Step size for the Euler integration

# Initialize oscillators as random unit vectors on the circle (N x 2)
x = torch.randn(N, dim)
x = F.normalize(x, p=2, dim=1)

# Connectivity matrix W (N x N): small random interactions
W = torch.randn(N, N) * 0.05

# Define matrix A (2 x 2) used to compute the antisymmetric natural rotation matrix Omega.
A = torch.randn(dim, dim) * 0.05  # random initialization
Omega = A - A.t()                # This ensures antisymmetry

# Conditional stimulus (bias) h for each oscillator; shape (N,2)
h = torch.randn(N, dim) * 0.05

# Function to perform one update step based on a Kuramoto-like rule.
def kuramoto_update(x, W, Omega, h, eta):
    """
    Update oscillator states x based on the Kuramoto update rule.
    
    Args:
        x: tensor of shape (N, 2) representing oscillator states on the unit circle.
        W: connectivity matrix (N, N)
        Omega: antisymmetric natural frequency matrix (2, 2)
        h: conditional stimulus for each oscillator (N, 2)
        eta: step size for Euler integration.
    Returns:
        Updated x (N,2) with unit norm.
    """
    # Interaction: each oscillator receives influence from every other oscillator.
    interaction = torch.matmul(W, x)  # shape: (N, 2)
    
    # Natural intrinsic rotation effect.
    natural = torch.matmul(x, Omega)  # shape: (N, 2)
    
    # Total drive is the sum of natural rotation, connectivity influence, and the bias.
    drive = natural + interaction + h  # (N, 2)
    
    # Project the drive onto the tangent space of each oscillator:
    # For each oscillator, v_tan = drive - (x dot drive)*x.
    dot = (x * drive).sum(dim=1, keepdim=True)  # shape: (N, 1)
    drive_tan = drive - dot * x  # ensures updates only affect direction
    
    # Euler integration update.
    x_new = x + eta * drive_tan
    # Renormalize to ensure the state stays on the unit circle.
    x_new = F.normalize(x_new, p=2, dim=1)
    return x_new

# Prepare to record the oscillator states over time for visualization.
states = []
order_params = []  # to store the global order parameter at each step

# Run simulation over the defined number of steps.
for t in range(num_steps):
    x = kuramoto_update(x, W, Omega, h, eta)
    states.append(x.clone().detach().numpy())
    # Compute the order parameter (average of oscillator vectors)
    avg = x.mean(dim=0)
    order_params.append(avg.clone().detach().numpy())

# Convert lists to numpy arrays for easier handling in visualization.
states = np.array(states)        # shape: (num_steps, N, 2)
order_params = np.array(order_params)  # shape: (num_steps, 2)

# ---------------------
# Visualization using matplotlib.animation
# ---------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_title("Kuramoto Oscillator Synchronization")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Draw a unit circle for reference.
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
ax.add_artist(circle)

# Initialize scatter plot for oscillator positions.
scat = ax.scatter([], [], s=60, color='blue', label="Oscillators")

# Initialize an arrow (quiver) for the order parameter.
order_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='red', label="Order Parameter")

def init():
    # scat.set_offsets([])
    order_arrow.set_UVC(0, 0)
    return scat, order_arrow

def update(frame):
    # frame is the current time step index
    curr_state = states[frame]  # shape: (N,2)
    # Update scatter plot data with oscillator positions.
    scat.set_offsets(curr_state)
    
    # Update the order parameter arrow.
    op = order_params[frame]
    # Length of order parameter (magnitude) indicates the degree of synchronization.
    order_arrow.set_UVC(op[0], op[1])
    
    ax.set_title(f"Step: {frame+1}/{num_steps}")
    return scat, order_arrow

# Create animation.
ani = animation.FuncAnimation(fig, update, frames=num_steps,
                              init_func=init, interval=50, blit=True)

# Optional: Save the animation as an mp4 file (requires ffmpeg).
ani.save('kuramoto_sync.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.legend()
plt.show()
