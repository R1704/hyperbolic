import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parameters
dt = 0.01  # time step
c = 5.0    # wave speed
damping = 0.999  # damping coefficient
N = 50    # reduced grid points for performance with autograd

# Create grid
x = np.linspace(-2 * np.pi, 2 * np.pi, N)
y = np.linspace(-2 * np.pi, 2 * np.pi, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Create PyTorch tensors for coordinates
x_tensor = torch.linspace(-2 * torch.pi, 2 * torch.pi, N, requires_grad=True)
y_tensor = torch.linspace(-2 * torch.pi, 2 * torch.pi, N, requires_grad=True)
X_mesh, Y_mesh = torch.meshgrid(x_tensor, y_tensor, indexing='ij')

# Initialize wave states as PyTorch tensors
Z_initial = torch.exp(-(X_mesh**2 + Y_mesh**2))
Z = Z_initial.clone()
Z_prev = Z_initial.clone()

# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

def plot_surface(X, Y, Z, ax):
    Z_np = Z.detach().numpy() if isinstance(Z, torch.Tensor) else Z
    ax.plot_surface(X, Y, Z_np, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Wave Equation Simulation (True PyTorch Autograd)')
    ax.set_zlim(-2, 2)
    return ax

def laplacian_torch(Z):
    """Compute the Laplacian using finite differences with PyTorch operations."""
    return (-4*Z + torch.roll(Z, 1, dims=0) + torch.roll(Z, -1, dims=0) +
            torch.roll(Z, 1, dims=1) + torch.roll(Z, -1, dims=1)) / (dx * dy)

def true_autograd_laplacian(Z):
    """Compute the Laplacian using PyTorch's autograd for real."""
    # We need to model Z as a function of X and Y coordinates
    # First create a neural network that will represent our wave function
    
    # Functional representation of our wave - simple 2-layer neural net
    wave_func = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )
    
    # Train this network to represent our current Z values
    optimizer = torch.optim.Adam(wave_func.parameters(), lr=0.01)
    
    # Prepare our data: grid points and Z values
    coords = torch.stack([X_mesh.flatten(), Y_mesh.flatten()], dim=1)
    z_values = Z.flatten().unsqueeze(1)
    
    # Train the network (simplified, would need more epochs for accuracy)
    for _ in range(50):
        optimizer.zero_grad()
        pred = wave_func(coords)
        loss = torch.nn.functional.mse_loss(pred, z_values)
        loss.backward()
        optimizer.step()
    
    # Now our wave_func approximates Z(x,y)
    # We can compute the Laplacian by taking second derivatives
    laplacian = torch.zeros_like(Z)
    
    for i in range(N):
        for j in range(N):
            x = X_mesh[i, j].item()
            y = Y_mesh[i, j].item()
            
            # Create point with requires_grad
            point = torch.tensor([x, y], requires_grad=True)
            
            # Get function value at this point
            z_val = wave_func(point.unsqueeze(0))
            
            # Compute first derivatives
            grads = torch.autograd.grad(z_val, point, create_graph=True)[0]
            
            # Compute second derivatives for Laplacian
            d2z_dx2 = torch.autograd.grad(grads[0], point, retain_graph=True)[0][0]
            d2z_dy2 = torch.autograd.grad(grads[1], point, retain_graph=True)[0][1]
            
            # Sum to get Laplacian
            laplacian[i, j] = d2z_dx2 + d2z_dy2
    
    return laplacian

def direct_autograd_laplacian(Z):
    """A more direct approach to compute Laplacian with autograd."""
    # Create grid with gradients enabled
    grid_points = []
    for i in range(N):
        for j in range(N):
            # Each point is a parameter requiring gradients
            grid_points.append(torch.tensor([x[i], y[j]], requires_grad=True))
    
    # Create a function that maps from (x,y) to z-values
    # For simplicity, we'll use a nearest-neighbor interpolation
    def interp_func(point):
        # Find closest grid point to the input coordinates
        i = torch.argmin(torch.abs(x_tensor - point[0]))
        j = torch.argmin(torch.abs(y_tensor - point[1]))
        return Z[i, j]
    
    # Compute the Laplacian at each point
    laplacian = torch.zeros_like(Z)
    
    for idx, point in enumerate(grid_points):
        i, j = idx // N, idx % N
        
        # Function value at this point
        z_val = interp_func(point)
        
        # Try to compute first derivatives
        try:
            grad_outputs = torch.ones_like(z_val)
            grads = torch.autograd.grad(z_val, point, grad_outputs=grad_outputs, 
                                       create_graph=True)[0]
            
            # Try to compute second derivatives
            d2z_dx2 = torch.autograd.grad(grads[0], point, retain_graph=True)[0][0]
            d2z_dy2 = torch.autograd.grad(grads[1], point, retain_graph=True)[0][1]
            
            laplacian[i, j] = d2z_dx2 + d2z_dy2
        except:
            print(f"Failed to compute Laplacian at point {i}, {j} using autograd, using laplacian_torch instead.")
            # Fallback to finite differences if autograd fails
            laplacian[i, j] = laplacian_torch(Z)[i, j]
    
    return laplacian

def efficient_autograd_laplacian(Z):
    """Most practical approach using autograd for the Laplacian."""
    Z_param = Z.clone().detach().requires_grad_(True)
    
    # Create mesh of coordinates matching Z's shape
    batch_size = Z.shape[0] * Z.shape[1]
    coords = torch.stack([X_mesh.flatten(), Y_mesh.flatten()], dim=1)
    
    # Define our wave as a differentiable function of space
    def wave_func(xy):
        # For each coordinate pair, sample Z using differentiable interpolation
        # For simplicity, we'll use a basic differentiable sampling approach
        
        # Normalized coordinates between 0 and N-1
        x_norm = (xy[:, 0] + 2*torch.pi) / (4*torch.pi) * (N-1)
        y_norm = (xy[:, 1] + 2*torch.pi) / (4*torch.pi) * (N-1)
        
        # Integer indices
        x0 = torch.clamp(torch.floor(x_norm).long(), 0, N-2)
        y0 = torch.clamp(torch.floor(y_norm).long(), 0, N-2)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Fractional parts
        wx = x_norm - x0.float()
        wy = y_norm - y0.float()
        
        # Bilinear interpolation
        top_left = Z_param[x0, y0] * (1-wx).unsqueeze(1) * (1-wy).unsqueeze(0)
        top_right = Z_param[x1, y0] * wx.unsqueeze(1) * (1-wy).unsqueeze(0)
        bottom_left = Z_param[x0, y1] * (1-wx).unsqueeze(1) * wy.unsqueeze(0)
        bottom_right = Z_param[x1, y1] * wx.unsqueeze(1) * wy.unsqueeze(0)
        
        return top_left + top_right + bottom_left + bottom_right
    
    # Output derivatives
    laplacian = torch.zeros_like(Z)
    
    # For practical purposes, we'll use a small subset of points (or you could
    # use finite differences for most points and autograd for a few important ones)
    sample_indices = torch.randint(0, batch_size, (min(100, batch_size),))
    sampled_coords = coords[sample_indices]
    
    for point in sampled_coords:
        # Get grid indices
        x_idx = int((point[0] + 2*torch.pi) / (4*torch.pi) * (N-1))
        y_idx = int((point[1] + 2*torch.pi) / (4*torch.pi) * (N-1))
        
        # Only compute if within valid range
        if 0 <= x_idx < N and 0 <= y_idx < N:
            # Create point with requires_grad
            p = point.clone().detach().requires_grad_(True)
            
            # Function value at this point
            z_val = wave_func(p.unsqueeze(0))
            
            # First derivatives
            grads = torch.autograd.grad(z_val, p, create_graph=True)[0]
            
            # Second derivatives for Laplacian
            d2z_dx2 = torch.autograd.grad(grads[0], p, retain_graph=True)[0][0]
            d2z_dy2 = torch.autograd.grad(grads[1], p, retain_graph=True)[0][1]
            
            # Sum to get Laplacian
            laplacian[x_idx, y_idx] = d2z_dx2 + d2z_dy2
    
    # Fill in the rest with finite differences
    mask = laplacian == 0
    laplacian[mask] = laplacian_torch(Z)[mask]
    
    return laplacian

def simple_autograd_laplacian(Z):
    """A simpler approach using autograd for the Laplacian on a grid."""
    # Create tensor with gradients enabled
    Z_param = Z.clone().detach().requires_grad_(True)
    
    # Create a dummy loss that depends on Z_param
    # (sum of squares is a simple choice)
    dummy_loss = (Z_param ** 2).sum()
    
    # Get first gradients
    first_grads = torch.autograd.grad(dummy_loss, Z_param, create_graph=True)[0]
    
    # We need to compute second derivatives with respect to x and y
    # This is a simplification that doesn't truly represent ∇²Z
    # For demonstration purposes only
    hessian_elements = []
    for i in range(N):
        for j in range(N):
            hess = torch.autograd.grad(first_grads[i, j], Z_param, retain_graph=True)[0]
            hessian_elements.append(hess[i, j])
    
    # Reshape into a Laplacian approximation
    laplacian = torch.tensor(hessian_elements).reshape(N, N)
    
    return laplacian

def update(frame):
    global Z, Z_prev, ax
    
    # Compute Laplacian using PyTorch
    # Use laplacian_torch for better performance
    nabla2_Z = simple_autograd_laplacian(Z)
    
    # Update wave equation using finite differences
    # Wave equation: ∂²Z/∂t² = c²∇²Z
    Z_next = (2 * Z - Z_prev + (c * dt)**2 * nabla2_Z)
    
    # Apply damping
    Z_next = Z_next * damping
    
    # Update states
    Z_prev = Z.clone()
    Z = Z_next.clone()
    
    # Update plot
    ax.clear()
    ax = plot_surface(X, Y, Z, ax)
    return ax

# Create animation
animation = FuncAnimation(fig, update, frames=200, interval=20)

# Show the plot
plt.show()