# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split

# # --- Helper Functions ---
# def proj(x, y):
#     """Project y onto the tangent space at x (x is assumed to have unit norm)."""
#     dot = (x * y).sum(dim=1, keepdim=True)
#     return y - dot * x

# # --- Oscillatory Layer ---
# class OscillatoryLayer(nn.Module):
#     def __init__(self, num_nodes, d, gamma=0.1):
#         """
#         Args:
#             num_nodes: number of neurons (oscillators)
#             d: dimensionality (for visualization, d=2)
#             gamma: step size for the update
#         """
#         super(OscillatoryLayer, self).__init__()
#         self.num_nodes = num_nodes
#         self.d = d
#         self.gamma = gamma
        
#         # Initialize oscillator states on the unit sphere (in 2D, use angles)
#         angles = 2 * np.pi * torch.rand(num_nodes)
#         self.x = nn.Parameter(torch.stack([torch.cos(angles), torch.sin(angles)], dim=1))
        
#         # Natural frequency term (for 2D, represented as a scalar per neuron)
#         self.omega = nn.Parameter(torch.randn(num_nodes))
        
#         # External stimulus (symmetry-breaking bias)
#         self.c = nn.Parameter(torch.randn(num_nodes, d))
        
#         # Coupling matrix (learnable weights)
#         self.J = nn.Parameter(torch.randn(num_nodes, num_nodes))
    
#     def forward(self, x_in=None):
#         # Compute rotation due to natural frequency
#         rot = torch.stack([-self.omega * self.x[:, 1], self.omega * self.x[:, 0]], dim=1)
        
#         # Compute coupling: weighted sum over other neurons
#         coupling = torch.matmul(self.J, self.x)
        
#         # Total input = external stimulus + coupling
#         total_input = self.c + coupling
        
#         # Project total input onto the tangent space at self.x
#         proj_input = proj(self.x, total_input)
        
#         # Compute update in tangent space
#         delta_x = rot + proj_input
        
#         # Euler update and re-normalize
#         x_new = self.x + self.gamma * delta_x
#         x_new = x_new / x_new.norm(dim=1, keepdim=True)
        
#         # Update state (for demonstration, update in-place)
#         self.x.data = x_new.data
#         return self.x

# # --- Readout Layer ---
# class ReadoutLayer(nn.Module):
#     def __init__(self, num_nodes, input_dim, output_dim):
#         super(ReadoutLayer, self).__init__()
#         # For simplicity, average the oscillatory outputs and pass through a linear layer.
#         self.linear = nn.Linear(input_dim, output_dim)
    
#     def forward(self, x):
#         # x has shape (num_nodes, d). We average over nodes.
#         pooled = torch.mean(x, dim=0)  # shape: (d,)
#         return self.linear(pooled)

# # --- Full Model ---
# class OscillatoryNN(nn.Module):
#     def __init__(self, num_nodes, d, num_steps, output_dim, gamma=0.1):
#         super(OscillatoryNN, self).__init__()
#         self.num_steps = num_steps
#         self.oscillatory_layer = OscillatoryLayer(num_nodes, d, gamma)
#         self.readout = ReadoutLayer(num_nodes, d, output_dim)
    
#     def forward(self, x_in=None):
#         # Run oscillatory dynamics for num_steps iterations
#         for _ in range(self.num_steps):
#             self.oscillatory_layer.forward()
#         # Read out final oscillator states
#         return self.readout(self.oscillatory_layer.x)

# # --- Toy Classification Example ---

# # Generate synthetic data (e.g., two moons)
# X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # For simplicity, let's assume each data sample is used to initialize the oscillatory states.
# # We use a very small "batch" simulation by learning a single set of oscillators per class.
# # In practice, you would integrate this into a larger architecture.

# # Here we create an OscillatoryNN for classification (binary, so output_dim=2)
# num_nodes = 100  # number of oscillators (can be viewed as "neurons" for one sample)
# d = 2           # 2D oscillatory states (on the unit circle)
# num_steps = 50  # number of oscillatory update steps
# output_dim = 2  # binary classification

# # Instantiate model and optimizer
# model = OscillatoryNN(num_nodes, d, num_steps, output_dim, gamma=0.05)
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.CrossEntropyLoss()

# # For our toy example, we simulate training on the synthetic data by
# # assuming that the input X somehow modulates the initial external stimulus c.
# # Here, for demonstration, we simply train the oscillatory network parameters
# # on the classification loss using a fixed dummy batch.

# num_epochs = 1000
# loss_history = []

# for epoch in range(num_epochs):
#     optimizer.zero_grad()
    
#     # In a real model, x_in could be used to set initial states or modulate c.
#     # Here we ignore x_in and focus on the dynamics.
#     outputs = model()  # output is of shape (output_dim,)
#     # To create a batch, we replicate the output and pretend each oscillator contributes.
#     logits = outputs.unsqueeze(0)  # shape: (1, output_dim)
    
#     # Dummy target: alternating classes for demonstration
#     target = torch.tensor([epoch % 2])  # just an example target
    
#     loss = criterion(logits, target)
#     loss.backward()
#     optimizer.step()
    
#     loss_history.append(loss.item())
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}: Loss = {loss.item()}")


# # Plot training loss over epochs
# plt.figure(figsize=(8, 4))
# plt.plot(loss_history, label="Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Oscillatory NN Training Loss")
# plt.legend()
# plt.savefig('osc_loss.png')
# plt.show()

# # ----- Enhanced Visualizations -----

# # 1. Visualize oscillator states on the unit circle
# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111, projection='polar')
# theta = torch.atan2(model.oscillatory_layer.x[:, 1], model.oscillatory_layer.x[:, 0]).detach().numpy()
# r = torch.ones(num_nodes)
# colors = torch.linspace(0, 1, num_nodes)

# # Plot the oscillator positions
# scatter = ax.scatter(theta, r.detach().numpy(), c=colors, cmap='viridis', alpha=0.7)
# ax.set_rticks([0.5, 1.0])
# ax.set_title("Oscillator States on Unit Circle")
# plt.colorbar(scatter, label="Oscillator Index")
# plt.savefig('osc_states.png')
# plt.show()

# # 2. Visualize coupling matrix as a heatmap
# plt.figure(figsize=(10, 8))
# coupling = model.oscillatory_layer.J.detach().numpy()
# plt.imshow(coupling, cmap='coolwarm')
# plt.colorbar(label="Coupling Strength")
# plt.title("Coupling Matrix between Oscillators")
# plt.xlabel("Oscillator Index (to)")
# plt.ylabel("Oscillator Index (from)")
# plt.savefig('osc_coupling.png')
# plt.show()

# # 3. Visualize model predictions on the two moons dataset
# def predict(model, X):
#     # Use your model to predict class probabilities
#     results = []
    
#     # Save original parameters to restore later
#     original_c = model.oscillatory_layer.c.data.clone()
    
#     for x in X:
#         # Reset oscillator states
#         angles = 2 * np.pi * torch.rand(num_nodes)
#         model.oscillatory_layer.x.data = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        
#         # Modulate the external stimulus with the input (without cumulative effect)
#         x_tensor = torch.tensor(x, dtype=torch.float32)
#         model.oscillatory_layer.c.data = original_c + x_tensor.repeat(num_nodes, 1)
        
#         # Forward pass
#         with torch.no_grad():
#             output = model()
#         results.append(output.softmax(dim=0).numpy())
    
#     # Restore original parameters
#     model.oscillatory_layer.c.data = original_c
    
#     return np.array(results)
# # Create meshgrid for decision boundary visualization
# h = 0.01
# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# grid_points = np.c_[xx.ravel(), yy.ravel()]

# # Get predictions for grid points
# Z = predict(model, grid_points)
# Z_class = np.argmax(Z, axis=1).reshape(xx.shape)

# # Plot decision boundary and data points
# plt.figure(figsize=(10, 8))
# plt.contourf(xx, yy, Z_class, cmap=plt.cm.coolwarm, alpha=0.8)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, alpha=0.6, marker='s')
# plt.title("Model Decision Boundary and Data Points")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.savefig('osc_decision_boundary.png')
# plt.show()

# # 4. Visualize oscillator dynamics over time
# def visualize_dynamics(model, steps=50):
#     # Store the dynamics of oscillators
#     dynamics = []
    
#     # Reset oscillator states
#     angles = 2 * np.pi * torch.rand(num_nodes)
#     model.oscillatory_layer.x.data = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    
#     # Initial state
#     dynamics.append(model.oscillatory_layer.x.detach().clone().numpy())
    
#     # Run dynamics
#     for _ in range(steps):
#         model.oscillatory_layer.forward()
#         dynamics.append(model.oscillatory_layer.x.detach().clone().numpy())
    
#     # Visualize dynamics (sample only 10 oscillators for clarity)
#     sample_indices = np.linspace(0, num_nodes-1, 10, dtype=int)
#     plt.figure(figsize=(12, 10))
    
#     # Plot trajectories on unit circle
#     ax = plt.subplot(111)
#     circle = plt.Circle((0, 0), 1, fill=False, color='black')
#     ax.add_artist(circle)
    
#     # Plot unit circle
#     theta = np.linspace(0, 2*np.pi, 100)
#     ax.plot(np.cos(theta), np.sin(theta), 'k-')
    
#     # Plot oscillator trajectories with gradient color
#     for idx in sample_indices:
#         x_coords = [d[idx, 0] for d in dynamics]
#         y_coords = [d[idx, 1] for d in dynamics]
        
#         # Create gradient color along trajectory
#         points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
#         lc = plt.LineCollection(segments, cmap='viridis', 
#                                norm=plt.Normalize(0, len(dynamics)-1))
#         lc.set_array(np.linspace(0, 1, len(dynamics)-1))
#         ax.add_collection(lc)
        
#         # Mark start and end points
#         ax.plot(x_coords[0], y_coords[0], 'go', markersize=5)
#         ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=5)
    
#     plt.colorbar(lc, label='Time Step')
#     ax.set_xlim(-1.1, 1.1)
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_aspect('equal')
#     ax.grid(True)
#     plt.title("Oscillator Dynamics Over Time")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.savefig('osc_dynamics.png')
#     plt.show()

# # Visualize the dynamics
# visualize_dynamics(model)




# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np


# # Init: 4 oscillators on unit circle (2D vectors)
# num_neurons = 4
# dim = 2
# T = 20
# gamma = 0.2
# J = 0.5 * (torch.ones(num_neurons, num_neurons) - torch.eye(num_neurons))
# c = F.normalize(torch.tensor([[1.0, 0.0]] * num_neurons), dim=-1)
# x = F.normalize(torch.randn(num_neurons, dim), dim=-1)

# trajectory = []

# def project_to_tangent(x, y):
#     return y - (x * y).sum(dim=-1, keepdim=True) * x

# for t in range(T):
#     Jx = torch.matmul(J, x)  # coupling
#     force = c + Jx
#     delta = project_to_tangent(x, force)
#     x = F.normalize(x + gamma * delta, dim=-1)
#     trajectory.append(x.detach().numpy())

# trajectory = np.stack(trajectory)  # (T, N, 2)

# # Plotting animation
# fig, ax = plt.subplots(figsize=(6,6))
# circle = plt.Circle((0, 0), 1, fill=False, linestyle='--')
# ax.add_artist(circle)
# colors = ['red', 'blue', 'green', 'orange']
# dots = [ax.plot([], [], 'o', color=colors[i])[0] for i in range(num_neurons)]
# ax.set_xlim(-1.2, 1.2)
# ax.set_ylim(-1.2, 1.2)
# ax.set_aspect('equal')
# ax.axis('off')

# def update(frame):
#     for i, dot in enumerate(dots):
#         dot.set_data(trajectory[frame][i][0], trajectory[frame][i][1])
#     return dots

# ani = animation.FuncAnimation(fig, update, frames=T, interval=300)
# plt.savefig('oscillator_dynamics.gif', dpi=80, writer='imagemagick')
# plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Define the AKOrN Block
# -----------------------------
class AKOrNBlock(nn.Module):
    def __init__(self, num_oscillators, dim, num_steps=10):
        """
        Implements the Artificial Kuramoto Oscillatory Neurons block.
        
        Args:
            num_oscillators: Number of oscillators (N)
            dim: Dimensionality of each oscillator (d)
            num_steps: Number of discrete Kuramoto update steps
        """
        super(AKOrNBlock, self).__init__()
        self.num_osc = num_oscillators
        self.dim = dim
        self.num_steps = num_steps
        
        # Learnable step size for Euler updates.
        self.eta = nn.Parameter(torch.tensor(0.1))
        
        # Connectivity matrix (interaction among oscillators); shape: (N, N)
        self.W = nn.Parameter(torch.randn(num_oscillators, num_oscillators) * 0.1)
        
        # Learnable matrix to construct antisymmetric natural frequency matrix Omega.
        # Omega = A - A^T ensures antisymmetry.
        A_init = torch.randn(dim, dim) * 0.1
        self.A = nn.Parameter(A_init)
        
        # Conditional stimulus (bias) for each oscillator, shape: (N, d)
        self.h = nn.Parameter(torch.randn(num_oscillators, dim) * 0.1)
        
        # A simple readout layer mapping each oscillator's state to a scalar.
        self.readout = nn.Linear(dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of initial oscillator states, shape (batch, N, d).
        Returns:
            out: Aggregated output from the readout (batch, 1)
            x: Final oscillator states after the iterative updates.
        """
        batch_size = x.size(0)
        # Normalize input oscillator states so they lie on the unit sphere.
        x = F.normalize(x, p=2, dim=-1)
        
        # Construct the antisymmetric natural frequency matrix: Omega = A - A^T.
        Omega = self.A - self.A.t()
        
        # Perform iterative Kuramoto updates.
        for _ in range(self.num_steps):
            # Compute interaction: each oscillator receives influence from all others.
            interaction = torch.einsum('ij,bjd->bid', self.W, x)
            # Compute intrinsic rotation from natural frequencies.
            natural = torch.matmul(x, Omega)
            # Combine connectivity, natural rotation, and the conditional stimulus.
            drive = natural + (interaction + self.h.unsqueeze(0))
            # Project the drive onto the tangent space (ensure updates change only the direction).
            dot = (x * drive).sum(dim=-1, keepdim=True)
            drive_tan = drive - dot * x
            # Euler integration update.
            x = x + self.eta * drive_tan
            # Re-normalize to maintain unit norm.
            x = F.normalize(x, p=2, dim=-1)
        
        # Aggregate each oscillator's final state via a linear projection.
        out = self.readout(x)  # shape: (batch, N, 1)
        out = out.mean(dim=1)  # Aggregate over oscillators -> shape: (batch, 1)
        return out, x

# -----------------------------
# Define the Classifier Network
# -----------------------------
class AKOrNClassifier(nn.Module):
    def __init__(self, input_dim, num_osc, d, num_steps, num_classes):
        """
        A classifier that uses the AKOrN block.
        
        Args:
            input_dim: Dimensionality of the input data.
            num_osc: Number of oscillators.
            d: Dimensionality of each oscillator.
            num_steps: Number of Kuramoto update steps.
            num_classes: Number of output classes.
        """
        super(AKOrNClassifier, self).__init__()
        self.num_osc = num_osc
        self.d = d
        # Map input features into a set of oscillators (flattened then reshaped).
        self.fc = nn.Linear(input_dim, num_osc * d)
        self.akorn = AKOrNBlock(num_osc, d, num_steps)
        # Final classification layer.
        self.classifier = nn.Linear(1, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        # Project input to oscillator space.
        x = self.fc(x)  # shape: (batch, num_osc * d)
        x = x.view(batch_size, self.num_osc, self.d)  # shape: (batch, num_osc, d)
        # Apply the AKOrN block.
        out, _ = self.akorn(x)
        # Generate class logits.
        logits = self.classifier(out)
        return logits

# -----------------------------
# Experiment: Synthetic Data Classification
# -----------------------------
def train_experiment():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate a synthetic dataset for binary classification.
    num_samples = 1000
    input_dim = 20
    num_classes = 2
    # Class 0: samples centered at -1, Class 1: samples centered at +1.
    X0 = np.random.randn(num_samples//2, input_dim) + (-1.0)
    X1 = np.random.randn(num_samples//2, input_dim) + (1.0)
    X = np.vstack((X0, X1)).astype(np.float32)
    y = np.hstack((np.zeros(num_samples//2), np.ones(num_samples//2))).astype(np.int64)
    
    # Shuffle the data.
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]
    
    # Split into training (80%) and validation (20%) sets.
    split = int(0.8 * num_samples)
    X_train = torch.tensor(X[:split])
    y_train = torch.tensor(y[:split])
    X_val = torch.tensor(X[split:])
    y_val = torch.tensor(y[split:])
    
    # Create data loaders.
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Hyperparameters for the model.
    num_osc = 64   # Number of oscillators.
    d = 3          # Dimensionality of each oscillator.
    num_steps = 10 # Number of iterative Kuramoto updates.
    
    model = AKOrNClassifier(input_dim=input_dim, num_osc=num_osc, d=d, num_steps=num_steps, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 30
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Evaluate on validation set.
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {val_acc:.4f}")
        best_val_acc = max(best_val_acc, val_acc)
    
    print("Best validation accuracy: {:.2f}%".format(best_val_acc * 100))
    
    # Optionally, plot training and validation losses.
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig('kk.png')
    plt.show()

if __name__ == '__main__':
    train_experiment()
