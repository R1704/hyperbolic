import torch
import torch.nn as nn
import torch.optim as optim

# Simple neural network defining the Lagrangian (L = T - V)
class LagrangianNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, q, dq):
        x = torch.cat([q, dq], dim=-1)
        return self.net(x)

# Euler-Lagrange equations implemented via autograd
def euler_lagrange(lagrangian, q, dq):
    q.requires_grad_(True)
    dq.requires_grad_(True)

    L = lagrangian(q, dq)
    dL_dq = torch.autograd.grad(L.sum(), q, create_graph=True)[0]
    dL_ddq = torch.autograd.grad(L.sum(), dq, create_graph=True)[0]

    # Time derivative d/dt(dL/ddq)
    dd_dt_dL_ddq = torch.autograd.grad(dL_ddq.sum(), q, create_graph=True)[0] * dq + \
                   torch.autograd.grad(dL_ddq.sum(), dq, create_graph=True)[0] * (-dL_dq)

    return dd_dt_dL_ddq - dL_dq

# Synthetic data for harmonic oscillator: L = (1/2)*dq^2 - (1/2)*k*q^2
def generate_data(k=1.0, dt=0.1, steps=100):
    q = torch.tensor([[1.0]])
    dq = torch.tensor([[0.0]])
    trajectory = []

    for _ in range(steps):
        ddq = -k * q
        dq = dq + ddq * dt
        q = q + dq * dt
        trajectory.append((q.clone(), dq.clone()))

    return trajectory

# Training loop
def train():
    lagrangian = LagrangianNN()
    optimizer = optim.Adam(lagrangian.parameters(), lr=1e-3)

    data = generate_data()
    epochs = 5000

    for epoch in range(epochs):
        loss = 0.0
        for q, dq in data:
            el_eq = euler_lagrange(lagrangian, q, dq)
            loss += el_eq.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return lagrangian

if __name__ == "__main__":
    trained_lagrangian = train()