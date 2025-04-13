import numpy as np
import matplotlib.pyplot as plt

# --- Simulation parameters ---
N = 50                # Number of oscillators
T = 20                # Total simulation time (seconds)
dt = 0.01             # Time step for Euler integration
steps = int(T/dt)     # Number of time steps
K = 2.0               # Global coupling strength

# --- Initialize oscillator phases and natural frequencies ---
# Phases are initialized uniformly at random in [0, 2pi)
theta = np.random.uniform(0, 2*np.pi, N)

# For demonstration, we draw frequencies from a Lorentzian (Cauchy) distribution.
# The probability density function is: g(omega) = gamma / (pi * ((omega - omega0)**2 + gamma**2))
omega0 = 1.0          # central frequency
gamma = 0.5           # scale (half-width at half-maximum)
# Draw N frequencies from a Cauchy distribution using numpy's standard Cauchy generator,
# then rescale and shift them appropriately.
omega = omega0 + gamma * np.tan(np.pi * (np.random.uniform(size=N) - 0.5))

# --- Prepare storage for simulation data ---
theta_history = np.zeros((steps, N))
order_param_history = np.zeros(steps, dtype=complex)

# --- Euler integration of the Kuramoto model ---
for t in range(steps):
    # Save the current phases
    theta_history[t] = theta
    # Compute the Kuramoto order parameter: Z = R*exp(i*psi)
    Z = np.mean(np.exp(1j*theta))
    order_param_history[t] = Z
    # Compute the coupling term for each oscillator
    # Each oscillator i receives a term sum_j sin(theta_j - theta_i)
    coupling = np.sum(np.sin(theta - theta[:, None]), axis=1)
    # Update phases using Euler's method
    # Note: We use (K/N)*coupling as the coupling term.
    theta += dt * (omega + (K/N) * coupling)
    # Optionally, wrap theta back into [0,2pi)
    theta = np.mod(theta, 2*np.pi)

# --- Visualization: Order Parameter vs. Time ---
time = np.linspace(0, T, steps)
R = np.abs(order_param_history)  # magnitude of the order parameter

plt.figure(figsize=(8, 4))
plt.plot(time, R, label='|Order Parameter|')
plt.xlabel("Time")
plt.ylabel("Synchronization (R)")
plt.title("Kuramoto Order Parameter over Time")
plt.legend()
plt.grid(True)
plt.show()

# --- Visualization: Oscillator Phases on the Unit Circle ---
# Take a snapshot at the final time step
theta_final = theta_history[-1]
# Compute (x,y) coordinates on the circle
x = np.cos(theta_final)
y = np.sin(theta_final)

plt.figure(figsize=(6, 6))
circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.scatter(x, y, c='blue', s=100, zorder=2)
for i in range(N):
    plt.text(x[i]*1.05, y[i]*1.05, f'{i}', fontsize=9, ha='center')
plt.title("Oscillator Phases on the Unit Circle at Final Time")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()
