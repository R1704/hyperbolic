import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Set random seed for reproducibility
np.random.seed(42)


def kuramoto(theta, t, omega, K, N):
    """
    Computes the phase evolution of N coupled Kuramoto oscillators.

    Parameters:
        theta : array-like
            Current phase angles of oscillators.
        t : float
            Current time step.
        omega : array-like
            Natural frequencies of the oscillators.
        K : float
            Coupling strength.
        N : int
            Number of oscillators.

    Returns:
        dtheta_dt : array-like
            Rate of change of phases.
    """
    theta_diff = np.subtract.outer(theta, theta)  # Compute all pairwise phase differences
    coupling_term = (K / N) * np.sum(np.sin(theta_diff), axis=1)  # Coupling term
    dtheta_dt = omega + coupling_term  # Phase evolution
    return dtheta_dt


# Number of oscillators
N = 10  

# Natural frequencies drawn from a normal distribution
omega = np.random.normal(loc=1.0, scale=0.2, size=N)  

# Coupling strength
K = 1.5  

# Initial random phases in the range [0, 2π]
theta0 = np.random.uniform(0, 2 * np.pi, N)

# Time parameters
T = 10  # Total simulation time
dt = 0.05  # Time step
time = np.arange(0, T, dt)


# Solve ODE
theta_t = odeint(kuramoto, theta0, time, args=(omega, K, N))

plt.figure(figsize=(10, 6))

# Plot each oscillator's phase evolution
for i in range(N):
    plt.plot(time, np.mod(theta_t[:, i], 2 * np.pi), label=f'Oscillator {i+1}')

plt.xlabel("Time")
plt.ylabel("Phase (radians)")
plt.title("Kuramoto Model: Phase Synchronization Over Time")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=8)
plt.grid()
plt.show()



# Compute order parameter r(t)
r_t = np.abs(np.sum(np.exp(1j * theta_t), axis=1) / N)

plt.figure(figsize=(8, 5))
plt.plot(time, r_t, color='b', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Order Parameter (r)")
plt.title("Kuramoto Model: Synchronization Order Parameter Over Time")
plt.grid()
plt.show()
