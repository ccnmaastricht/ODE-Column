import numpy as np
import torch
import pickle
from scipy.integrate import solve_ivp

# Define Subcritical Hopf Bifurcation Dynamics
def hopf_dynamics(t, state, t_eval, mu_vec, omega):
    x, y = state
    mu_t = np.interp(t, t_eval, mu_vec)  # Interpolate time-dependent mu
    dx = (mu_t + (x**2 + y**2)) * x - omega * y  # Subcritical Hopf
    dy = (mu_t + (x**2 + y**2)) * y + omega * x
    return [dx, dy]

# Constants
num_instances = 20  # Number of data instances
num_timepoints = 1000  # Number of time steps
t_span = (0, 25)
t_eval = np.linspace(*t_span, num_timepoints)  # 1000 time points
omega = 1.0  # Angular frequency

# Initialize an empty list to store data instances
dataset = []

# Generate 5000 instances
for _ in range(num_instances):
    # Generate time-varying mu (sinusoidal)
    a = 0.1  # Amplitude
    b = 0.25  # Frequency
    c = (np.random.rand(1) - 0.5) * 2 * np.pi  # Random phase shift
    d = 0
    mu_vec = (a / 2) * np.sin(t_eval * b + c) + (a / 2) + d  # Range [0, 1]
    mu_vec = np.clip(mu_vec, -1, 1)  # Keep within [-1, 1]

    # Random initial condition for x and y
    initial_state = np.random.rand(2) * 2 - 1  # Random in range [-1, 1]

    # Solve ODE
    sol = solve_ivp(hopf_dynamics, t_span, initial_state, t_eval=t_eval, args=(t_eval, mu_vec, omega))

    print(sol.success)

    # # Extract x and y
    # x_vals, y_vals = sol.y
    #
    # # Reshape and concatenate into (1000, 1, 3)
    # instance = np.stack([x_vals, y_vals, mu_vec], axis=1)  # Shape (1000, 3)
    # instance = instance[:, np.newaxis, :]  # Shape (1000, 1, 3)
    #
    # # Append to dataset
    # dataset.append(instance)

# Convert list to NumPy array and reshape to (1000, 5000, 3)
dataset = np.concatenate(dataset, axis=1)  # Shape (1000, 5000, 3)

# Convert to PyTorch tensor
dataset_tensor = torch.tensor(dataset, dtype=torch.float32)

# Save as pickle
with open("pickled_ds/ds_sub.pkl", "wb") as f:
    pickle.dump(dataset_tensor, f)

print(f"Dataset saved as 'hopf_trajectories.pkl' with shape {dataset_tensor.shape}")
