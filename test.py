import torch
import torch.nn as nn
from torchdiffeq import odeint


class NeuralODE(nn.Module):
    def __init__(self, num_neurons, dt):
        super().__init__()

        self.dt = dt  # Time step
        self.W = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.1)  # Learnable weights

        # Fixed parameters
        self.tau_s = torch.tensor(5.0)
        self.tau_m = torch.tensor(10.0)
        self.tau_a = torch.tensor(20.0)
        self.R_ = torch.tensor(1.5)
        self.kappa = torch.tensor(0.2)
        self.W_bg = torch.tensor(0.1)
        self.nu_bg = torch.tensor(1.0)

    def ode_func(self, t, state, stim):
        """Computes the derivatives for odeint"""
        I, H, A, R = state  # Unpack the state variables

        # Update current (I)
        dI = (-I / self.tau_s) + torch.matmul(self.W, R.detach()) + self.W_bg * self.nu_bg + stim

        # Update membrane potential (H) and adaptation (A)
        dH = (-H + self.R_ * dI) / self.tau_m
        dA = (-A + R * self.kappa) / self.tau_a

        # Update firing rate (R)
        dR = torch.tanh(H - A)  # Activation function

        return torch.stack([dI, dH, dA, dR])  # Return derivatives

    def forward(self, t, state, stim):
        """Integrate over time using odeint"""
        state_trajectory = odeint(lambda t, y: self.ode_func(t, y, stim), state, t)
        return state_trajectory


# Define time steps
t = torch.linspace(0, 10, steps=100)

# Initial state (I, H, A, R)
state_0 = torch.tensor([0.1, 0.0, 0.0, 0.1], dtype=torch.float32)

# Define stimulus (can be a function of time)
stim = torch.zeros(100)  # No external input

# Create model
model = NeuralODE(num_neurons=1, dt=0.1)

# Run the model
state_trajectory = model(t, state_0, stim)

# Extract firing rates (R)
R_values = state_trajectory[:, 3].detach()

# Plot results (optional)
import matplotlib.pyplot as plt
plt.plot(t, R_values, label="Firing Rate (R)")
plt.xlabel("Time")
plt.ylabel("Firing Rate")
plt.legend()
plt.show()

