import numpy as np
import torch
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
import matplotlib.pyplot as plt

def dynamics(state, t, time_vec, mu_vec, omega):
  x, y = state
  # interpolate mu depending on the current time
  mu_t = np.interp(t, time_vec, mu_vec)
  dx = (mu_t - x**2 - y**2) * x - omega * y
  dy = (mu_t - x**2 - y**2) * y + omega * x
  return (dx, dy)

omega = 1.0

total_time = 25.
time_steps = 1000
t = torch.linspace(0, total_time, time_steps)
mu = torch.sin(t * 0.25)

y0 = [0, 1]

parameters = (t, mu, omega)
state = solve_ivp(dynamics, (0, total_time), y0, args=parameters)

plt.plot(mu)
plt.show()
plt.plot(state)
plt.show()


# Some extra code no longer using

# # Training data (true y)
# def time_varying_input(t, a, b, c, d):
#     return a * torch.sin(t * b + c) + d
#
# def A_matrix(xy, u):
#     x, y, u = xy[:,0].item(), xy[:,1].item(), u.item()
#     A = torch.tensor([
#         [(u - x**2 - y**2),     2.0],
#         [-2.0,                  (u - x**2 - y**2)]
#     ])
#     return A
#
# class Lambda(nn.Module):
#     def forward(self, t, xyu, u):
#         xy = xyu[:,:-1]
#         u = torch.tensor([[u]])
#         forward_xy = torch.mm(xy, A_matrix(xy, u))  # apply matrix A
#         forward_xyu = torch.concat((forward_xy, u), dim=1)
#         return forward_xyu
