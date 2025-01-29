import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint, odeint_adjoint
import matplotlib.pyplot as plt


# Define the Neural Network model
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.hidden = nn.Linear(2, 50)
        self.output = nn.Linear(50, 1)

    def forward(self, t, y, u):
        y = y.view(1, 1)
        u = u.view(1, 1)
        combined_input = torch.cat([y, u], dim=-1)
        hidden_output = torch.relu(self.hidden(combined_input))
        dydt = self.output(hidden_output)
        return dydt


# Time points for evaluation
t_points = torch.linspace(0, 10, 1000)


# True solution (sine wave)
def true_solution(t):
    return torch.sin(t)


# Define the time-varying input (sine wave)
def time_varying_input(t):
    return torch.sin(t)


# Define the initial condition
y0 = torch.tensor([1.0])

# Initialize the neural network
ode_func = ODEFunc()

# Define the optimizer and loss function
optimizer = optim.Adam(ode_func.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Solve the ODE
    solution = odeint(lambda t, y: ode_func(t, y, time_varying_input(t)), y0, t_points)

    # Compute the loss (mean squared error with the true sine wave)
    loss = criterion(solution.squeeze(), true_solution(t_points))

    # Backpropagate and update weights
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# After training, plot the results
solution_np = solution.detach().numpy()
t_points_np = t_points.numpy()

plt.figure(figsize=(10, 6))
plt.plot(t_points_np, solution_np, label='Trained Neural ODE Solution', color='b', linestyle='-', linewidth=2)
plt.plot(t_points_np, torch.sin(t_points).numpy(), label='True Solution (sin(t))', color='r', linestyle='--',
         linewidth=2)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Trained Neural ODE Solution vs True Solution', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
