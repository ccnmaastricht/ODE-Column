""" Demonstrates the easy of integration of a custom layer """
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from DMF import get_params


def make_states(num=100, M=16):
    states = torch.zeros((num, M, 4), dtype=torch.float32)
    for i in range(num):
        states[i, :, 0] = torch.rand(M, dtype=torch.float32) - 0.5
        states[i, :, 1] = torch.rand(M, dtype=torch.float32) - 0.5
        states[i, :, 2] = torch.rand(M, dtype=torch.float32) - 0.5
        states[i, :, 3] = torch.rand(M, dtype=torch.float32)
    return states

class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()

        self.W = nn.Parameter(torch.tensor(params['W'], dtype=torch.float32))
        self.dt = torch.tensor(params['dt'], dtype=torch.float32)
        self.tau_s = torch.tensor(params['tau_s'], dtype=torch.float32)
        self.W_bg = torch.tensor(params['W_bg'], dtype=torch.float32)
        self.nu_bg = torch.tensor(params['nu_bg'], dtype=torch.float32)
        self.R_ = torch.tensor(params['R'], dtype=torch.float32)  # not to be confused with state['R']
        self.tau_m = torch.tensor(params['tau_m'], dtype=torch.float32)
        self.kappa = torch.tensor(params['kappa'], dtype=torch.float32)
        self.tau_a = torch.tensor(params['tau_a'], dtype=torch.float32)

        self.state = {
        'I': torch.zeros(M, dtype=torch.float32),
        'H': torch.zeros(M, dtype=torch.float32),
        'A': torch.zeros(M, dtype=torch.float32),
        'R': torch.ones(M, dtype=torch.float32)
        }

    def forward(self, x, stim):

        self.state['I'] = x + self.dt * (-x / self.tau_s)  # self inhibition
        self.state['I'] = self.state['I'] + self.dt * (torch.matmul(self.W, self.state['I']))  # recurrent input
        self.state['I'] = self.state['I'] + self.dt * self.W_bg * self.nu_bg  # background input
        self.state['I'] = self.state['I'] + self.dt * stim  # external output

        return self.state['I']


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = MyLinearLayer()

    def forward(self, x, stim):
        return self.linear(x, stim)


params = get_params(J_local=0.13, J_lateral=0.172, area='MT')
params['dt'] = 1e-4  # timestep
M = params['M']  # num of populations

model = BasicModel()
x, y = make_states(10, 16), make_states(10, 16)

state = {
        'I': torch.zeros(M, dtype=torch.float32),
        'H': torch.zeros(M, dtype=torch.float32),
        'A': torch.zeros(M, dtype=torch.float32),
        'R': torch.ones(M, dtype=torch.float32)
}

# model.linear.state = state

stim = torch.ones(M, dtype=torch.float32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for i in range(10):
    optimizer.zero_grad()

    output = model(state['I'], stim)
    loss = torch.mean(torch.abs(output - y[i, :, 0]))

    print(output)
    print(loss)
    # plt.imshow(model.linear.W.detach().numpy(), cmap="viridis", interpolation="nearest")
    # plt.show()

    loss.backward()
    optimizer.step()



'''

class ThresholdFiringRate(nn.Module):
    def __init__(self):
        super(ThresholdFiringRate, self).__init__()
        self.a = torch.tensor(params['a'], dtype=torch.float32)    # gain
        self.b = torch.tensor(params['b'], dtype=torch.float32)    # threshold
        self.d = torch.tensor(params['d'], dtype=torch.float32)    # noise factor

    def forward(self, x):
        x_nom = self.a * x - self.b
        x_activ = x_nom / (1 - torch.exp(-self.d * x_nom))

        return x_activ


class DMFdynamics(nn.Module):
    def __init__(self):
        super(DMFdynamics, self).__init__()

        self.W = nn.Parameter(torch.tensor(params['W'], dtype=torch.float32, requires_grad=True))
        self.dt     = torch.tensor(params['dt'], dtype=torch.float32)
        self.tau_s  = torch.tensor(params['tau_s'], dtype=torch.float32)
        self.W_bg   = torch.tensor(params['W_bg'], dtype=torch.float32)
        self.nu_bg  = torch.tensor(params['nu_bg'], dtype=torch.float32)
        self.R_     = torch.tensor(params['R'], dtype=torch.float32)  # not to be confused with state['R']
        self.tau_m  = torch.tensor(params['tau_m'], dtype=torch.float32)
        self.kappa  = torch.tensor(params['kappa'], dtype=torch.float32)
        self.tau_a  = torch.tensor(params['tau_a'], dtype=torch.float32)

    def forward(self, x, stim):

        # Update the current
        x['I'] = x['I'] + self.dt * (-x['I'] / self.tau_s)  # self inhibition
        x['I'] = x['I'] + self.dt * (torch.matmul(self.W, x['R']))  # recurrent input
        x['I'] = x['I'] + self.dt * self.W_bg * self.nu_bg  # background input
        x['I'] = x['I'] + self.dt * stim  # external output

        # Update the membrane potential and adaptation
        x['H'] = x['H'] + self.dt * ((-x['H'] + self.R_ * x['I']) / self.tau_m)
        x['A'] = x['A'] + self.dt * ((-x['A'] + x['R'] * self.kappa) / self.tau_a)

        return x



class TwoColumnODE(nn.Module):
    def __init__(self):
        super(TwoColumnODE, self).__init__()
        self.dmf_layer = DMFdynamics()
        # self.lin_layer = nn.Linear(16, 16)
        self.activation = ThresholdFiringRate()

        # nn.init.normal_(self.lin_layer.weight, mean=0, std=0.1)
        # nn.init.constant_(self.lin_layer.bias, val=0)

    def forward(self, x, stim):
        x = self.dmf_layer(x, stim)  # update current, mem_pot, and adaptation
        # x['H'] = self.lin_layer(x['H'])
        x['R'] = self.activation(x['H'] - x['A'])  # compute the firing rate
        return x

'''

