import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint, odeint_adjoint

import matplotlib.pyplot as plt

# Set params
niters = 1
vis_input = True
vis_training = False


# Running from terminal
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()


# Run on GPU if available
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


# Visualize and save the results
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if vis_training or niters < 10:
    makedirs('png')
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

def visualize(true_y, pred_y, odefunc, ii):
    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.plot(t_.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t_.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
                 'g-')
    ax_traj.plot(t_.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t_.cpu().numpy(),
                 pred_y.cpu().numpy()[:, 0, 1], 'b--')
    ax_traj.set_xlim(t_.cpu().min(), t_.cpu().max())
    ax_traj.set_ylim(-2, 2)

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')
    ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
    ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
    ax_phase.set_xlim(-2, 2)
    ax_phase.set_ylim(-2, 2)

    # ax_vecfield.cla()
    # ax_vecfield.set_title('Learned Vector Field')
    # ax_vecfield.set_xlabel('x')
    # ax_vecfield.set_ylabel('y')
    #
    # y, x = np.mgrid[-2:2:21j, -2:2:21j]
    # dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
    # mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
    # dydt = (dydt / mag)
    # dydt = dydt.reshape(21, 21, 2)
    #
    # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    # ax_vecfield.set_xlim(-2, 2)
    # ax_vecfield.set_ylim(-2, 2)

    fig.tight_layout()
    plt.savefig('png/{:03d}'.format(ii))
    plt.draw()
    plt.pause(0.001)


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, xyu):
        return self.net(xyu)

def test_ODE(func, true_y0, t, true_y):
    with torch.no_grad():
        pred_y = odeint(func, true_y0, t)
        loss = torch.mean(torch.abs(pred_y - true_y))
    return pred_y, loss


def time_varying_input(t, a, b, c, d):
    '''
    Define the time-varying input (sine wave)
    '''
    return a * torch.sin(t * b + c) + d
    # return torch.zeros((t.shape)) + 1
    # return 2 * torch.sigmoid((t - 3) / 3) - 1

def A_matrix(xy, u):
    '''
    Defines A, matrix that represents the underlying dynamics
    that have to be learned by the ODE
    '''
    x, y, u = xy[:,0].item(), xy[:,1].item(), u.item()
    A = torch.tensor([
        [(u - x**2 - y**2),     2.0],
        [-2.0,                  (u - x**2 - y**2)]
    ])
    return A

class Lambda(nn.Module):
    '''
    Used to generate the true_y trajectory based on a starting
    postion y0, a matrix A and input u
    '''
    def forward(self, t, xyu, u):
        xy = xyu[:,:-1]
        u = torch.tensor([[u]])
        forward_xy = torch.mm(xy, A_matrix(xy, u))  # apply matrix A
        forward_xyu = torch.concat((forward_xy, u), dim=1)
        return forward_xyu


def get_batch(true_y, t):
    '''
    Returns training batch of 20 samples
    '''
    s = torch.from_numpy(  # selects 20 random evaluation points along the trajectories
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, 1, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 1, D)

    batch_y0_vis = batch_y0.squeeze(1)
    batch_y0_vis = torch.cat((s.unsqueeze(1), batch_y0_vis), dim=1)
    batch_y0_vis = batch_y0_vis[batch_y0_vis[:, 0].argsort(dim=0)]

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_y0_vis



### Run ###

t_ = torch.linspace(0., 25., args.data_size).to(device)

func = ODEFunc().to(device)                                             # this is the NN that odeint uses as a function
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)                   # optimizer base class, should optimize NN parameters

end = time.time()
ii = 0

for itr in range(1, niters+1):
    optimizer.zero_grad()                                               # set gradients to zero (already used to update weights with loss.backward()

    # Initialize true_y
    y_0_x = np.random.uniform(-2, 2)
    y_0_y = np.random.uniform(-2, 2)

    a = np.random.uniform(0.1, 2.0)  # between 0.1 and 2.0
    b = np.random.uniform(0.1, 1.5)  # between 0.1 and 1.5
    c = (torch.rand(1) - 0.5) * 2 * torch.pi
    d = np.random.uniform(-0.5, 0.5)  # between -0.5 and 5

    true_y0 = torch.tensor([[y_0_x, y_0_y, time_varying_input(t_[0], a, b, c, d)]]).to(device)
    with torch.no_grad():
        lambda_func = Lambda()
        true_y = odeint(lambda t, y: lambda_func(t, y, time_varying_input(t, a, b, c, d)),
                        true_y0, t_, method='dopri5')                   # use ODE to generate true_y using Lambda() as a function instead of a NN
    u_ = time_varying_input(t_, a, b, c, d).unsqueeze(1)                            # substitute input u with the real input, not given by odeint
    true_y[:, :, 2] = u_

    # Train with batches
    batch_y0, batch_t, batch_y, batch_y0_vis = get_batch(true_y, t_)    # we get 20 random data samples, each lasting 10 time steps (with set defaults)
    pred_y = odeint(func, batch_y0, batch_t).to(device)                 # use the ODE with the NN as func to predict the next 10 time steps from the 20 samples
    loss = torch.mean(torch.abs(pred_y - batch_y))                      # compute the loss

    loss.backward()                                                     # compute gradients
    optimizer.step()                                                    # update parameters

    # Visualizing
    if vis_input:
        time_ax = t_ * args.data_size / t_[-1]

        fig = plt.figure(figsize=(8, 4), facecolor='white')
        plt.subplot(1, 2, 1)
        plt.plot(time_ax, true_y[:, 0, 0], time_ax, true_y[:, 0, 1])
        plt.scatter(batch_y0_vis[:, 0], batch_y0_vis[:, 1])
        plt.scatter(batch_y0_vis[:, 0], batch_y0_vis[:, 2])

        plt.subplot(1, 2, 2)
        plt.plot(t_.detach().numpy(), time_varying_input(t_, a, b, c, d))
        plt.plot(t_.detach().numpy(), true_y[:, 0, 2])

        pred_y, loss = test_ODE(func, true_y0, t_, true_y)
        visualize(true_y, pred_y, func, ii)

        plt.show()

    if vis_training:
        if itr % args.test_freq == 0:
            pred_y, loss = test_ODE(func, true_y0, t_, true_y)
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            visualize(true_y, pred_y, func, ii)
            ii += 1
    end = time.time()

