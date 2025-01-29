import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

import matplotlib.pyplot as plt


# Set params
niters = 2000
vis_input = False
vis_training = True


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
    ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
                 'g-')
    ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(),
                 pred_y.cpu().numpy()[:, 0, 1], 'b--')
    ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
    ax_traj.set_ylim(-2, 2)

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')
    ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
    ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
    ax_phase.set_xlim(-2, 2)
    ax_phase.set_ylim(-2, 2)

    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    y, x = np.mgrid[-2:2:21j, -2:2:21j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
    mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-2, 2)
    ax_vecfield.set_ylim(-2, 2)

    fig.tight_layout()
    plt.savefig('png/{:03d}'.format(ii))
    plt.draw()
    plt.pause(0.001)


# Track time and loss while training
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# Get training batch
def get_batch(true_y, t):
    # default data_size=1000
    # default batch_time=10
    # default batch_size=20
    s = torch.from_numpy(  # selects 20 random evaluation points along the trajectories
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, 1, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 1, D)

    batch_y0_vis = batch_y0.squeeze(1)
    batch_y0_vis = torch.cat((s.unsqueeze(1), batch_y0_vis), dim=1)
    batch_y0_vis = batch_y0_vis[batch_y0_vis[:, 0].argsort(dim=0)]

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_y0_vis


# ODE network
class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        x = torch.zeros(y.shape) + 1  # for now, specify the input here
        y_x = torch.cat((y, x), dim=-1)
        return self.net(y_x)
        # return self.net(y)


# Test ODE
def test_ODE(func, true_y0, t, true_y):
    with torch.no_grad():
        pred_y = odeint(func, true_y0, t)
        loss = torch.mean(torch.abs(pred_y - true_y))
    return pred_y, loss


# Initialize the true data (spiral)
true_y0 = torch.tensor([[2., 0.]]).to(device)                           # this is the initial condition - starting point to compute true_y
t = torch.linspace(0., 25., args.data_size).to(device)
# true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)           # these are the dynamics that change y (and form spiral)
# true_A = torch.tensor([[0.0, 2.0], [-2.0, 0.0]]).to(device)             # spiraling trajectory that does not get smaller (so circle :)
# x = torch.ones((args.data_size, 1)).to(device)
X = torch.zeros((1, 1)).to(device) + 1                                  # represents the input

def A_matrix(x, y, u):
    A = torch.tensor([
        [(u - x**2 - y**2),     1.0],
        [-1.0,                  (u - x**2 - y**2)]
    ])
    return A

class Lambda(nn.Module):
    def forward(self, t, y):
        y_1 = y[:,0].item()  # 1st item of y
        y_2 = y[:,1].item()  # 2nd item of y
        u   = X[:,0].item()  # input
        return torch.mm(y, A_matrix(y_1, y_2, u))

with torch.no_grad():  # no gradients are computed, because we don't need to train anything
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')              # use ODE to generate true_y using Lambda() as a function instead of a NN



### Run ###

func = ODEFunc().to(device)                                             # this is the NN that specifies the differential equations function
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)                   # optimizer base class, should optimize NN parameters

time_meter = RunningAverageMeter(0.97)
loss_meter = RunningAverageMeter(0.97)
end = time.time()
ii = 0

for itr in range(1, niters+1):
    optimizer.zero_grad()                                               # set gradients to zero (already used to update weights with loss.backward()
    batch_y0, batch_t, batch_y, batch_y0_vis = get_batch(true_y, t)     # we get 20 random data samples, each lasting 10 time steps (with set defaults)

    pred_y = odeint(func, batch_y0, batch_t).to(device)                 # use the ODE with the NN as func to predict the next 10 time steps from the 20 samples
    loss = torch.mean(torch.abs(pred_y - batch_y))                      # compute the loss
    loss.backward()                                                     # compute gradients
    optimizer.step()                                                    # update parameters

    # Tracking and visualizing
    time_meter.update(time.time() - end)
    loss_meter.update(loss.item())

    if vis_input:
        time_ax = t * args.data_size / t[-1]

        fig = plt.figure(figsize=(8, 4), facecolor='white')
        plt.subplot(1, 2, 1)
        plt.plot(time_ax, true_y[:, 0, 0], time_ax, true_y[:, 0, 1])
        plt.scatter(batch_y0_vis[:, 0], batch_y0_vis[:, 1])
        plt.scatter(batch_y0_vis[:, 0], batch_y0_vis[:, 2])

        plt.subplot(1, 2, 2)
        pred_y, loss = test_ODE(func, true_y0, t, true_y)
        plt.plot(t, true_y[:, 0, 0], t, true_y[:, 0, 1], 'g-')
        plt.plot(t, pred_y[:, 0, 0], '--', t, pred_y[:, 0, 1], 'b--')

        visualize(true_y, pred_y, func, ii)

        plt.show()

    if vis_training:
        if itr % args.test_freq == 0:
            pred_y, loss = test_ODE(func, true_y0, t, true_y)
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            visualize(true_y, pred_y, func, ii)
            ii += 1
    end = time.time()

