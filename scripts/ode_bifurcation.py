import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.fft

from torchdiffeq import odeint
from scipy.integrate import solve_ivp



def visualize_hopf(true_y, pred_y, ii, val_loss, train_loss):
    if not os.path.exists('..results/png'):
        os.makedirs('..results/png')
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_input = fig.add_subplot(131, frameon=False)
    ax_traj = fig.add_subplot(132, frameon=False)
    ax_phase = fig.add_subplot(133, frameon=False)

    fig.text(0.4, 0.03, f"Validation loss: {val_loss:.4f}", ha='center', fontsize=12, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.4f}", ha='center', fontsize=12, fontweight='bold')

    ax_input.cla()
    ax_input.set_title('Input')
    ax_input.plot(time_vec.detach().numpy(), true_y[:, 0, 2])
    ax_input.set_xlim(time_vec.cpu().min(), time_vec.cpu().max())
    ax_input.set_ylim(-3, 3)

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_ylabel('x,y')
    ax_traj.plot(time_vec.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], time_vec.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
                 'g-')
    ax_traj.plot(time_vec.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', time_vec.cpu().numpy(),
                 pred_y.cpu().numpy()[:, 0, 1], 'b--')
    ax_traj.set_xlim(time_vec.cpu().min(), time_vec.cpu().max())
    ax_traj.set_ylim(-2, 2)

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')
    ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
    ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
    ax_phase.set_xlim(-2, 2)
    ax_phase.set_ylim(-2, 2)

    fig.tight_layout()
    plt.savefig('../results/png/{:03d}'.format(ii))
    plt.close(fig)


# Loss functions
def huber_loss(y_pred, y_true):
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)
    return hub_loss(y_pred, y_true)

def mse(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


# ODE stuff
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y, time_vec, mu_vec):
        mu_t = torch.tensor([np.interp(
                            t.detach().numpy(), time_vec.detach().numpy(), mu_vec[:, i].detach().numpy())
                            for i in range(len(mu_vec[0]))]).unsqueeze(-1)  # get mu at timepoint t
        y_mu = torch.concat((y, mu_t), dim=-1).float()
        return self.net(y_mu)

def val_ODE(nn, t_, true_y):
    with torch.no_grad():
        start_y = true_y[0, :, :2]
        true_y, true_mu = true_y[:, :, :2], true_y[:, :, 2]
        pred_y = odeint(lambda t, y: nn(t, y, t_, true_mu), start_y, t_)

        loss = huber_loss(pred_y, true_y)
    return pred_y, loss


# Time batches - no longer in use
def get_time_batch(true_y, t, batch_time, nr_eval_points):
    s = torch.from_numpy(  # selects 20 random evaluation points along the trajectories
        np.random.choice(np.arange(total_time - batch_time, dtype=np.int64), nr_eval_points, replace=False))
    batch_y0 = true_y[s, :, :]  # (eval_points, training samples, dims)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i, :, :] for i in range(batch_time)], dim=0)  # (T, eval_points, training samples, dims)
    return batch_y0, batch_t, batch_y


# Make Hopf bifurcation training data
def dynamics(t, state, time_vec, mu_vec, omega):
    x, y = state[0]
    # interpolate mu depending on the current time
    mu_t = np.interp(t, time_vec, mu_vec)
    dx = (mu_t - x**2 - y**2) * x - omega.item() * y
    dy = (mu_t - x**2 - y**2) * y + omega.item() * x
    return torch.tensor([dx, dy])

def make_hopf_ds(save=True):
    if os.path.exists(ds_file):
        with open(ds_file, 'rb') as f:
            ds = pickle.load(f)
    else:
        total_samples = int(nr_samples + (nr_samples/batch_size)/test_freq) + 1  # +1 for good measure
        ds = torch.empty((total_time, total_samples, 3))

        for itr in range(total_samples):
            # Initialize true_y0
            y_0_x = np.random.uniform(-2, 2)
            y_0_y = np.random.uniform(-2, 2)
            true_y0 = torch.tensor([[y_0_x, y_0_y]])

            # Sine wave params
            a = 1.                                          # amplitude
            b = 0.25                                        # frequency
            c = (torch.rand(1) - 0.5) * 2 * torch.pi        # horizontal shift
            d = 0.                                          # vertical shift

            # Sine wave and 'oscillation speed'
            mu = a * torch.sin(time_vec * b + c) + d
            omega = torch.tensor([1.0])

            # Create true_y
            true_y = odeint(lambda t, y: dynamics(t, y, time_vec, mu, omega), true_y0, time_vec, method='dopri5')
            true_y = torch.cat((true_y, mu.unsqueeze(-1).unsqueeze(-1)), dim=-1)

            # Add to ds
            assert ds[:, itr, :].shape == true_y[:, 0, :].shape
            ds[:, itr, :] = true_y[:, 0, :]

        # Save ds
        if save is True:
            with open(ds_file, 'wb') as f:
                pickle.dump(ds, f)

    return ds[:, :nr_samples, :], ds[:, nr_samples:, :]



if __name__ == '__main__':

    # set params for hopf bifurcation
    ds_file          = '../data/ds_bifurcation.pkl'
    nr_samples       = 5000
    batch_size       = 32
    total_time       = 1000
    test_freq        = 3  # test after every n batches
    vis_training     = True

    time_vec = torch.linspace(0., 25., total_time)

    train_ds, val_ds = make_hopf_ds()
    train_ds = train_ds.transpose(0, 1)

    train_dataset = TensorDataset(train_ds)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    nn = ODEFunc()  # this is the NN that odeint uses as a function
    optimizer = optim.RMSprop(nn.parameters(), lr=1e-3)

    ii = 0

    for batch_idx, (true_y,) in enumerate(train_loader):
        start = time.time()
        optimizer.zero_grad()

        true_y = true_y.clone().detach().transpose(0, 1)

        start_y = true_y[0, :, :2]
        true_y, true_mu = true_y[:, :, :2], true_y[:, :, 2]

        pred_y = odeint(lambda t, y: nn(t, y, time_vec, true_mu), start_y, time_vec)
        loss = huber_loss(pred_y, true_y)

        loss.backward()
        optimizer.step()

        # Testing and visualizing
        if vis_training and batch_idx % test_freq == 0:
            if ii < len(val_ds):
                val_true_y = val_ds[:, ii, :].unsqueeze(1)
            else:
                val_true_y = val_ds[:, -1, :].unsqueeze(1)
            val_pred_y, val_loss = val_ODE(nn, time_vec, val_true_y)

            end = time.time() - start
            print('Iter {:04d} | Total Loss {:.6f} | Time {:.2f}'.format(ii, val_loss.item(), end))
            visualize_hopf(val_true_y, val_pred_y, ii, val_loss.item(), loss.item())
            ii += 1

