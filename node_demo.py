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
from fastdtw import fastdtw

from torchdiffeq import odeint


# Set params
ds_file          = 'ds_uc_5000_omg.pkl'
nr_samples       = 5000
batch_size       = 32
total_time       = 1000
batch_time       = 200
nr_eval_points   = 5
test_freq        = 3 # test after every n batches
vis_training     = True
epochs           = 1


# Run on GPU 0 if available
device = torch.device('cpu')

# Visualize and save the results
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def visualize(true_y, pred_y, odefunc, ii, val_loss, train_loss):
    makedirs('png')
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_input = fig.add_subplot(131, frameon=False)
    ax_traj = fig.add_subplot(132, frameon=False)
    ax_phase = fig.add_subplot(133, frameon=False)
    #ax_vecfield = fig.add_subplot(133, frameon=False)

    fig.text(0.4, 0.03, f"Validation loss: {val_loss:.4f}", ha='center', fontsize=12, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.4f}", ha='center', fontsize=12, fontweight='bold')

    ax_input.cla()
    ax_input.set_title('Input')
    ax_input.plot(t_.detach().numpy(), true_y[:, 0, 2])
    ax_input.set_xlim(t_.cpu().min(), t_.cpu().max())
    ax_input.set_ylim(-3, 3)

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    #ax_traj.set_xlabel('t')
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
    plt.close(fig)


# Loss functions
def huber_loss(y_pred, y_true):
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)
    return hub_loss(y_pred, y_true)

def dtw_loss(y_pred, y_true):
  y_pred, y_true = y_pred.unsqueeze(0), y_true.unsqueeze(0)
  batch_size = y_pred.shape[0]
  loss = 0.0
  for i in range(batch_size):
      pred_seq = y_pred[i].detach().cpu().numpy().squeeze()
      true_seq = y_true[i].detach().cpu().numpy().squeeze()
      distance, _ = fastdtw(pred_seq, true_seq, dist=2)  # Using Minkowski distance with p=2
      loss += distance
  return torch.tensor(loss / batch_size, requires_grad=True)

def fourier_loss(y_pred, y_true):
    fft_pred = torch.fft.fft(y_pred, dim=-1)
    fft_true = torch.fft.fft(y_true, dim=-1)
    loss = torch.mean(torch.abs(torch.abs(fft_pred) - torch.abs(fft_true)))  # Magnitude difference
    return loss


# ODE stuff
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
            # nn.Linear(100, 100),
            # nn.ReLU(),
            # nn.Linear(100, 100),
            # nn.ReLU(),
            nn.Linear(100, 3)
        )
        # Set weights and bias for the third output unit to zero permanently
        # with torch.no_grad():
        #     self.net[2].weight[2].zero_()  # Zero out the third row of weights
        #     self.net[2].bias[2].zero_()  # Zero out the third unit's bias

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, xy, time_vec, mu_vec):
        okay = 0
        mu_t = np.interp(t, time_vec, mu_vec)
        return self.net(torch.concat((xy, mu_t), dim=-1))
        # return self.net(xyu)
        # out = self.net(xyu)
        # out[:, 2] = 0  # Ensure the third output unit is always zero
        # return out

def val_ODE(func, t, true_y):
    with torch.no_grad():
        pred_y = odeint(func, true_y[0], t)
        loss = huber_loss(pred_y, true_y)
    return pred_y, loss

def freeze_third_unit(grad):
    grad[2] = 0  # Zero out the gradient for the third output unit
    return grad


# Make training data
def get_batch(true_y, t):
    s = torch.from_numpy(  # selects 20 random evaluation points along the trajectories
        np.random.choice(np.arange(total_time - batch_time, dtype=np.int64), nr_eval_points, replace=False))
    batch_y0 = true_y[s, :, :]  # (eval_points, training samples, dims)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i, :, :] for i in range(batch_time)], dim=0)  # (T, eval_points, training samples, dims)

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def dynamics(t, state, time_vec, mu_vec, omega):
    x, y = state[0]
    # interpolate mu depending on the current time
    mu_t = np.interp(t, time_vec, mu_vec)
    dx = (mu_t - x**2 - y**2) * x - omega.item() * y
    dy = (mu_t - x**2 - y**2) * y + omega.item() * x
    return torch.tensor([dx, dy])

def make_ds(save=True):
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
            a = 1 # np.random.uniform(0.1, 2.0)             # amplitude, between 0.1 and 2.0
            b = 0.25 #np.random.uniform(0.1, 0.5)           # frequency, between 0.1 and 0.5
            c = (torch.rand(1) - 0.5) * 2 * torch.pi        # horizontal shift
            d = 0 # np.random.uniform(-0.5, 0.5)            # vertical shift, between -0.5 and 5

            # Sine wave and 'oscillation speed'
            mu = a * torch.sin(t_ * b + c) + d
            omega = torch.tensor([1.0])

            # Create true_y
            true_y = odeint(lambda t, y: dynamics(t, y, t_, mu, omega), true_y0, t_, method='dopri5')
            true_y = torch.cat((true_y, mu.unsqueeze(-1).unsqueeze(-1)), dim=-1)

            # Add to ds
            assert ds[:, itr, :].shape == true_y[:, 0, :].shape
            ds[:, itr, :] = true_y[:, 0, :]

        # Save ds
        if save is True:
            with open(ds_file, 'wb') as f:
                pickle.dump(ds, f)

    return ds[:, :nr_samples, :], ds[:, nr_samples:, :]



### Run ###

t_ = torch.linspace(0., 25., total_time).to(device)

train_ds, val_ds = make_ds()
train_ds = train_ds.transpose(0, 1)

train_dataset = TensorDataset(train_ds)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

nn = ODEFunc().to(device)                                             # this is the NN that odeint uses as a function
nn.net[2].weight.register_hook(freeze_third_unit)                     # freeze weights of u output in NN
nn.net[2].bias.register_hook(freeze_third_unit)

optimizer = optim.RMSprop(nn.parameters(), lr=1e-3)                   # optimizer base class, should optimize NN parameters
ii = 0

for epoch in range(epochs):
    for batch_idx, (true_y,) in enumerate(train_loader):
        start = time.time()
        optimizer.zero_grad()                                               # set gradients to zero (already used to update weights with loss.backward()

        true_y = true_y.clone().detach().transpose(0, 1)
        batch_y0, batch_t, batch_y = get_batch(true_y, t_)
        #
        # start_odeint = time.time()
        # pred_y = odeint(nn, batch_y0, batch_t).to(device)                 # use the ODE with the NN as func
        # odeint_time = time.time() - start_odeint
        # #print(f"ODEint time: {odeint_time:.2f} seconds")
        #
        # loss = huber_loss(pred_y, batch_y)                      # compute the loss

        true_y0 = true_y[0, :, :]  # run with entire trajectory, no time batches
        start_y = true_y0[:, :2]
        true_y, true_mu = true_y[:, :, :2], true_y[:, :, 2]

        pred_y = odeint(lambda t, y: nn(t, y, t_, true_mu), start_y, t_)
        # pred_y = odeint(nn, true_y0, t_).to(device)
        loss = huber_loss(pred_y, true_y)

        loss.backward()                                                     # compute gradients
        optimizer.step()                                                    # update parameters

        # Visualizing
        if vis_training and batch_idx % test_freq == 0:
            if ii < len(val_ds):
                val_true_y = val_ds[:, ii, :].unsqueeze(1)
            else:
                val_true_y = val_ds[:, -1, :].unsqueeze(1)
            val_pred_y, val_loss = val_ODE(nn, t_, val_true_y)

            end = time.time() - start
            print('Iter {:04d} | Total Loss {:.6f} | Time {:.2f}'.format(ii, val_loss.item(), end))
            visualize(val_true_y, val_pred_y, nn, ii, val_loss.item(), loss.item())
            ii += 1
    ii = 0 # after epoch

