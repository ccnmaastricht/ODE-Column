import os
import time
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchdiffeq import odeint, odeint_adjoint

import matplotlib.pyplot as plt

# Set params
nr_samples = 1000
epochs = 1
vis_training = True
total_time = 1000
batch_size = 16
batch_time = 100
nr_eval_points = 10
test_freq = batch_size
ds_file = 'ds_u_1000.pkl'


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
        # Set weights and bias for the third output unit to zero permanently
        # with torch.no_grad():
        #     self.net[2].weight[2].zero_()  # Zero out the third row of weights
        #     self.net[2].bias[2].zero_()  # Zero out the third unit's bias

        # is this code really necessary?
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, xyu):
        return self.net(xyu)
        # out = self.net(xyu)
        # out[:, 2] = 0  # Ensure the third output unit is always zero
        # return out

def val_ODE(func, t, true_y):
    with torch.no_grad():
        pred_y = odeint(func, true_y[0], t)
        loss = torch.mean(torch.abs(pred_y - true_y))
    return pred_y, loss


def time_varying_input(t, a, b, c, d):
    return a * torch.sin(t * b + c) + d

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
    s = torch.from_numpy(  # selects 20 random evaluation points along the trajectories
        np.random.choice(np.arange(total_time - batch_time, dtype=np.int64), nr_eval_points, replace=False))
    batch_y0 = true_y[s, :, :]  # (eval_points, training samples, dims)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i, :, :] for i in range(batch_time)], dim=0)  # (T, eval_points, training samples, dims)

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def make_ds(save=True):
    if os.path.exists(ds_file):
        with open(ds_file, 'rb') as f:
            ds = pickle.load(f)
    else:
        total_samples = int(nr_samples + (nr_samples / test_freq)) + 1  # +1 for good measure
        ds = torch.empty((total_time, total_samples, 3))

        for itr in range(total_samples):
            # Initialize true_y0
            y_0_x = np.random.uniform(-2, 2)
            y_0_y = np.random.uniform(-2, 2)
            # y_0_x = 2.0
            # y_0_y = 2.0

            # Sine wave params
            # a = np.random.uniform(0.1, 2.0)     # amplitude, between 0.1 and 2.0
            # b = np.random.uniform(0.1, 0.5)     # frequency, between 0.1 and 0.5
            # c = (torch.rand(1) - 0.5) * 2 * torch.pi      # horizontal shift
            # d = np.random.uniform(-0.5, 0.5)         # vertical shift, between -0.5 and 5
            a = 1
            b = 0.25
            c = 0
            d = 0

            # Make the sine wave with ODE
            true_y0 = torch.tensor([[y_0_x, y_0_y, time_varying_input(t_[0], a, b, c, d)]])
            with torch.no_grad():
                lambda_func = Lambda()
                true_y = odeint(lambda t, y: lambda_func(t, y, time_varying_input(t, a, b, c, d)),
                                true_y0, t_,
                                method='dopri5')  # use ODE to generate true_y using Lambda() as a function instead of a NN
            u_ = time_varying_input(t_, a, b, c, d).unsqueeze(1)  # substitute input u with the real input, not given by odeint
            true_y[:, :, 2] = u_

            # Add to ds
            assert ds[:, itr, :].shape == true_y[:, 0, :].shape
            ds[:, itr, :] = true_y[:, 0, :]

            # Save ds
            if save is True:
                with open(ds_file, 'wb') as f:
                    pickle.dump(ds, f)

    return ds[:, :nr_samples, :], ds[:, nr_samples:, :]

def freeze_third_unit(grad):
    grad[2] = 0  # Zero out the gradient for the third output unit
    return grad


### Run ###
t_ = torch.linspace(0., 25., total_time).to(device)
train_ds, val_ds = make_ds()

func = ODEFunc().to(device)                                             # this is the NN that odeint uses as a function
func.net[2].weight.register_hook(freeze_third_unit)                     # freeze weights of u output in NN
func.net[2].bias.register_hook(freeze_third_unit)

optimizer = optim.RMSprop(func.parameters(), lr=1e-3)                   # optimizer base class, should optimize NN parameters
ii = 0

for epoch in range(epochs):
    for itr in range(0, nr_samples, batch_size):
        start = time.time()
        optimizer.zero_grad()                                               # set gradients to zero (already used to update weights with loss.backward()

        # Train with batches
        if itr+batch_size < len(train_ds):
            true_y = train_ds[:, itr:itr+batch_size, :]
        else:
            true_y = train_ds[:, itr:, :]
        batch_y0, batch_t, batch_y = get_batch(true_y, t_)                  # we get 20 random data samples, each lasting 10 time steps (with set defaults)
        #pred_y = odeint(func, batch_y0, batch_t).to(device)                 # use the ODE with the NN as func to predict the next 10 time steps from the 20 samples
        start_odeint = time.time()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        odeint_time = time.time() - start_odeint
        #print(f"ODEint time: {odeint_time:.2f} seconds")

        loss = torch.mean(torch.abs(pred_y - batch_y))                      # compute the loss

        loss.backward()                                                     # compute gradients
        optimizer.step()                                                    # update parameters

        # Visualizing
        if vis_training:
            if ii < len(val_ds):
                val_true_y = val_ds[:, ii, :].unsqueeze(1)
            else:
                val_true_y = val_ds[:, -1, :].unsqueeze(1)
            val_pred_y, val_loss = val_ODE(func, t_, val_true_y)

            end = time.time() - start
            #print('Iter {:04d} | Total Loss {:.6f} | Time {:.2f}'.format(ii, val_loss.item(), end))
            visualize(val_true_y, val_pred_y, func, ii, val_loss.item(), loss.item())
            ii += 1
    ii = 0 # after epoch


# train_tensor = torch.tensor(train_ds)  # assuming `train_ds` is a tensor
# train_dataset = TensorDataset(train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# for epoch in range(epochs):
#     for batch_idx, (true_y,) in enumerate(train_loader):
#         start = time.time()
#         optimizer.zero_grad()
#
#         batch_y0, batch_t, batch_y = get_batch(true_y, t_)
#         pred_y = odeint(func, batch_y0, batch_t).to(device)
#         loss = torch.mean(torch.abs(pred_y - batch_y))
#
#         loss.backward()
#         optimizer.step()
#
#         # Visualizing
#         if vis_training:
#             if ii < len(val_ds):
#                 val_true_y = val_ds[:, ii, :].unsqueeze(1)
#             else:
#                 val_true_y = val_ds[:, -1, :].unsqueeze(1)
#             val_pred_y, val_loss = val_ODE(func, t_, val_true_y)
#
#             end = time.time() - start
#             print(f"Iter {ii:04d} | Total Loss {val_loss.item():.6f} | Time {end:.2f}")
#             visualize(val_true_y, val_pred_y, func, ii, val_loss.item(), loss.item())
#             ii += 1
#     ii = 0  # Reset after each epoch

