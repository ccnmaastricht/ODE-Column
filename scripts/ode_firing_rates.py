import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchdiffeq import odeint

from DMF import *
from DMF_single import *
from ode_bifurcation import ODEFunc, huber_loss, mse


# Set params
ds_file             = 'pickled_ds/dmf_fr_5000.pkl'
nr_samples          = 5000          # nr of training samples
T                   = 1000          # total time
dt                  = 1e-3          # timestep
batch_size          = 32
test_freq           = 3             # after n batches
vis_testing         = True
device              = torch.device('cpu')


# Generate stimulus data
def make_stim(stim, T, dt):
    t = np.arange(0, T * dt, dt)

    # Sine wave params
    amplitude = 1
    frequency = 1
    shift = np.random.uniform(0, 2 * np.pi)  # random horizontal shift

    # Sine wave
    sine = amplitude * np.sin(t * frequency * 2 * np.pi + shift)

    stim[2] = stim[2] * sine
    stim[3] = stim[3] * sine
    return stim, sine

def make_fr_stim(fr_input, stim):
    stim[2] = stim[2] * np.array(fr_input)
    stim[3] = stim[3] * np.array(fr_input)
    return stim, fr_input


def make_ds(save=True, firing_rate_input=False):
    if os.path.exists(ds_file):
        with open(ds_file, 'rb') as f:
            ds = pickle.load(f)
    else:
        ds_len = nr_samples + int(nr_samples/10)
        ds = torch.empty((ds_len, T, 3))

        if firing_rate_input:  # load previously made firing rates to use as input
            with open('pickled_ds/dmf_5000.pkl', 'rb') as f:
                fr_ds = pickle.load(f)

        for i in range(ds_len):  # add extra test samples
            # Initialize the column model
            column = SingleColumnDMF(area='MT')

            # Make a stimulus input
            stim = np.zeros((column.params['M'], T))

            if firing_rate_input:  # use firing rates as input
                fr = fr_ds[i, :, 0]  # slice ~excitatory~ firing rate
                stim = column.set_stim_ext(stim=stim, nu=1.0, params=column.params)
                stim, input = make_fr_stim(fr, stim)
            else:  # sine wave input
                stim = column.set_stim_ext(stim=stim, nu=20.0, params=column.params)
                stim, input = make_stim(stim, T, dt)

            # Simulate the stimulation
            firing_rates = column.simulate(stim=stim, T=T, state_var='R', vis=False)

            # Slice only layer 4 and add the original input (sine wave)
            layer_4 = firing_rates[2:4, :]
            layer_4_w_input = np.vstack([layer_4, input.reshape(1, T)])

            # Cast to tensor, transpose and add to ds
            layer_4_w_input_T = torch.tensor(layer_4_w_input).transpose(0, 1)
            ds[i, :, :] = layer_4_w_input_T

        # Save ds
        if save is True:
            with open(ds_file, 'wb') as f:
                pickle.dump(ds, f)
    return ds


def minmax_norm(ds, nr_indices):
    for activ in range(nr_indices):
        firing_rates_min = ds[:, :, activ].min()
        firing_rates_max = ds[:, :, activ].max()
        ds[:, :, activ] = (ds[:, :, activ] - firing_rates_min) / (firing_rates_max - firing_rates_min)
    return ds


def compute_loss(pred_y, true_y):
    nr_batch_samples = pred_y.shape[1]
    loss = 0
    for i in range(nr_batch_samples):
        loss += huber_loss(pred_y[:, i, :], true_y[:, i, :])
    return loss/nr_batch_samples


def test_ODE(ode_model, t_, test_ds, test_itr):
    # Get test sample
    if test_itr < len(test_ds):
        true_y_input = test_ds[test_itr, :, :].unsqueeze(1)
    else:
        true_y_input = test_ds[-1, :, :].unsqueeze(1)
    true_y_input = true_y_input.clone().detach()

    # Apply ODE model to test sample
    with torch.no_grad():
        start_y = true_y_input[0, :, :2]
        true_y, input = true_y_input[:, :, :2], true_y_input[:, :, 2]
        pred_y = odeint(lambda t, y: ode_model(t, y, t_, input), start_y, t_)

        loss = compute_loss(pred_y, true_y)
    return true_y_input, pred_y, loss


def visualize_results(true_y, pred_y, t_, val_loss, train_loss, itr):
    if not os.path.exists('results/png'):
        os.makedirs('results/png')

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_input = fig.add_subplot(131, frameon=False)
    ax_exci = fig.add_subplot(132, frameon=False)
    ax_inhi = fig.add_subplot(133, frameon=False)
    plt.subplots_adjust(bottom=0.3)

    fig.text(0.4, 0.03, f"Validation loss: {val_loss:.4f}", ha='center', fontsize=12, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.4f}", ha='center', fontsize=12, fontweight='bold')

    ax_input.cla()
    ax_input.set_title('Input')
    ax_input.plot(t_.detach().numpy(), true_y[:, 0, 2])
    ax_input.set_xlim(t_.cpu().min(), t_.cpu().max())
    ax_input.set_ylim(-1, 1)

    ax_exci.cla()
    ax_exci.set_title('Excitatory')
    ax_exci.plot(t_.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t_.cpu().numpy(),
                 pred_y.cpu().numpy()[:, 0, 0], '--')
    ax_exci.set_xlim(t_.cpu().min(), t_.cpu().max())
    ax_exci.set_ylim(-1, 1)

    ax_inhi.cla()
    ax_inhi.set_title('Inhibitory')
    ax_inhi.plot(t_.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], t_.cpu().numpy(),
                 pred_y.cpu().numpy()[:, 0, 1], '--')
    ax_inhi.set_xlim(t_.cpu().min(), t_.cpu().max())
    ax_inhi.set_ylim(-1, 1)

    fig.tight_layout(pad=3)
    plt.savefig('results/png/{:03d}'.format(itr))
    plt.close(fig)


if __name__ == '__main__':

    # Initialize time and dataset (load existing or make new one)
    t_ = torch.linspace(0., T*dt, T).to(device)
    ds = make_ds(firing_rate_input=True)
    t_, ds = t_[:500], ds[:, :500, :] # take the first 500 timesteps instead of all 1000

    # Normalize ds, separately for excitatory and inhibitory activations
    # !!! Also normalizing firing rate input (index 3)
    ds = minmax_norm(ds, 3)

    # Split train/test and transfer training data to dataloader
    train_ds, test_ds = ds[:nr_samples], ds[nr_samples:]
    train_ds = TensorDataset(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Initialize neural ODE and optimizer
    ode_model = ODEFunc().to(device)
    optimizer = optim.RMSprop(ode_model.parameters(), lr=1e-3)

    test_itr = 0

    for batch_idx, (true_y,) in enumerate(train_loader):
        start = time.time()
        optimizer.zero_grad()

        true_y = true_y.clone().detach().transpose(0, 1)

        start_y = true_y[0, :, :2].to(device)
        true_y, input = true_y[:, :, :2].to(device), true_y[:, :, 2].to(device)

        pred_y = odeint(lambda t, y: ode_model(t, y, t_, input), start_y, t_).to(device)
        loss = compute_loss(pred_y, true_y)

        loss.backward()
        optimizer.step()

        # Testing and visualizing
        if vis_testing and batch_idx % test_freq == 0:
            val_true_y, val_pred_y, val_loss = test_ODE(ode_model, t_, test_ds, test_itr)

            end = time.time() - start
            print('Iter {:04d} | Total Loss {:.6f} | Time {:.2f}'.format(test_itr, val_loss.item(), end))
            visualize_results(val_true_y, val_pred_y, t_, val_loss, loss, test_itr)

            test_itr += 1

