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
from neural_ode import ODEFunc, huber_loss


# Set params
ds_file             = 'pickled_ds/dmf_5000.pkl'
nr_samples          = 5000
T                   = 1000          # total time
dt                  = 1e-3          # timestep
batch_size          = 32
test_freq           = 3             # after n batches
vis_testing         = True
device              = torch.device('cpu')


# Run two columns (DMF)
def double(column):
    # Initialize the parameters
    params = get_params(J_local=0.13, J_lateral=0.172, area='MT')

    # Intialize the starting state (all zeros?)
    state = {}
    M = params['M']  # number of populations (=16)
    state['I'] = np.zeros(M)  # input current
    state['A'] = np.zeros(M)  # adaptation
    state['H'] = np.zeros(M)  # membrane potential
    state['R'] = np.zeros(M)  # rate
    state['N'] = np.zeros(M)  # noise

    # Initialize the stimulation
    stim = np.zeros(M)
    stim = set_vis(stim, column='H', nu=20.0, params=params)  # horizontal column
    stim = set_vis(stim, column='V', nu=20.0, params=params)  # vertical column

    stim = set_stimulation(stim, column='H', layer='L23', nu=20, params=params)
    stim = set_stimulation(stim, column='V', layer='L23', nu=20, params=params)

    stim = set_stimulation(stim, column='H', layer='L4', nu=20, params=params)
    stim = set_stimulation(stim, column='V', layer='L4', nu=20, params=params)

    stim = set_stimulation(stim, column='H', layer='L5', nu=20, params=params)
    stim = set_stimulation(stim, column='V', layer='L5', nu=20, params=params)

    stim = set_stimulation(stim, column='H', layer='L6', nu=20, params=params)
    stim = set_stimulation(stim, column='V', layer='L6', nu=20, params=params)

    # Total time steps
    T = 1000

    # Array for saving firing rate
    R = np.zeros((M, T))

    # Run simulation
    # note: stim does not change for the entirety of the simulation
    for t in range(T):
        state = update(state, params, stim)
        R[:, t] = state['R']

    # Plot the firing rate for each layer
    # column.plot_firing_rates(R[:8])
    # column.plot_firing_rates(R[8:])
    column.heatmap_over_time(R)


# Generate stimulus data
def make_stim(stim, T, dt):
    t = np.arange(0, T * dt, dt)

    # Sine wave params
    amplitude = 1
    frequency = 1
    shift = np.random.uniform(0, 2 * np.pi)

    # Sine wave
    sine = amplitude * np.sin(t * frequency * 2 * np.pi + shift)

    stim[2] = stim[2] * sine
    stim[3] = stim[3] * sine
    return stim, sine


def make_ds(save=True):
    if os.path.exists(ds_file):
        with open(ds_file, 'rb') as f:
            ds = pickle.load(f)
    else:
        ds = torch.empty((nr_samples, T, 3))

        for i in range(nr_samples + int(nr_samples/10)):  # add extra test samples
            # Initialize the column model
            column = SingleColumnDMF(area='MT')

            # Make a stimulus input
            stim = np.zeros((column.params['M'], T))
            stim = column.set_stim_ext(stim=stim, nu=20.0, params=column.params)
            stim, input = make_stim(stim, T, dt)

            # Simulate the stimulation
            firing_rates = column.simulate(stim=stim, T=T, state_var='R', vis=False)

            # Slice only layer 4 and add the original input (sine wave)
            layer_4 = firing_rates[2:4, :]
            layer_4_w_input = np.vstack([layer_4, input.reshape(1, T)])

            # Cast to tensor and add to ds
            layer_4_w_input_T = torch.tensor(layer_4_w_input).transpose(0, 1)
            ds[i, :, :] = layer_4_w_input_T

        # Save ds
        if save is True:
            with open(ds_file, 'wb') as f:
                pickle.dump(ds, f)
    return ds


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

        loss = huber_loss(pred_y, true_y)
    return true_y_input, pred_y, loss


def visualize_results(true_y, pred_y, t_, val_loss, train_loss, itr):
    if not os.path.exists('results/png'):
        os.makedirs('results/png')

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_input = fig.add_subplot(131, frameon=False)
    ax_exci = fig.add_subplot(132, frameon=False)
    ax_inhi = fig.add_subplot(133, frameon=False)

    fig.text(0.4, 0.03, f"Validation loss: {val_loss:.4f}", ha='center', fontsize=12, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.4f}", ha='center', fontsize=12, fontweight='bold')

    ax_input.cla()
    ax_input.set_title('Input')
    ax_input.plot(t_.detach().numpy(), true_y[:, 0, 2])
    ax_input.set_xlim(t_.cpu().min(), t_.cpu().max())
    ax_input.set_ylim(-3, 3)

    ax_exci.cla()
    ax_exci.set_title('Excitatory')
    ax_exci.plot(t_.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t_.cpu().numpy(),
                 pred_y.cpu().numpy()[:, 0, 0], '--')
    ax_exci.set_xlim(t_.cpu().min(), t_.cpu().max())
    ax_exci.set_ylim(-5, 5)

    ax_inhi.cla()
    ax_inhi.set_title('Inhibitory')
    ax_inhi.plot(t_.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], t_.cpu().numpy(),
                 pred_y.cpu().numpy()[:, 0, 1], '--')
    ax_inhi.set_xlim(t_.cpu().min(), t_.cpu().max())
    ax_inhi.set_ylim(-5, 5)

    fig.tight_layout()
    plt.savefig('results/png/{:03d}'.format(itr))
    plt.close(fig)


if __name__ == '__main__':

    t_ = torch.linspace(0., T*dt, T).to(device)
    t_ = t_[:500] #!!!!!

    ds = make_ds()

    # Normalize ds
    ds[:, :, :2] = (ds[:, :, :2] - torch.mean(ds[:, :, :2])) / torch.std(ds[:, :, :2])

    # Take half the time of ds
    ds = ds[:, :500, :]

    train_ds, test_ds = ds[:nr_samples], ds[nr_samples:]

    train_ds = TensorDataset(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

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
        loss = huber_loss(pred_y, true_y)

        loss.backward()
        optimizer.step()

        # Testing and visualizing
        if vis_testing and batch_idx % test_freq == 0:
            val_true_y, val_pred_y, val_loss = test_ODE(ode_model, t_, test_ds, test_itr)

            end = time.time() - start
            print('Iter {:04d} | Total Loss {:.6f} | Time {:.2f}'.format(test_itr, val_loss.item(), end))
            visualize_results(val_true_y, val_pred_y, t_, val_loss, loss, test_itr)

            test_itr += 1

