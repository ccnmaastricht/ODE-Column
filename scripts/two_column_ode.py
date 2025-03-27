import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.coupled_columns import CoupledColumns, ColumnODEFunc
from src.utils import load_config, huber_loss, mse_halfway_point


def make_ds_dmf(ds_file, nr_samples):
    if not os.path.exists('../data'):
        os.makedirs('../data')
    if os.path.exists(ds_file):
        with open(ds_file, 'rb') as f:
            ds = pickle.load(f)
    else:
        # Initialize two columns
        col_params = load_config('../config/model.toml')
        sim_params = load_config('../config/simulation.toml')
        columns = CoupledColumns(col_params, 'MT')

        # Make ds dict with two tensors states and stims
        protocol, dt = sim_params['protocol'], sim_params['time_step']
        time_steps = (protocol['pre_stimulus_period'] + protocol['post_stimulus_period'] + protocol['stimulus_duration']) / dt
        ds = {
            'states': torch.Tensor(nr_samples, int(time_steps), 3, columns.num_populations),
            'stims': torch.Tensor(nr_samples, columns.num_populations)
        }

        # Generate training data and store in ds dict
        for i in range(nr_samples):
            state, stim = columns.run_single_sim(sim_params, col_params, ff_input='random', num_stim_phases=3)
            ds['states'][i, :, :, :] = torch.tensor(state)   # membrane, adaptation and firing rate
            ds['stims'][i, :] = torch.tensor(stim)  # stimulus input used

        # Save ds dict as pickle
        with open(ds_file, 'wb') as f:
            pickle.dump(ds, f)
    return ds['states'], ds['stims']

def get_data(nr_samples, batch_size, fn):
    # Get dataset (load pickle if existing, else make new one)
    states, stims = make_ds_dmf(fn, nr_samples)

    # Prepare train and test sets
    split = int(nr_samples * 0.9)
    train_states, test_states = states[:split, :, :2, :], states[split:, :, :2, :]  # lose the firing rate
    train_stims, test_stims = stims[:split, :], stims[split:, :]

    train_dataset = TensorDataset(train_states, train_stims)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_states, test_stims

def visualize_results(pred, true, stim, odefunc, train_loss, test_loss, weights, show=False):
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    fig.text(0.2, 0.03, f"Input column 1: {stim[2]:.1f}", ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    fig.text(0.4, 0.03, f"Input column 2: {stim[10]:.1f}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Validation loss: {test_loss:.2f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')

    # Plot membrane potential
    axes[0, 0].plot(true[:, 0, 0], label='true col 1')
    axes[0, 0].plot(true[:, 0, 8], label='true col 2')
    axes[0, 0].plot(pred[:, 0, 0], '--', label='pred col 1')
    axes[0, 0].plot(pred[:, 0, 8], '--', label='pred col 2')
    axes[0, 0].set_title("Membrane potential in layer 2/3")
    fig.legend(loc="upper left")

    # Plot firing rate
    col1_true_fr = odefunc.compute_firing_rate_torch(true[:, 0, 0] - true[:, 1, 0])
    col2_true_fr = odefunc.compute_firing_rate_torch(true[:, 0, 8] - true[:, 1, 8])
    col1_pred_fr = odefunc.compute_firing_rate_torch(pred[:, 0, 0] - pred[:, 1, 0])
    col2_pred_fr = odefunc.compute_firing_rate_torch(pred[:, 0, 8] - pred[:, 1, 8])

    axes[1, 0].plot(col1_true_fr, label='true col 1')
    axes[1, 0].plot(col2_true_fr, label='true col 2')
    axes[1, 0].plot(col1_pred_fr, '--', label='pred col 1')
    axes[1, 0].plot(col2_pred_fr, '--', label='pred col 2')
    axes[1, 0].set_title("Firing rates in layer 2/3")

    # Plot current weights
    heatmap1 = axes[0, 1].imshow(weights[-1], cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[0, 1])
    axes[0, 1].set_title("Current weights")

    # Plot difference weights current and last timestep
    if len(weights) > 1:
        heatmap2 = axes[1, 1].imshow(weights[-2] - weights[-1], cmap="viridis", interpolation="nearest", vmin=-0.06, vmax=0.06)
        fig.colorbar(heatmap2, ax=axes[1, 1])
        axes[1, 1].set_title("Difference weights")

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig('../results/png/{:02d}'.format(len(weights)))
    if show:
        plt.show()
    plt.close(fig)

def test_ode(test_states, test_stims, iter, odefunc, time_vec, train_loss, weights, show=False):
    with torch.no_grad():
        stim = test_stims[iter]
        true_state = test_states[iter, :, :, :]

        pred_state = odefunc.run_ode_stim_phases(true_state[0], stim, time_vec, num_stim_phases=3)
        test_loss = huber_loss(pred_state.unsqueeze(0), true_state.unsqueeze(0))
    visualize_results(pred_state, true_state, stim, odefunc, train_loss, test_loss, weights, show)


def train_ode_two_columns(nr_samples, batch_size, fn):
    '''
    Train the coupled columns ODE to learn the lateral connections
    between the columns.
    '''
    # Get the train and test set
    train_loader, test_states, test_stims = get_data(nr_samples, batch_size, fn)

    # Initialize the ODE function
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')
    odefunc = ColumnODEFunc(col_params, 'MT')

    # Initialize the optimizer and add connection weights as learnable parameter
    optimizer = torch.optim.RMSprop([odefunc.connection_weights], lr=1e-2, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # higher gamma = slower decay

    # Store weights
    weights = []

    for iter, (true_states, stim_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        nr_batch_samples = true_states.shape[0]
        time_steps = true_states.shape[1]
        time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)

        pred_states = torch.Tensor(true_states.shape)

        for batch_iter in range(nr_batch_samples):
            stim = stim_batch[batch_iter]
            input_state = true_states[batch_iter, 0, :, :]

            ode_output = odefunc.run_ode_stim_phases(input_state, stim, time_vec, 3)
            pred_states[batch_iter, :, :, :] = ode_output

        loss = huber_loss(pred_states, true_states)  # only train on membrane potential
        loss.backward()
        with torch.no_grad():  # only update the lateral weights
            odefunc.connection_weights.grad *= odefunc.strict_mask
        optimizer.step()
        scheduler.step()  # adjust learning rate

        # Print loss, save current weights in array
        print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))
        weights.append(odefunc.connection_weights.detach().numpy().copy())
        # Test ODE model and visualize results
        test_ode(test_states, test_stims, iter, odefunc, time_vec, loss.item(), weights)


if __name__ == '__main__':
    nr_samples = 1000
    batch_size = 16

    train_ode_two_columns(nr_samples, batch_size, '../data/states_two_cols.pkl')
