import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.coupled_columns import CoupledColumns, ColumnODEFunc
from src.utils import load_config

from ode_bifurcation import huber_loss



def visualize_results(pred, true, stim, odefunc, train_loss, test_loss, weights, show=False):
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    fig.text(0.2, 0.03, f"Input column 1: {stim[2]:.1f}", ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    fig.text(0.4, 0.03, f"Input column 2: {stim[10]:.1f}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Validation loss: {test_loss:.4f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.4f}", ha='center', fontsize=10, fontweight='bold')

    # Plot membrane potential
    axes[0, 0].plot(true[:, 0, 0], label='true col 1')
    axes[0, 0].plot(true[:, 0, 8], label='true col 2')
    axes[0, 0].plot(pred[:, 0, 0], '--', label='pred col 1')
    axes[0, 0].plot(pred[:, 0, 8], '--', label='pred col 2')
    axes[0, 0].set_title("Membrane potential in layer 2/3")
    fig.legend(loc="upper left")

    # Plot firing rate
    col1_true_fr = odefunc.compute_firing_rate_torch(true[:, 0, 0] - true[:, 1, 0], odefunc.gain_function_parameters)
    col2_true_fr = odefunc.compute_firing_rate_torch(true[:, 0, 8] - true[:, 1, 8], odefunc.gain_function_parameters)
    col1_pred_fr = odefunc.compute_firing_rate_torch(pred[:, 0, 0] - pred[:, 1, 0], odefunc.gain_function_parameters)
    col2_pred_fr = odefunc.compute_firing_rate_torch(pred[:, 0, 8] - pred[:, 1, 8], odefunc.gain_function_parameters)

    axes[1, 0].plot(col1_true_fr, label='true col 1')
    axes[1, 0].plot(col2_true_fr, label='true col 2')
    axes[1, 0].plot(col1_pred_fr, '--', label='pred col 1')
    axes[1, 0].plot(col2_pred_fr, '--', label='pred col 2')
    axes[1, 0].set_title("Firing rates in layer 2/3")

    # Plot current weights
    axes[0, 1].imshow(weights[-1], cmap="viridis", interpolation="nearest")
    axes[0, 1].set_title("Current weights")

    # Plot difference weights current and first
    axes[1, 1].imshow(weights[0] - weights[-1], cmap="viridis", interpolation="nearest")
    axes[1, 1].set_title("Difference weights")

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig('../results/png/{:02d}'.format(len(weights)))
    if show:
        plt.show()

def test_ode(test_states, test_stims, iter, odefunc, time_vec, train_loss, weights, show=False):
    with torch.no_grad():
        stim = test_stims[iter]
        true_state = test_states[iter, :, :, :]

        pred_state = odefunc.run_ode_sample(true_state[0], stim, time_vec)
        test_loss = huber_loss(pred_state[:, 0, :], true_state[:, 0, :])  # 0 = membrane
    visualize_results(pred_state, true_state, stim, odefunc, train_loss, test_loss, weights, show)

def make_ds_dmf(ds_file, nr_samples):
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
            state, stim = columns.run_single_sim(sim_params, col_params, rand_input=True)
            ds['states'][i, :, :, :] = torch.tensor(state)   # membrane, adaptation and firing rate
            ds['stims'][i, :] = torch.tensor(stim)  # stimulus input used

        # Save ds dict as pickle
        with open(ds_file, 'wb') as f:
            pickle.dump(ds, f)
    return ds['states'], ds['stims']


if __name__ == '__main__':

    nr_samples = 1000
    batch_size = 16

    # Get dataset
    states, stims = make_ds_dmf('../pickled_ds/new_dmf_states.pkl', nr_samples)

    # Prepare train and test sets
    split = int(nr_samples * 0.5)
    train_states, test_states = states[:split, :, :2, :], states[split:, :, :2, :]  # lose the firing rate
    train_stims, test_stims = stims[:split, :], stims[split:, :]

    train_dataset = TensorDataset(train_states, train_stims)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the ODE function
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')
    odefunc = ColumnODEFunc(col_params, 'MT')

    # Initialize the optimizer and add connection weights as learnable parameter
    optimizer = torch.optim.RMSprop([odefunc.connection_weights], lr=1e-2, alpha=0.99)

    # Store weights
    weights = [odefunc.connection_weights.detach().numpy().copy()]

    for iter, (true_states, stim_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        nr_batch_samples = len(true_states)
        time_steps = int(len(true_states[0]) / 3)  # divide by 3, we have three stimulus phases
        time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)

        pred_states = torch.Tensor(true_states.shape)

        for batch_iter in range(nr_batch_samples):
            stim = stim_batch[batch_iter]
            input_state = true_states[batch_iter, 0, :, :]

            ode_output = odefunc.run_ode_sample(input_state, stim, time_vec)
            pred_states[batch_iter, :, :, :] = ode_output

        loss = huber_loss(pred_states[:, :, 0, :], true_states[:, :, 0, :])  # 0=membrane
        # Compute gradients and update
        loss.backward()
        with torch.no_grad():
            odefunc.connection_weights.grad *= odefunc.strict_mask
        optimizer.step()

        # Print loss, save current weights in array
        print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))
        weights.append(odefunc.connection_weights.detach().numpy().copy())
        # Test ODE model and visualize results
        test_ode(test_states, test_stims, iter, odefunc, time_vec, loss.item(), weights)
