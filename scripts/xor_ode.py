import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from src.coupled_columns import CoupledColumns, ColumnODEFunc
from src.utils import load_config, huber_loss, mse_halfway_point


def visualize_results(pred, true, stim, odefunc, train_loss, test_loss, weights, show=False):
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    fig.text(0.2, 0.03, f"Input column 1: {stim[2]:.1f}", ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    fig.text(0.4, 0.03, f"Input column 2: {stim[10]:.1f}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Validation loss: {test_loss:.2f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')

    # Plot membrane potential
    axes[0, 0].plot(true[:, 0, 0], label='col A1')
    axes[0, 0].plot(true[:, 0, 8], label='col B1')
    axes[0, 0].plot(pred[:, 0, 0], '--', label='col A2')
    axes[0, 0].plot(pred[:, 0, 8], '--', label='col B2')
    axes[0, 0].set_title("Membrane potential in layer 2/3")
    fig.legend(loc="upper left")

    # Plot firing rate
    col1_true_fr = odefunc.compute_firing_rate_torch(true[:, 0, 0] - true[:, 1, 0], odefunc.gain_function_parameters)
    col2_true_fr = odefunc.compute_firing_rate_torch(true[:, 0, 8] - true[:, 1, 8], odefunc.gain_function_parameters)
    col1_pred_fr = odefunc.compute_firing_rate_torch(pred[:, 0, 0] - pred[:, 1, 0], odefunc.gain_function_parameters)
    col2_pred_fr = odefunc.compute_firing_rate_torch(pred[:, 0, 8] - pred[:, 1, 8], odefunc.gain_function_parameters)

    axes[1, 0].plot(col1_true_fr, label='col A1')
    axes[1, 0].plot(col2_true_fr, label='col B1')
    axes[1, 0].plot(col1_pred_fr, '--', label='col A2')
    axes[1, 0].plot(col2_pred_fr, '--', label='col B2')
    axes[1, 0].set_title("Firing rates in layer 2/3")

    # Plot current weights
    heatmap1 = axes[0, 1].imshow(weights[-1], cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[0, 1])
    axes[0, 1].set_title("Current weights")

    # Plot difference weights current and last timestep
    if len(weights) > 1:
        heatmap2 = axes[1, 1].imshow(weights[-2] - weights[-1], cmap="viridis", interpolation="nearest") # , vmin=-0.06, vmax=0.06
        fig.colorbar(heatmap2, ax=axes[1, 1])
        axes[1, 1].set_title("Difference weights")

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig('../results/png/{:02d}'.format(len(weights)))
    if show:
        plt.show()
    plt.close(fig)

def test_ode(test_ff_input, test_stims, iter, odefunc, xor_classifier, time_vec, train_loss, weights, show=False):
    with torch.no_grad():
        fr_input = test_ff_input[iter, :, 2, :]  # firing rates
        prev_columns = test_ff_input[iter, :, :2, :]  # membrane and adaptation

        curr_columns = odefunc.run_ode_variable_stim(prev_columns[0], fr_input, time_vec)

        # compute firing rates
        firing_rates = odefunc.compute_firing_rate_torch(curr_columns[:, 0, :] - curr_columns[:, 1, :],
                                                         odefunc.gain_function_parameters)
        # pass through classifier and sigmoid function
        raw_xor_output = xor_classifier(firing_rates[-1, [0, 8]])
        xor_output = torch.sigmoid(raw_xor_output)
        # compute loss
        xor_targets = (test_stims[iter, 2] != test_stims[iter, 10]).int()
        test_loss = torch.mean(abs(xor_output - xor_targets.unsqueeze(0)))
        visualize_results(curr_columns, prev_columns, test_stims[iter], odefunc, train_loss, test_loss, weights, show)

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
        time_steps = (protocol['pre_stimulus_period'] + protocol['stimulus_duration']) / dt
        ds = {
            'states': torch.Tensor(nr_samples, int(time_steps), 3, columns.num_populations),
            'stims': torch.Tensor(nr_samples, columns.num_populations)
        }

        # Generate training data and store in ds dict
        for i in range(nr_samples):
            state, stim = columns.run_single_sim(sim_params, col_params, ff_input='xor', rand_membrane_init=False, num_stim_phases=2)
            ds['states'][i, :, :, :] = torch.tensor(state)   # membrane, adaptation and firing rate
            ds['stims'][i, :] = torch.tensor(stim)  # stimulus input used

        # Save ds dict as pickle
        with open(ds_file, 'wb') as f:
            pickle.dump(ds, f)
    return ds['states'], ds['stims']

def get_data(nr_samples, batch_size, fn):
    # Get dataset (load pickle if existing, else make new one)
    states, stims = make_ds_dmf('../pickled_ds/' + fn + '.pkl', nr_samples)

    # Prepare train and test sets
    split = int(nr_samples * 0.9)
    train_states, test_states = states[:split, :, :, :], states[split:, :, :, :]
    train_stims, test_stims = stims[:split, :], stims[split:, :]

    train_dataset = TensorDataset(train_states, train_stims)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_states, test_stims


def train_xor_ode(nr_samples, batch_size, fn, xor_classifier):
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
    optimizer = torch.optim.RMSprop([odefunc.connection_weights], lr=0.01, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # higher gamma = slower decay
    loss_func = nn.BCEWithLogitsLoss()

    # Store weights
    weights = []

    for iter, (ff_input, stim_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        nr_batch_samples = ff_input.shape[0]
        time_steps = ff_input.shape[1]
        time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)

        curr_columns = torch.Tensor(ff_input[:, :, :2, :].shape)

        for batch_iter in range(nr_batch_samples):
            fr_input = ff_input[batch_iter, :, 2, :]  # firing rates
            initial_state = ff_input[batch_iter, 0, :2, :]  # membrane and adaptation

            ode_output = odefunc.run_ode_variable_stim(initial_state, fr_input, time_vec)
            curr_columns[batch_iter, :, :, :] = ode_output

        # Train on xor classification
        # compute firing rates
        firing_rates = odefunc.compute_firing_rate_torch(curr_columns[:, :, 0, :] - curr_columns[:, :, 1, :], odefunc.gain_function_parameters)
        # pass through classifier and sigmoid function
        raw_xor_output = xor_classifier(firing_rates[:, -1, [0, 8]])
        xor_output = torch.sigmoid(raw_xor_output)
        # compute loss
        xor_targets = (stim_batch[:,2] != stim_batch[:,10]).int()
        # loss = torch.mean(abs(xor_output - xor_targets.unsqueeze(1)))
        loss = loss_func(xor_output, xor_targets.unsqueeze(1).float())

        loss.backward()
        with torch.no_grad():
            odefunc.connection_weights.grad *= odefunc.strict_mask
        optimizer.step()
        scheduler.step()  # adjust learning rate

        # Print loss, save current weights in array
        print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))
        weights.append(odefunc.connection_weights.detach().numpy().copy())
        # Test ODE model and visualize results
        test_ode(test_states, test_stims, iter, odefunc, xor_classifier, time_vec, loss.item(), weights)


def train_fr_classifier(batch_size, fn_ds):
    '''
    Trains a simple linear classifier that takes the firing rates of
    two columns as input, and outputs a value indicating which column
    is the winner and which is the loser.
    '''

    # Load the data
    with open('../pickled_ds/' + fn_ds + '.pkl', 'rb') as f:
        ds = pickle.load(f)
    states, stims = ds['states'], ds['stims']

    # Prepare data into input data and targets
    nr_samples = len(states)
    input_data = torch.Tensor(nr_samples, 2)
    targets = torch.Tensor(nr_samples)

    for i in range(nr_samples):
        state = states[i]

        # Get last firing rate of both columns, layer 2/3e
        last_fr_col_A = state[-1, 2, 0]
        last_fr_col_B = state[-1, 2, 8]
        input_data[i] = torch.tensor([last_fr_col_A, last_fr_col_B])

        # Identify winning and losing column
        if stims[i, 2] > stims[i, 10]:
            targets[i] = 1.0  # column A wins
        else:
            targets[i] = 0.0  # column B wins

    # Train and test set
    train_x, test_x = input_data[:900], input_data[900:]
    train_y, test_y = targets[:900], targets[900:]
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Init classifier and optimizer
    classifier = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1.0)

    # Training loop
    for batch_i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        raw_preds = classifier(inputs)
        preds = torch.sigmoid(raw_preds)

        loss = torch.mean(abs(preds - targets.unsqueeze(1)))
        loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     test_sample = test_x[batch_i]
        #     test_target = test_y[batch_i]
        #
        #     test_pred_raw = classifier(test_sample)
        #     test_pred = torch.sigmoid(test_pred_raw)
        #
        #     print('Classifier output:', test_pred.item())
        #     print('Target output:', test_target.item())
        #     print('Validation loss', torch.mean(abs(test_pred - test_target)).item())
        #     print()
    return classifier


if __name__ == '__main__':
    nr_samples = 1000
    batch_size = 16

    classifier = train_fr_classifier(20, 'states_dmf_17')
    classifier.eval()

    train_xor_ode(nr_samples, batch_size, 'xor_1000', classifier)
