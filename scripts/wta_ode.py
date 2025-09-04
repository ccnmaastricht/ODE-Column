import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from src.utils import *
from src.ww_model import DM
from src.coupled_columns import ColumnAreaWTA


def visualize_results(pred, true, stim, network, train_loss, test_loss, weights):
    '''
    Visualize the firing rates of L23e during training.
    '''
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    fig.text(0.2, 0.03, f"Input column 1: {stim[2]:.1f}", ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    fig.text(0.4, 0.03, f"Input column 2: {stim[10]:.1f}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Validation loss: {test_loss:.2f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')

    # Plot firing rate
    firing_rates = compute_firing_rate(pred[:, 0, :16] - pred[:, 0, 16:32])
    col1_pred_fr_all = firing_rates[:, :8]
    col2_pred_fr_all = firing_rates[:, 8:]
    col1_pred_fr = torch.sum(col1_pred_fr_all * network.output_weights, dim=-1)
    col2_pred_fr = torch.sum(col2_pred_fr_all * network.output_weights, dim=-1)

    col1_true_fr = true[:, 0]
    col2_true_fr = true[:, 1]

    axes[0].plot(col1_true_fr, '--', label='true col 1')
    axes[0].plot(col2_true_fr, '--', label='true col 2')
    axes[0].plot(col1_pred_fr, label='pred col 1')
    axes[0].plot(col2_pred_fr, label='pred col 2')
    axes[0].set_title("Firing rates in layer 2/3")

    # Plot current weights
    heatmap1 = axes[1].imshow(weights[-1], cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[1])
    axes[1].set_title("Current weights")

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig('../results/png/{:02d}'.format(len(weights)))
    plt.close(fig)

def make_ds_wwp(ds_file, nr_samples, time_steps):
    '''
    Make a dataset of Wang-Wong training samples. If filename
    already exists, it will load the existing dataset.
    '''
    if not os.path.exists('../data'):
        os.makedirs('../data')
    if os.path.exists(ds_file):
        with open(ds_file, 'rb') as f:
            ds = pickle.load(f)
    else:

        ds = {
            'states': torch.Tensor(nr_samples, time_steps, 2),
            'stims': torch.Tensor(nr_samples, 2)
        }

        dm = DM()  # Wang Wong model

        for i in range(nr_samples):

            # Random input
            muA = np.random.uniform(15.0, 25.0)
            muB = muA + np.random.uniform(10., 20.)
            mu_vals = [muA, muB]
            np.random.shuffle(mu_vals)
            muA, muB = mu_vals

            R = dm.run_sim(muA, muB)
            R = R[:, ::10]  # only take every tenth time sample
            R = R[:, :time_steps]  # lose any extra time samples

            R_t = torch.tensor(R).transpose(0, 1)
            ds['states'][i, :, :] = R_t
            ds['stims'][i, :] = torch.tensor([muA, muB])

        with open(ds_file, 'wb') as f:
            pickle.dump(ds, f)
    return ds['states'], ds['stims']

def get_data(nr_samples, batch_size, time_steps, fn):
    '''
    Gets the training dataset made with Wang-Wong model,
    scales it down by 20 to match our L23 firing rates.
    '''
    states, stims = make_ds_wwp(fn, nr_samples+10, time_steps)

    states = states / 20.  # scale down wang-wong firing rates to match with L23

    ds = TensorDataset(states, stims)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return data_loader

def set_stim_three_phases(num_populations, time_vec, raw_stim):
    '''
    Extent the given input stimulus to fit the time vector
    in three phases: pre-, stimulus, and post-.
    '''
    stim = torch.zeros(16)
    stim[2], stim[3]    = raw_stim[0], raw_stim[0]
    stim[10], stim[11]  = raw_stim[1], raw_stim[1]

    stim_vector = torch.zeros((len(time_vec), num_populations))
    stim_onset = int(len(time_vec) / 3)
    stim_offset = int(stim_onset + len(time_vec) / 3)
    stim_vector[stim_onset:stim_offset, :] = stim
    return stim_vector

def init_network(network_class, time_steps, dt):
    '''
    Initialize the two-column network, initial state and time vector.
    '''
    # Column network setup
    col_params = load_config('../config/model.toml')
    network = network_class(col_params, area='mt')

    # Initial state
    initial_state = torch.zeros(48).unsqueeze(0)

    # Time vector
    time_vec = torch.linspace(0., time_steps * dt, time_steps)
    return network, initial_state, time_vec

def train_wta_ode(nr_samples, batch_size, fn):
    '''
    Learn the lateral connections between two cortical columns using
    data from Wang-Wong (WTA dynamics) as a training target.
    '''
    # Time steps for three stimulus phases (pre- and post-stimulus phase)
    dt = 1e-4
    stim_phase = 0.05
    time_steps = int((stim_phase * 3) / dt)  # add pre- and post-stimulus phase

    # Get the train and test data
    data_loader = get_data(nr_samples, batch_size, time_steps, fn)

    # Initialize network, initial state and time vector
    network, initial_state, time_vec = init_network(ColumnAreaWTA, time_steps, dt)
    network.time_vec = time_vec

    # Initialize the optimizer and add connection weights as learnable parameter
    optimizer = torch.optim.RMSprop([network.recurrent_weights], lr=10.0, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # higher gamma = slower decay

    # Store weights for visualization
    weights = []

    for iter, (true_states, stim_batch) in enumerate(data_loader):
        optimizer.zero_grad()

        nr_batch_samples = true_states.shape[0] - 1  # use last sample for testing
        pred_states = torch.Tensor(nr_batch_samples, time_steps, 1, network.num_populations * 3)  # *3 bc mem, adap and fr

        for batch_iter in range(nr_batch_samples):
            stim = set_stim_three_phases(network.num_populations, time_vec, stim_batch[batch_iter])
            network.stim = stim

            # ode_output = odeint(network, initial_state, time_vec)
            ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

            pred_states[batch_iter, :, :, :] = ode_output

        # Compute firing rate and loss between pred and true
        loss = huber_loss_wta(pred_states, true_states[:-1], network)
        loss.backward()

        with torch.no_grad():  # only update lateral and self-excitation weights
            network.recurrent_weights.grad *= network.lat_in_mask

        optimizer.step()
        scheduler.step()  # adjust learning rate

        # Print loss, save current weights in array
        print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))
        weights.append(network.recurrent_weights.detach().numpy().copy())

        # Test ODE model and visualize results
        with torch.no_grad():

            stim = set_stim_three_phases(network.num_populations, time_vec, stim_batch[-1])
            network.stim = stim
            test_state = true_states[-1, :, :]

            # pred_state = odeint(network, initial_state, time_vec)
            pred_state = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

            # Visualize final test sample
            test_loss = huber_loss_wta(pred_state.unsqueeze(0), test_state.unsqueeze(0), network)
            visualize_results(pred_state, test_state, stim[500], network, loss.item(), test_loss, weights)
    return network


if __name__ == '__main__':

    nr_samples = 3000
    batch_size = 16

    network = train_wta_ode(nr_samples, batch_size, '../data/ds_wta_3000_noise.pkl')

    with open('../ww_trained_model.pkl', 'wb') as f:
        pickle.dump(network, f)

