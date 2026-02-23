import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

import torch
from torch.utils.data import TensorDataset, DataLoader

from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from src.utils import *
from src.ww_model import DM
from src.column_network_wta import ColumnAreaWTA


def visualize_results(pred, true, stim, network, train_loss, test_loss, weights):
    '''
    Visualize the firing rates of L23e during training.
    '''
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    fig.text(0.2, 0.03, f"Input column 1: {stim[0]:.1f}", ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    fig.text(0.4, 0.03, f"Input column 2: {stim[1]:.1f}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
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

    axes[0].plot(col1_true_fr.cpu().numpy(), '--', label='true col 1')
    axes[0].plot(col2_true_fr.cpu().numpy(), '--', label='true col 2')
    axes[0].plot(col1_pred_fr.cpu().numpy(), label='pred col 1')
    axes[0].plot(col2_pred_fr.cpu().numpy(), label='pred col 2')
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
            muA = np.random.uniform(15.0, 20.0)
            muB = muA + np.random.uniform(10., 15.)
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
    scales it down to match our L23 firing rates.
    '''
    states, stims = make_ds_wwp(fn, nr_samples+10, time_steps)
    states, stims = states[:nr_samples+10], stims[:nr_samples+10]

    states = states / 30.  # scale down wang-wong firing rates to match with our L23

    ds = TensorDataset(states, stims)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return data_loader

def set_stim_three_phases(num_populations, time_vec, raw_stim):
    '''
    Extent the given input stimulus to fit the time vector
    in three phases: pre-, stimulus, and post-.
    '''
    stim = torch.zeros(16)
    stim[:8] = torch.tile(raw_stim[0], (8,))
    stim[8:] = torch.tile(raw_stim[1], (8,))

    stim_vector = torch.zeros((len(time_vec), num_populations))
    stim_onset = int(len(time_vec) / 3)
    stim_offset = int(stim_onset + len(time_vec) / 3)
    stim_vector[stim_onset:stim_offset, :] = stim
    return stim_vector

def init_network(device):
    '''
    Initialize the two-column network, initial state and time vector.
    '''
    # Time steps for three stimulus phases (pre- and post-stimulus phase)
    dt = 1e-4
    stim_phase = 0.05
    time_steps = int((stim_phase * 3) / dt)  # add pre- and post-stimulus phase

    # Column network setup
    col_params = load_config('../config/model.toml')
    network = ColumnAreaWTA(col_params, area='mt')

    # Initial state
    initial_state = torch.zeros(48).unsqueeze(0)
    initial_state[:, :16] = torch.tile(torch.tensor([-1.7997e-01, 8.3757e+00, 1.1346e+01, 1.1953e+01,
                                                     -6.5426e+00, 1.0319e+01, -2.9719e+01, 1.2530e+01]), (1, 2,))

    # Time vector
    time_vec = torch.linspace(0., time_steps * dt, time_steps)
    return network.to(device), initial_state.to(device), time_vec.to(device)

def run_sample(network, time_vec, initial_state, stim_raw, with_noise):
    '''
    Runs one stimulus time course through the network
    '''
    stim = set_stim_three_phases(network.num_populations, time_vec, stim_raw)
    network.set_stim(stim)

    if with_noise:
        ode_output = sdeint(network,
                            initial_state,
                            time_vec,
                            names={'drift': 'forward', 'diffusion': 'diffusion'},
                            method='srk').to(device)
    else:
        ode_output = odeint(network,
                            initial_state,
                            time_vec).to(device)

    return ode_output

# new stuff
def get_rand_conn_matrix(network):
    rand_recurr_synapse_counts = np.array([[6.44611506e+03, 2.97641736e+03, 2.71606571e+03, 6.18350231e+02,
        5.11499648e+02, 9.73190684e-13, 1.26077066e+02, 8.11417735e-13],
       [8.82406026e+03, 2.52076805e+03, 8.99901238e+02, 3.66855765e+02,
        1.05762360e+03, 1.01762814e-12, 7.07995698e+01, 8.48468378e-13],
       [5.55731312e+02, 1.11906834e+02, 1.43766957e+03, 9.59527651e+02,
        1.01605231e+02, 1.22384794e+00, 6.92476378e+02, 8.78980871e-13],
       [4.70560130e+03, 4.35943444e+01, 2.09188067e+03, 1.22667421e+03,
        4.10005724e+01, 8.58419423e-13, 1.65638350e+03, 7.15724837e-13],
       [6.28890331e+03, 1.20702163e+03, 1.48410838e+03, 4.49314429e+01,
        1.22986559e+03, 1.46099534e+03, 3.03601330e+02, 7.87490395e-13],
       [3.14419829e+03, 5.40355095e+02, 8.32140386e+02, 1.84464212e+01,
        9.48459692e+02, 1.18139903e+03, 1.61956446e+02, 8.55011479e-13],
       [9.91090311e+02, 1.12801194e+02, 6.08519556e+02, 1.62759744e+02,
        7.54152810e+02, 5.31472514e+01, 6.39900796e+02, 8.27452569e+02],
       [2.08811279e+03, 1.56491584e+01, 9.70961186e+01, 3.33127755e+00,
        4.16450246e+02, 2.36999731e+01, 1.22071430e+03, 5.05374243e+02]])

    # Extent 8x8 connections to 16x16, i.e. two columns
    blocks = [rand_recurr_synapse_counts] * 2  # *2 columns
    rand_recurr_synapse_counts = block_diag(*blocks)

    # Multiply with synapse strength
    rand_recurr_synapse_counts = torch.tensor(rand_recurr_synapse_counts, dtype=torch.float32)
    rand_recurr_weights = rand_recurr_synapse_counts * network.recurrent_synaptic_strength
    return rand_recurr_weights

def compute_pd_deviation_penalty(network, pd_original_connectivity):
    '''
    Computes the mean absolute error between the original Potjans and
    Diesmann connectivity profile and the current connectivity profile.
    '''
    curr_internal_connections = network.recurrent_weights
    abs_diff = abs(curr_internal_connections - pd_original_connectivity)
    deviation_penalty = torch.mean(abs_diff)

    return deviation_penalty

def train_wta(nr_samples,
              batch_size,
              fn,
              device,
              with_noise=True,
              adjust_pd=False,
              scrambled_pd=False):
    '''
    Learn the lateral connections between two cortical columns using
    data from Wang-Wong (WTA dynamics) as a training target.
    '''
    # Initialize network, initial state and time vector
    network, initial_state, time_vec = init_network(device)
    time_steps = len(time_vec)
    network.set_time_vec(time_vec)

    # Get the train and test data
    data_loader = get_data(nr_samples, batch_size, time_steps, fn)

    if scrambled_pd:
        # Re-set the recurrent connections to scrambled version
        rand_recurr_weights = get_rand_conn_matrix(network)
        network.recurrent_weights = rand_recurr_weights

    if adjust_pd:
        # Set column-intrinsic connectivity as learnable parameter
        network.initialize_recurrent_weights()
        optimizer = torch.optim.RMSprop([{'params': network.lat_in_weights, 'lr': 10.0},
                                         {'params': network.recurrent_weights, 'lr': 1.0}], alpha=0.9)
    else:
        optimizer = torch.optim.RMSprop([network.lat_in_weights], lr=10.0, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # higher gamma = slower decay

    # Store weights for visualization
    pd_original_connectivity = network.recurrent_weights.clone().detach()
    weights = []

    for iter, (true_states, stim_batch) in enumerate(data_loader):
        optimizer.zero_grad()
        network.constrain_recurr_matrix()

        nr_batch_samples = true_states.shape[0] - 1  # use last sample for testing
        pred_states = torch.Tensor(nr_batch_samples, time_steps, 1, network.num_populations * 3).to(device)  # *3 bc mem, adap and fr
        true_states = true_states.to(device)

        # Run each training sample in the batch
        for batch_iter in range(nr_batch_samples):
            ode_output = run_sample(network, time_vec, initial_state, stim_batch[batch_iter], with_noise)
            pred_states[batch_iter, :, :, :] = ode_output

        # Compute loss between pred and true
        hub_loss = huber_loss_wta(pred_states, true_states[:-1], network)
        loss = hub_loss
        print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))

        if adjust_pd:
            penalty = compute_pd_deviation_penalty(network, pd_original_connectivity)
            # loss += (penalty * 0.1)  # penalty weight!
            print(hub_loss.item())
            print(penalty.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validate network and visualize results
        with torch.no_grad():
            # Save current weights
            network.constrain_recurr_matrix()
            curr_weights = network.W.detach().cpu().numpy()
            weights.append(curr_weights - pd_original_connectivity.cpu().numpy())

            # Run test sample
            pred_state = run_sample(network, time_vec, initial_state, stim_batch[-1], with_noise)

            # Visualize final test sample
            test_state = true_states[-1, :, :]
            test_loss = huber_loss_wta(pred_state.unsqueeze(0), test_state.unsqueeze(0), network)
            visualize_results(pred_state, test_state, stim_batch[-1], network, loss.item(), test_loss, weights)

    return network




if __name__ == '__main__':

    set_seed(1)
    device = torch.device('mps')
    ds_target = '../data/ds_wta_6000_15_20_10_15.pkl'

    network = train_wta(nr_samples=3000,
                        batch_size=16,
                        fn=ds_target,
                        device=device,
                        with_noise=True,
                        adjust_pd=False,
                        scrambled_pd=False
                        )

    with open('../wta_trained_model.pkl', 'wb') as f:
        pickle.dump(network, f)

