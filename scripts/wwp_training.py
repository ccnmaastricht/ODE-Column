import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils import load_config, huber_loss_firing_rates, create_feedforward_input
from src.ww_model import DM
from src.coupled_columns import ColumnODEFunc



def visualize_all_layers(pred, true, stim, odefunc, train_loss, test_loss, weights, show=False):
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    fig.text(0.2, 0.03, f"Input column 1: {stim[2]:.1f}", ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    fig.text(0.4, 0.03, f"Input column 2: {stim[10]:.1f}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Validation loss: {test_loss:.2f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')

    # Plot firing rates
    fr_rates = odefunc.compute_firing_rate_torch(pred[:, 0, :] - pred[:, 1, :])

    for i in range(2):
        for j in range(2):
            idx = int((i * 4) + (j * 2))

            axes[i, j].plot(fr_rates[:,idx], label='column 1, excitatory')
            axes[i, j].plot(fr_rates[:,idx + 8], label='column 2, excitatory')
            # axes[i, j].plot(fr_rates[:,idx + 1], '--', label='column 1, inhibitory')
            # axes[i, j].plot(fr_rates[:,idx + 9], '--', label='column 1, inhibitory')

            titles = ['L23', 'L4', 'L5', 'L6']
            axes[i, j].set_title(titles[int(idx / 2)])

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig('../results/png/{:02d}'.format(len(weights)))
    if show:
        plt.show()
    plt.close(fig)

def visualize_results(pred, true, stim, odefunc, train_loss, test_loss, weights, show=False):
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    fig.text(0.2, 0.03, f"Input column 1: {stim[2]:.1f}", ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    fig.text(0.4, 0.03, f"Input column 2: {stim[10]:.1f}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Validation loss: {test_loss:.2f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')

    # Plot firing rate
    col1_pred_fr = odefunc.compute_firing_rate_torch(pred[:, 0, 0] - pred[:, 1, 0])
    col2_pred_fr = odefunc.compute_firing_rate_torch(pred[:, 0, 8] - pred[:, 1, 8])
    col1_true_fr = true[:, 0]
    col2_true_fr = true[:, 1]

    axes[0].plot(col1_true_fr, label='true col 1')
    axes[0].plot(col2_true_fr, label='true col 2')
    axes[0].plot(col1_pred_fr, '--', label='pred col 1')
    axes[0].plot(col2_pred_fr, '--', label='pred col 2')
    axes[0].set_title("Firing rates in layer 2/3")

    # Plot current weights
    heatmap1 = axes[1].imshow(weights[-1], cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[1])
    axes[1].set_title("Current weights")

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig('../results/png/{:02d}'.format(len(weights)))
    if show:
        plt.show()
    plt.close(fig)

def make_ds_wwp(ds_file, nr_samples, time_steps):
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

            # Random input between 15 and 40Hz, which random (but small) difference between them
            muA = np.random.uniform(15.0, 40.0)
            diff_mag = muA / np.random.uniform(7.5, 15.0)
            muB = muA + np.random.choice([diff_mag, diff_mag*-1.0])

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
    states, stims = make_ds_wwp(fn, nr_samples+10, time_steps)

    states = states / 20.  # scale down wang-wong firing rates to match with L23

    ds = TensorDataset(states, stims)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return data_loader

def get_stim(raw_stim):
    stim = create_feedforward_input(16, raw_stim[0], raw_stim[1])
    return torch.tensor(stim)


def train_wwp_ode(nr_samples, batch_size, time_steps, fn, lambda_potjans):
    '''
    Learn the lateral connections between two cortical column using
    data from Wang-Wong (WTA dynamics) and Potjans (resting state)
    '''

    # Get the train and test data
    data_loader = get_data(nr_samples, batch_size, time_steps, fn)

    # Initialize the ODE function
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')
    odefunc = ColumnODEFunc(col_params, 'MT', learn_wta=True)

    # Initial state is always the same
    membrane = torch.tensor(sim_params['initial_conditions']['membrane_potential'])
    adaptation = torch.tensor(sim_params['initial_conditions']['adaptation'])
    input_state = torch.stack((membrane, adaptation))

    # Mean firing rates in resting state, according to Potjans
    potjans_mean_fr = torch.tensor([
        0.85, 2.9, 4.45, 5.8, 7.5, 8.6, 1.1, 7.75,
        0.85, 2.9, 4.45, 5.8, 7.5, 8.6, 1.1, 7.75
    ])

    # Initialize the optimizer and add connection weights as learnable parameter
    optimizer = torch.optim.RMSprop([odefunc.connection_weights], lr=1e-2, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # higher gamma = slower decay

    # Store weights
    weights = []
    # Store loss
    losses = []

    for iter, (true_states, stim_batch) in enumerate(data_loader):
        optimizer.zero_grad()

        nr_batch_samples = true_states.shape[0] - 4  # use last four samples for testing
        time_steps = true_states.shape[1]
        time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)


        ### Wang Wong WTA training
        pred_states = torch.Tensor(nr_batch_samples, time_steps, 2, 16)  # prediction for all 16 populations

        for batch_iter in range(nr_batch_samples):
            stim = get_stim(stim_batch[batch_iter])

            ode_output = odefunc.run_ode_stim_phases(input_state, stim, time_vec, 3)  # input_state defined earlier, always the same
            pred_states[batch_iter, :, :, :] = ode_output
        wang_wong_loss = huber_loss_firing_rates(pred_states, true_states[:-4], odefunc)


        # ### Potjans resting state training
        # stim = get_stim(torch.tensor([0., 0.]))  # input = 0Hz
        # ode_output = odefunc.run_ode_stim_phases(input_state, stim, time_vec, 3)
        #
        # # Compute firing rates
        # mem, adap = ode_output[:, 0, :], ode_output[:, 1, :]
        # resting_state_fr = odefunc.compute_firing_rate_torch(mem - adap)
        # mean_resting_state_fr = torch.mean(resting_state_fr, dim=0)
        # potjans_loss = abs(torch.mean(mean_resting_state_fr - potjans_mean_fr))


        loss = wang_wong_loss # + (potjans_loss * lambda_potjans)
        loss.backward()
        with torch.no_grad():  # only update lateral and self-excitation weights
            odefunc.connection_weights.grad *= odefunc.wta_mask
        optimizer.step()
        scheduler.step()  # adjust learning rate

        # Print loss, save current weights in array
        print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))
        weights.append(odefunc.connection_weights.detach().numpy().copy())

        # Test ODE model and visualize results
        with torch.no_grad():
            batch_loss = 0
            nr_test_samples = 1

            for i in range(nr_test_samples):
                test_iter = nr_batch_samples + i

                stim = get_stim(stim_batch[test_iter])
                test_state = true_states[test_iter, :, :]

                pred_state = odefunc.run_ode_stim_phases(input_state, stim, time_vec, num_stim_phases=3)

                test_loss = huber_loss_firing_rates(pred_state.unsqueeze(0), test_state.unsqueeze(0), odefunc)
                batch_loss += test_loss / nr_test_samples

            # Add mean loss to list
            losses.append(batch_loss)
            # Visualize final test sample
            visualize_results(pred_state, test_state, stim, odefunc, loss.item(), test_loss, weights, show=False)
    return odefunc


if __name__ == '__main__':

    time_steps = 1500
    nr_samples = 2000
    batch_size = 20

    lambda_potjans = 0.1

    fn = '../data/ds_wwp.pkl'

    model = train_wwp_ode(nr_samples, batch_size, time_steps, fn, lambda_potjans)

    with open('../ww_trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

