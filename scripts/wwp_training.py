import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils import load_config
from src.ww_model import DM
from src.coupled_columns import ColumnODEFunc



def make_ds_wwp(ds_file, nr_samples):
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

            if i % 10 == 0:
                # 0.0 input (10% of samples)
                muA, muB = 0.0, 0.0
                print(i)
            else:
                # Random input (90% of samples)
                muA, muB = np.random.uniform(0.0, 40.0, 2)

            R = dm.run_sim(muA, muB)
            R = R[:, ::10]  # only take every tenth time sample
            R = R[:, :time_steps]  # lose any extra time samples

            R_t = torch.tensor(R).transpose(0, 1)
            ds['states'][i, :, :] = R_t
            ds['stims'][i, :] = torch.tensor([muA, muB])


        with open(ds_file, 'wb') as f:
            pickle.dump(ds, f)
    return ds['states'], ds['stims']

def get_data(nr_samples, batch_size, fn):
    states, stims = make_ds_wwp(fn, nr_samples)

    states = states / 20.

    # Prepare train and test sets
    split = int(nr_samples * 0.9)
    train_states, test_states = states[:split, :, :], states[split:, :, :]
    train_stims, test_stims = stims[:split, :], stims[split:, :]

    train_dataset = TensorDataset(train_states, train_stims)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_states, test_stims

def get_stim(raw_stim):
    stim = torch.zeros(16)
    stim[2:4] += raw_stim[0]  # input column A
    stim[10:12] += raw_stim[1]  # input column B
    return stim

def huber_loss_firing_rates(pred, true, odefunc):
    # Compute firing rates for pred L2/3e
    mem_pred, adap_pred = pred[:, :, 0, [0, 8]], pred[:, :, 1, [0, 8]]
    fr_pred = odefunc.compute_firing_rate_torch(mem_pred - adap_pred)

    # Compute loss between ode prediction and WangWong simulated data
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)
    return hub_loss(fr_pred, true)

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

def test_ode(input_state, test_states, test_stims, iter, odefunc, time_vec, train_loss, weights, show=False):
    with torch.no_grad():
        stim = get_stim(test_stims[iter])
        true_state = test_states[iter, :, :]

        pred_state = odefunc.run_ode_stim_phases(input_state, stim, time_vec, num_stim_phases=3)
        test_loss = huber_loss_firing_rates(pred_state.unsqueeze(0), true_state.unsqueeze(0), odefunc)
    visualize_results(pred_state, true_state, stim, odefunc, train_loss, test_loss, weights, show)


def train_wwp_ode(nr_samples, batch_size, fn):
    '''
    Learn the lateral connections between two cortical column using
    data from Wang-Wong (WTA dynamics) and Potjans (resting state)
    '''

    # Get the train and test data
    train_loader, test_states, test_stims = get_data(nr_samples, batch_size, fn)

    # Initialize the ODE function
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')
    odefunc = ColumnODEFunc(col_params, 'MT')

    # Initial state is always the same
    membrane = torch.tensor(sim_params['initial_conditions']['membrane_potential'])
    adaptation = torch.tensor(sim_params['initial_conditions']['adaptation'])
    input_state = torch.stack((membrane, adaptation))

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

        pred_states = torch.Tensor(nr_batch_samples, time_steps, 2, 16)  # prediction for all 16 populations

        for batch_iter in range(nr_batch_samples):
            stim = get_stim(stim_batch[batch_iter])

            ode_output = odefunc.run_ode_stim_phases(input_state, stim, time_vec, 3)  # input_state defined earlier, always the same
            pred_states[batch_iter, :, :, :] = ode_output

        # for i in range(nr_batch_samples):
        #     plt.plot(pred_states[i, :, 0, [0, 8]].detach().numpy())
        #     plt.show()

        loss = huber_loss_firing_rates(pred_states, true_states, odefunc)
        loss.backward()
        with torch.no_grad():  # only update the lateral weights
            odefunc.connection_weights.grad *= odefunc.strict_mask
        optimizer.step()
        scheduler.step()  # adjust learning rate

        # Print loss, save current weights in array
        print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))
        weights.append(odefunc.connection_weights.detach().numpy().copy())
        # Test ODE model and visualize results
        test_ode(input_state, test_states, test_stims, iter, odefunc, time_vec, loss.item(), weights)


if __name__ == '__main__':

    time_steps = 1500
    nr_samples = 1000
    batch_size = 16

    fn = '../data/ds_wwp.pkl'

    train_wwp_ode(nr_samples, batch_size, fn)
