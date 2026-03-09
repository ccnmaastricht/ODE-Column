import matplotlib.pyplot as plt
import os
import itertools
import random
import pickle
import torch
from pprint import pprint
import numpy as np

from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from src.column_network_parity import ColumnNetwork
from src.utils import *



def visualize_results(network, firing_rates, stims, loss, train_iter, batch_size, target_trajectory, two_output_cols):
    '''
    Visualize the firing rates of the last few columns while training.
    '''
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')

    nr_samples = batch_size  # how many samples to visualize

    firing_rates_cpu = firing_rates.cpu().numpy()

    for i in range(nr_samples):
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))

        idx_col1 = 64  # first 8 columns
        axes[0, 0].plot(firing_rates_cpu[:, i, idx_col1 + 0], label='L23e')
        axes[0, 0].plot(firing_rates_cpu[:, i, idx_col1 + 4] * 0.1, label='L5e')
        axes[0, 0].plot(firing_rates_cpu[:, i, idx_col1 + 6], label='L6e')
        fig.legend(loc="upper left")

        idx_col1 = 64 + 8
        axes[0, 1].plot(firing_rates_cpu[:, i, idx_col1 + 0], label='L23e')
        axes[0, 1].plot(firing_rates_cpu[:, i, idx_col1 + 4] * 0.1, label='L5e')
        axes[0, 1].plot(firing_rates_cpu[:, i, idx_col1 + 6], label='L6e')

        idx_col1 = 64 + 16
        axes[0, 2].plot(firing_rates_cpu[:, i, idx_col1 + 0], label='L23e')
        axes[0, 2].plot(firing_rates_cpu[:, i, idx_col1 + 4] * 0.1, label='L5e')
        axes[0, 2].plot(firing_rates_cpu[:, i, idx_col1 + 6], label='L6e')

        idx_col1 = 64 + 32
        axes[0, 3].plot(firing_rates_cpu[:, i, idx_col1 + 0], label='L23e')
        axes[0, 3].plot(firing_rates_cpu[:, i, idx_col1 + 4] * 0.1, label='L5e')
        axes[0, 3].plot(firing_rates_cpu[:, i, idx_col1 + 6], label='L6e')
        fig.legend(loc="upper left")

        idx_col1 = 64 + 40
        axes[1, 0].plot(firing_rates_cpu[:, i, idx_col1 + 0], label='L23e')
        axes[1, 0].plot(firing_rates_cpu[:, i, idx_col1 + 4] * 0.1, label='L5e')
        axes[1, 0].plot(firing_rates_cpu[:, i, idx_col1 + 6], label='L6e')

        idx_col1 = 64 + 48
        axes[1, 1].plot(firing_rates_cpu[:, i, idx_col1 + 0], label='L23e')
        axes[1, 1].plot(firing_rates_cpu[:, i, idx_col1 + 4] * 0.1, label='L5e')
        axes[1, 1].plot(firing_rates_cpu[:, i, idx_col1 + 6], label='L6e')

        # Final column
        if two_output_cols:
            final_column = torch.sum((firing_rates[:, i, -16:-8] * network.output_weights[-16:-8]), dim=-1)
            final_column_2 = torch.sum((firing_rates[:, i, -8:] * network.output_weights[-8:]), dim=-1)
            axes[0, 3].plot(final_column.cpu().numpy())
            axes[0, 3].plot(target_trajectory, linestyle='--')
            axes[0, 3].set_title('Final column')
            axes[1, 3].plot(final_column_2.cpu().numpy())
            axes[1, 3].plot(target_trajectory, linestyle='--')
            axes[1, 3].set_title('Final column')
        else:
            final_column = torch.sum((firing_rates[:, i, -8:] * network.output_weights), dim=-1)
            axes[1, 3].plot(final_column.cpu().numpy())
            axes[1, 3].plot(target_trajectory, linestyle='--')
            axes[1, 3].set_title('Final column')

        fig.text(0.2, 0.03, f"Training loss: {loss:.2f}", ha='center', fontsize=10, fontweight='bold')
        fig.text(0.5, 0.03, f"Input: {stims[i]}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
        fig.text(0.8, 0.03, f"Final FR: {final_column[-1]:.2f}", ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout(pad=3.0)
        fig.subplots_adjust(left=0.15)
        plt.savefig('../results/png/firing_rates_{:02d}_{:1d}'.format(train_iter + 1, i))
        plt.close(fig)

        # Also plot the first eight columns
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))

        fig.text(0.2, 0.03, f"Training loss: {loss:.2f}", ha='center', fontsize=10, fontweight='bold')
        fig.text(0.5, 0.03, f"Input: {stims[i]}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
        fig.text(0.8, 0.03, f"Final FR: {final_column[-1]:.2f}", ha='center', fontsize=10, fontweight='bold')

        col_indices = [[0, 8, 16, 24], [32, 40, 48, 56]]

        for idx_1 in [0, 1]:
            for idx_2 in [0, 1, 2, 3]:

                idx_col = col_indices[idx_1][idx_2]
                axes[idx_1, idx_2].plot(firing_rates_cpu[:, i, idx_col + 0], label='L23e')
                axes[idx_1, idx_2].plot(firing_rates_cpu[:, i, idx_col + 4] * 0.1, label='L5e')
                axes[idx_1, idx_2].plot(firing_rates_cpu[:, i, idx_col + 6], label='L6e')

        plt.tight_layout(pad=3.0)
        fig.subplots_adjust(left=0.15)
        plt.savefig('../results/png/firing_rates_first8_{:02d}_{:1d}'.format(train_iter + 1, i))
        plt.close(fig)

def visualize_weights(network, train_iter):
    '''
    Visualize the learnable weights (ff and lateral)
    during training.
    '''
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')

    for name, param in network.named_parameters():
        param_data = param.detach().cpu().numpy()

        if np.sum(param_data) != 0:
            fig, ax = plt.subplots(figsize=(13, 7))

            if param_data.ndim == 2:  # 2D weight matrices: use heatmap
                heatmap = ax.imshow(param_data, cmap="viridis", interpolation="nearest")
                fig.colorbar(heatmap, ax=ax)
                ax.set_title(f"Weight Matrix: {name}")
            elif param_data.ndim == 1:  # output weights are 1D: use bar plot
                ax.bar(np.arange(len(param_data)), param_data, color="slateblue")
                ax.set_title(f"Bias Vector: {name}")
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")

            # Clean filename (remove problematic characters)
            clean_name = name.replace('.', '_')
            plt.savefig('../results/png/{}_{:02d}'.format(clean_name, train_iter + 1))
            plt.close(fig)

def make_ds(batch_size):
    '''
    Make a dataset of all possible combinations. Either with fixed position,
    or position-invariant (i.e. all possible
    '''
    all_combinations = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1.],
                                     [0., 0., 0., 0., 0., 0., 1., 1.],
                                     [0., 0., 0., 0., 0., 1., 1., 1.],
                                     [0., 0., 0., 0., 1., 1., 1., 1.],
                                     [0., 0., 0., 1., 1., 1., 1., 1.],
                                     [0., 0., 1., 1., 1., 1., 1., 1.],
                                     [0., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.],
                                    ], dtype=torch.float32)
    all_combinations *= 15.

    train_set = all_combinations[torch.randperm(all_combinations.size(0))][:batch_size]
    test_set = all_combinations[torch.randperm(all_combinations.size(0))][:1]
    return train_set, test_set

def prep_parity_stim(stim_raw, time_vec, num_columns):
    '''
    Prepare the stimulus with a pre-stimulus phase and
    stimulus-phase, with the specified time vector.
    '''
    phase_length = int(len(time_vec) / 2)
    stim_phase = stim_raw.repeat(phase_length, 1)  # use this stim_phase (1x4) instead of (4x64)

    empty_stim_phase = torch.zeros(stim_phase.shape)

    return torch.cat((empty_stim_phase, stim_phase), dim=0)  # (time steps, num inputs, num populations)


def init_network(device, nr_inputs, batch_size, two_output_cols):
    '''
    Initialize the network, initial state and time vector.
    '''
    col_params = load_config('../config/model.toml')

    if two_output_cols:
        network_architecture = {'nr_areas': 3,
                         'areas': ['mt', 'mt', 'mt'],
                         'nr_columns_per_area': [8, 2, 2],
                         'nr_input_units': nr_inputs}
    else:
        network_architecture = {'nr_areas': 4,
                         'areas': ['mt', 'mt', 'mt', 'mt'],
                         'nr_columns_per_area': [8, 4, 2, 1],
                         'nr_input_units': nr_inputs}
    network = ColumnNetwork(col_params, network_architecture)
    num_columns = sum(network_architecture['nr_columns_per_area'])

    stim_duration = 0.5
    dt = 1e-3
    time_steps = int(stim_duration * 2 / dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)

    initial_state = torch.zeros(batch_size, num_columns * 8 * 2)  # 2 state variables
    membrane_init = torch.tensor([-1.7997e-01, 8.3757e+00, 1.1346e+01, 1.1953e+01, -6.5426e+00, 1.0319e+01, -2.9719e+01, 1.2530e+01])
    num_populations_first_area = network.areas['0'].num_populations
    initial_state[:, :num_populations_first_area] = torch.tile(membrane_init, (batch_size, num_populations_first_area//8,))

    network.set_time_vec(time_vec)

    return network.to(device), time_vec.to(device), initial_state.to(device)

def mask_weights(network):
    '''
    Apply mask to grad to make sure no illegal updates are made.
    '''
    # network.output_weights.grad *= network.output_mask  # output weights

    network.areas['0'].input_weights.grad *= network.areas['0'].input_mask  # input weights

    for area_idx in range(1, network.nr_areas):  # feedforward weights, skip first area
        network.areas[str(area_idx)].feedforward_weights.grad *= network.areas[str(area_idx)].feedforward_mask

    for area_idx in range(network.nr_areas):
        if network.areas[str(area_idx)].num_columns > 1:  # areas with fewer than 1 column have no lateral connections
            network.areas[str(area_idx)].lateral_weights.grad *= network.areas[str(area_idx)].lateral_mask

def train_parity_ode(nr_inputs,
                     nr_samples,
                     batch_size,
                     device,
                     trajectory_based,
                     two_output_cols):
    '''
    Train a network to perform parity classification (even/odd)
    using a neural ODE to train feedforward and lateral weights.
    '''
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')

    network, time_vec, initial_state = init_network(device, nr_inputs, batch_size, two_output_cols)

    # # Load existing network
    # network = load_pkl_file('../results/parity_pre_training.pkl')

    # Save the network pre-training
    save_pkl_file('../results/png/parity_pre_training.pkl', network)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08)
    nr_batches = int(nr_samples/batch_size)

    # Load trajectory and define huber loss for trajectory training
    target_trajectory = load_pkl_file('../data/target_trajectory_parity.pkl')
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)

    losses = torch.Tensor(nr_batches)

    for batch_itr in range(nr_batches):
        optimizer.zero_grad()
        network.constrain()

        train_set, _ = make_ds(batch_size)
        train_set = train_set.to(device)
        network.set_stim(train_set)

        # Run neural ODE on train samples
        ode_output = odeint(network, initial_state, time_vec).to(device)

        split = network.network_as_area.num_populations
        firing_rates = compute_firing_rate(ode_output[:, :, :split] - ode_output[:, :, split:(split * 2)])

        # Compute loss based on the trajectory or the classification (final FR)
        if trajectory_based:
            # Training on trajectory
            if two_output_cols:
                output_col_fr = firing_rates[:, :, -16:]  # firing rates of output columns
                output_col_fr = (output_col_fr * network.output_weights) / network.output_scale
                output_col_fr_reshape = torch.stack([output_col_fr[:, :, :8], output_col_fr[:, :, 8:]])
                output_col_fr_summed = torch.sum(output_col_fr_reshape, dim=-1)

                target_trajectories = torch.zeros_like(output_col_fr_summed)
                parity_targets = (train_set.sum(dim=1) % 30 == 0).int()
                for target_idx, target in enumerate(parity_targets):
                    if target == 1:
                        target_trajectories[0, :, target_idx] = target_trajectory
                        target_trajectories[1, :, target_idx] = torch.zeros_like(target_trajectory)
                    elif target == 0:
                        target_trajectories[1, :, target_idx] = target_trajectory
                        target_trajectories[0, :, target_idx] = torch.zeros_like(target_trajectory)

                loss = hub_loss(output_col_fr_summed, target_trajectories)

            else:  # if only single output column
                output_col_fr = firing_rates[:, :, -8:]  # firing rates of output column
                output_col_fr_summed = torch.sum((output_col_fr * network.output_weights) / network.output_scale, dim=-1)

                target_trajectories = torch.Tensor(output_col_fr_summed.shape)
                parity_targets = (train_set.sum(dim=1) % 30 == 0).int()
                for target_idx, target in enumerate(parity_targets):
                    if target == 1:
                        target_trajectories[:, target_idx] = target_trajectory
                    elif target == 0:
                        target_trajectories[:, target_idx] = torch.zeros_like(target_trajectory)

                loss = hub_loss(output_col_fr_summed, target_trajectories)
        else:
            # Training on classification
            final_fr = firing_rates[-100:, :, -8:]  # final firing rates of output column
            final_fr_mean = torch.mean(final_fr, dim=0)  # mean firing rate over last 100 time steps
            final_fr_summed = torch.sum((final_fr_mean * (network.output_weights * network.output_mask)) , dim=-1)

            parity_targets = (train_set.sum(dim=1) % 30 == 0).float()
            parity_targets = parity_targets * 20.  # training target

            parity_targets = parity_targets
            loss = torch.mean(abs(final_fr_summed - parity_targets))

        loss.backward()

        # Make sure no illegal updates are made before making the optimizer step
        # mask_weights(network)
        optimizer.step()

        # Clamp the weights to ensure the weights are not below zero after updating (or are not higher than zero)
        for name, param in network.named_parameters():
            param.data.clamp_(min=0.0)

        print('Iter {:02d} | Total Loss {:.5f}'.format(batch_itr + 1, loss.item()))
        losses[batch_itr] = loss.item()
        save_pkl_file('../results/png/losses.pkl', losses)

        # Every five batches, visualize training and save the current network
        with torch.no_grad():
            if batch_itr % 5 == 0:
                visualize_results(network, firing_rates, train_set, loss.item(), batch_itr, batch_size, target_trajectory, two_output_cols)
                visualize_weights(network, batch_itr)
                save_pkl_file('../results/png/parity_post_training.pkl', network)
    pprint(losses)



if __name__ == '__main__':

    seed = 1
    device = torch.device("cpu")
    trajectory_based = False
    two_output_cols = False

    set_seed(seed)
    train_parity_ode(nr_inputs=8,
                     nr_samples=6400,
                     batch_size=8,
                     device=device,
                     trajectory_based=trajectory_based,
                     two_output_cols=two_output_cols)


'''
8 bits
8-4-2-1
fixed output weights

'''
