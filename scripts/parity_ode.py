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

from src.coupled_columns import ColumnNetwork
from src.utils import *



def visualize_results(network, firing_rates, stims, loss, train_iter, batch_size, target_trajectory, two_output_cols):
    '''
    Visualize the firing rates of the last few columns while training.
    '''
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')

    nr_samples = batch_size  # how many samples to visualize

    for i in range(nr_samples):
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))

        idx_col1 = 64  # first 8 columns
        axes[0, 0].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[0, 0].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[0, 0].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')
        fig.legend(loc="upper left")

        idx_col1 = 64 + 8
        axes[0, 1].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[0, 1].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[0, 1].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')

        idx_col1 = 64 + 16
        axes[1, 0].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[1, 0].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[1, 0].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')

        idx_col1 = 64 + 24
        axes[1, 1].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[1, 1].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[1, 1].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')

        idx_col1 = 64 + 32
        axes[0, 2].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[0, 2].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[0, 2].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')

        idx_col1 = 64 + 40
        axes[1, 2].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[1, 2].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[1, 2].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')

        # Final column
        if two_output_cols:
            final_column = torch.sum((firing_rates[i, :, 0, -16:-8] * network.output_weights[-16:-8]) / network.output_scale, dim=-1)
            final_column_2 = torch.sum((firing_rates[i, :, 0, -8:] * network.output_weights[-8:]) / network.output_scale, dim=-1)
            axes[0, 3].plot(final_column.detach().numpy())
            axes[0, 3].plot(target_trajectory, linestyle='--')
            axes[0, 3].set_title('Final column')
            axes[1, 3].plot(final_column_2.detach().numpy())
            axes[1, 3].plot(target_trajectory, linestyle='--')
            axes[1, 3].set_title('Final column')
        else:
            final_column = torch.sum((firing_rates[i, :, 0, -8:] * network.output_weights) / network.output_scale, dim=-1)
            axes[1, 3].plot(final_column.detach().numpy())
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
                axes[idx_1, idx_2].plot(firing_rates[i, :, 0, idx_col + 0].detach().numpy(), label='L23e')
                axes[idx_1, idx_2].plot(firing_rates[i, :, 0, idx_col + 4].detach().numpy() * 0.1, label='L5e')
                axes[idx_1, idx_2].plot(firing_rates[i, :, 0, idx_col + 6].detach().numpy(), label='L6e')

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

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_ds(nr_inputs, nr_samples, batch_size, fixed_position=True):
    '''
    Make a dataset of all possible combinations. Either with fixed position,
    or position-invariant (i.e. all possible
    '''
    if fixed_position:
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
        # all_combinations = torch.tile(all_combinations, (batch_size//nr_inputs, 1))

        train_set = all_combinations[torch.randperm(all_combinations.size(0))][:batch_size]
        test_set = all_combinations[torch.randperm(all_combinations.size(0))][:1]

    else:
        all_combinations = torch.tensor([[(i >> bit) & 1 for bit in reversed(range(nr_inputs))]
                                 for i in range(2 ** nr_inputs)], dtype=torch.float32)
        all_combinations *= 15.

        train_set = all_combinations[torch.randperm(all_combinations.size(0))][:batch_size]
        test_set = all_combinations[torch.randperm(all_combinations.size(0))][:1]

    return train_set, test_set

def prep_stim_ode(stim_raw, time_vec, num_columns):
    '''
    Prepare the stimulus with a pre-stimulus phase and
    stimulus-phase, with the specified time vector.
    '''
    phase_length = int(len(time_vec) / 2)
    stim_phase = stim_raw.repeat(phase_length, 1)  # use this stim_phase (1x4) instead of (4x64)

    empty_stim_phase = torch.zeros(stim_phase.shape)

    return torch.cat((empty_stim_phase, stim_phase), dim=0)  # (time steps, num inputs, num populations)


def init_network(device, two_output_cols):
    '''
    Initialize the network, initial state and time vector.
    '''
    col_params = load_config('../config/model.toml')

    if two_output_cols:
        network_input = {'nr_areas': 3,
                         'areas': ['mt', 'mt', 'mt'],
                         'nr_columns_per_area': [8, 4, 2],
                         'nr_input_units': nr_inputs}
    else:
        network_input = {'nr_areas': 3,
                         'areas': ['mt', 'mt', 'mt'],
                         'nr_columns_per_area': [8, 2, 1],
                         'nr_input_units': nr_inputs}
    network = ColumnNetwork(col_params, network_input, device)
    num_columns = sum(network_input['nr_columns_per_area'])

    stim_duration = 0.5
    dt = 1e-3
    time_steps = int(stim_duration * 2 / dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)

    initial_state = torch.zeros(num_columns * 8 * 3)  # 3 state variables
    membrane_init = torch.tensor([-1.7997e-01, 8.3757e+00, 1.1346e+01, 1.1953e+01, -6.5426e+00, 1.0319e+01, -2.9719e+01, 1.2530e+01])
    num_populations_first_area = network.areas['0'].num_populations
    initial_state[:num_populations_first_area] = torch.tile(membrane_init, (num_populations_first_area//8,))
    initial_state = initial_state.unsqueeze(0)

    network = network.to(device).to(torch.float32)
    initial_state = initial_state.to(device).to(torch.float32)
    time_vec = time_vec.to(device).to(torch.float32)

    network.time_vec = time_vec

    return network, time_vec, initial_state

def mask_weights(network):
    '''
    Apply mask to grad to make sure no illegal updates are made.
    '''
    network.output_weights.grad *= network.output_mask  # output weights

    network.areas['0'].input_weights.grad *= network.areas['0'].input_mask  # input weights

    for area_idx in range(1, network.nr_areas):  # feedforward weights, skip first area
        network.areas[str(area_idx)].feedforward_weights.grad *= network.areas[str(area_idx)].feedforward_mask

    for area_idx in range(network.nr_areas):
        if network.areas[str(area_idx)].num_columns > 1:  # areas with fewer than 1 column have no lateral connections
            network.areas[str(area_idx)].lateral_weights.grad *= network.areas[str(area_idx)].lateral_mask

def train_parity_ode(nr_inputs, nr_samples, batch_size, device, two_output_cols=True):
    '''
    Train a network to perform parity classification (even/odd)
    using a neural ODE to train feedforward and lateral weights.
    '''
    train_set, _ = make_ds(nr_inputs, nr_samples, batch_size)

    network, time_vec, initial_state = init_network(device, two_output_cols)
    num_populations = network.network_as_area.num_populations

    # # Load existing network
    # with open('../results/parity_8bit_classification_seed1_onlyL5output_L23einhibi/parity_post_training.pkl', 'rb') as f:
    #     network = pickle.load(f)

    # Save the network pre-training
    with open('../parity_pre_training.pkl', 'wb') as f:
        pickle.dump(network, f)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08)
    nr_batches = int(nr_samples/batch_size)

    # Load trajectory and define huber loss for trajectory training
    with open('../dream_trajectory_20Hz.pkl', 'rb') as f:
        target_trajectory_raw = pickle.load(f)
    target_trajectory = target_trajectory_raw[:1000]
    target_trajectory = (target_trajectory*1.1) - 2.5  # rescale the trajectory and adjust for resting state level
    target_trajectory = torch.clamp(target_trajectory, min=0.0)  # make sure trajectory is not below zero
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)

    for batch_itr in range(nr_batches):
        optimizer.zero_grad()

        train_set, _ = make_ds(nr_inputs, nr_samples, batch_size, fixed_position=True)  # change last variable to false if position should be varied

        # Storing results
        batch_output = torch.Tensor(batch_size, len(time_vec), 1, num_populations*3)  # *3 bc 3 state variables
        batch_output = batch_output.to(device)

        # Run neural ODE on train samples
        for itr, train_stim in enumerate(train_set):

            stim_ode = prep_stim_ode(train_stim, time_vec, network.areas['0'].num_columns)
            stim_ode = stim_ode.to(device).to(torch.float32)
            network.stim = stim_ode

            ode_output = odeint(network, initial_state, time_vec)
            # ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

            batch_output[itr, :, :, :] = ode_output

        split = network.network_as_area.num_populations
        firing_rates = compute_firing_rate(batch_output[:, :, :, :split] - batch_output[:, :, :, split:(split * 2)])

        # # Compute loss and update weights || Training on classification
        # final_fr = firing_rates[:, -100:, 0, -8:]  # final firing rates of output column
        # final_fr_mean = torch.mean(final_fr, dim=1)  # mean firing rate over last 100 time steps
        # final_fr_summed = torch.sum((final_fr_mean * network.output_weights) / network.output_scale , dim=-1)
        #
        # parity_targets = (train_set.sum(dim=1) % 30 == 0).float()
        # # parity_targets = torch.ones(8)
        # parity_targets = parity_targets * 20.  # training target
        #
        # parity_targets = parity_targets.to(device).to(torch.float32)
        # loss = torch.mean(abs(final_fr_summed - parity_targets))

        # Compute loss and update weights || Training on trajectory
        if two_output_cols:
            output_col_fr = firing_rates[:, :, 0, -16:]  # firing rates of output columns
            output_col_fr = (output_col_fr * network.output_weights) / network.output_scale
            output_col_fr_reshape = torch.stack([output_col_fr[:, :, :8], output_col_fr[:, :, 8:]])
            output_col_fr_summed = torch.sum(output_col_fr_reshape, dim=-1)

            target_trajectories = torch.Tensor(output_col_fr_summed.shape)
            parity_targets = (train_set.sum(dim=1) % 30 == 0).int()
            for target_idx, target in enumerate(parity_targets):
                if target == 1:
                    target_trajectories[0, target_idx, :] = target_trajectory
                    target_trajectories[1, target_idx, :] = torch.zeros(output_col_fr_summed.shape[-1])
                elif target == 0:
                    target_trajectories[1, target_idx, :] = target_trajectory
                    target_trajectories[0, target_idx, :] = torch.zeros(output_col_fr_summed.shape[-1])

            loss = hub_loss(output_col_fr_summed, target_trajectories)

        else:  # if only single output column
            output_col_fr = firing_rates[:, :, 0, -8:]  # firing rates of output column
            output_col_fr_summed = torch.sum((output_col_fr * network.output_weights) / network.output_scale, dim=-1)

            target_trajectories = torch.Tensor(output_col_fr_summed.shape)
            parity_targets = (train_set.sum(dim=1) % 30 == 0).int()
            for target_idx, target in enumerate(parity_targets):
                if target == 1:
                    target_trajectories[target_idx] = target_trajectory
                elif target == 0:
                    target_trajectories[target_idx] = torch.zeros(output_col_fr_summed.shape[1])

            loss = hub_loss(output_col_fr_summed, target_trajectories)

        loss.backward()

        print('Iter {:02d} | Total Loss {:.5f}'.format(batch_itr + 1, loss.item()))

        # Check for exploding/vanishing gradients
        for name, param in network.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"None gradient in {name}")
            if param.requires_grad and torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            if param.requires_grad and torch.norm(param.grad) > 1e4:
                print(f"Large gradient in {name}: {torch.norm(param.grad)}")

        # Make sure no illegal updates are made before making the optimizer step
        mask_weights(network)
        optimizer.step()

        # Clamp the weights to ensure the weights are not below zero after updating (or are not higher than zero)
        for name, param in network.named_parameters():
            param.data.clamp_(min=0.0)
            # if 'lateral' in name:
            #     param.data.clamp_(max=0.0)  # lateral inhibition weights can not be positive
            # if 'lateral' not in name:
            #     param.data.clamp_(min=0.0)  # other weights can not be negative
            # if 'output' in name:
            #     param.data.clamp_(min=0.0)

        # Every five batches, visualize training and save the current network
        with torch.no_grad():
            if batch_itr % 5 == 0:
                visualize_results(network, firing_rates, train_set, loss.item(), batch_itr, batch_size, target_trajectory, two_output_cols)
                visualize_weights(network, batch_itr)
                with open('../parity_post_training.pkl', 'wb') as f:
                    pickle.dump(network, f)



if __name__ == '__main__':

    nr_inputs = 8
    nr_samples = 6400
    batch_size = 8
    seed = 1
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")

    set_seed(seed)
    train_parity_ode(nr_inputs, nr_samples, batch_size, device)


'''
8 bits
8-4-2
original ff weight masks (only L23 init)
L23e -> L23i lateral inhibition weights (so they are positive!)
learnable output weights

double trajectory

seed 1
'''
