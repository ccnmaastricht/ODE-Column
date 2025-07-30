import matplotlib.pyplot as plt
import os
import itertools
import random
import pickle
import torch
from pprint import pprint

from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from src.coupled_columns import ColumnNetwork
from src.utils import *




# def make_ds(nr_inputs, nr_samples, batch_size):
#     all_combinations = torch.tensor([[(i >> bit) & 1 for bit in reversed(range(nr_inputs))]
#                              for i in range(2 ** nr_inputs)], dtype=torch.float32)
#     all_combinations *= 15.
#
#     train_set = all_combinations[torch.randperm(all_combinations.size(0))][:batch_size]
#     test_set = all_combinations[torch.randperm(all_combinations.size(0))][:1]
#
#     return train_set, test_set

def make_ds(nr_inputs, nr_samples, batch_size):

    # Generate all binary combinations for the given input size
    all_inputs = list(itertools.product([0, 1], repeat=nr_inputs))

    # Group by number of active units (Hamming weight)
    hamming_groups = {i: [] for i in range(nr_inputs + 1)}
    for vec in all_inputs:
        hamming_groups[sum(vec)].append(vec)

    # Add [0, 0, ..., 0] case only once
    dataset = [hamming_groups[0][0]]

    # Number of groups to balance (excluding the zero case)
    num_active_groups = nr_inputs
    samples_per_group = (batch_size - 1) // num_active_groups

    for h in range(1, nr_inputs + 1):
        group = hamming_groups[h]
        for _ in range(samples_per_group):
            sample = random.choice(group)
            dataset.append(sample)

    # Pad to reach desired batch size
    while len(dataset) < batch_size:
        dataset.append(random.choice(hamming_groups[nr_inputs]))

    # Shuffle
    random.shuffle(dataset)

    # Convert to tensor and scale inputs to 0–15
    X = torch.tensor(dataset, dtype=torch.float32) * 15.0
    y = (X.sum(dim=1) / 15 % 2).long()  # parity of active units

    # # Train/test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, stratify=y, random_state=42
    # )
    return X, y


def prep_stim_ode(stim_raw, time_vec, num_columns):

    all_inputs = []

    for i, stim_i in enumerate(stim_raw):

        input_column = torch.zeros(8)
        ff_indices = [2,3]
        input_column[ff_indices] = stim_i

        input_all_columns = torch.tile(input_column, (num_columns,))
        all_inputs.append(input_all_columns)

    all_inputs = torch.stack(all_inputs)

    phase_length = int(len(time_vec) / 2)
    # stim_phase = all_inputs.unsqueeze(0).repeat(phase_length, 1, 1)
    stim_phase = stim_raw.repeat(phase_length, 1)  # use this stim_phase (1x4) instead of (4x64)

    empty_stim_phase = torch.zeros(stim_phase.shape)

    return torch.cat((empty_stim_phase, stim_phase), dim=0)  # (time steps, num inputs, num populations)


def init_network(device):
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')

    network_input = {'nr_areas': 3,
                     'areas': ['mt', 'mt', 'mt'],
                     'nr_columns_per_area': [8, 4, 1],
                     'nr_input_units': nr_inputs}
    network = ColumnNetwork(col_params, network_input, device)
    num_columns = sum(network_input['nr_columns_per_area'])

    time_steps = int(sim_params['protocol']['stimulus_duration'] * 2 / sim_params['time_step'])
    time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)

    initial_state = torch.zeros(num_columns * 8 * 3)  # 3 state variables
    initial_state = initial_state.unsqueeze(0)

    network = network.to(device).to(torch.float32)
    initial_state = initial_state.to(device).to(torch.float32)
    time_vec = time_vec.to(device).to(torch.float32)

    network.time_vec = time_vec

    return network, time_vec, initial_state


def visualize_results(network, firing_rates, stims, loss, train_iter):

    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')

    nr_samples = 16  # how many to visualize

    for i in range(nr_samples):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))

        final_column = torch.sum((firing_rates[i, :, 0, -8:] * network.output_weights) / network.output_scale, dim=-1)

        fig.text(0.2, 0.03, f"Training loss: {loss:.2f}", ha='center', fontsize=10, fontweight='bold')
        fig.text(0.5, 0.03, f"Input: {stims[i]}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
        fig.text(0.8, 0.03, f"Final FR: {final_column[-1]:.2f}", ha='center', fontsize=10, fontweight='bold')

        idx_col1 = 208 + 64  # membrane,adap + first 8 columns
        axes[0, 0].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[0, 0].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[0, 0].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')
        fig.legend(loc="upper left")

        idx_col1 = 208 + 64 + 8
        axes[0, 1].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[0, 1].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[0, 1].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')

        idx_col1 = 208 + 64 + 16
        axes[1, 0].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[1, 0].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[1, 0].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')

        idx_col1 = 208 + 64 + 24
        axes[1, 1].plot(firing_rates[i, :, 0, idx_col1 + 0].detach().numpy(), label='L23e')
        axes[1, 1].plot(firing_rates[i, :, 0, idx_col1 + 4].detach().numpy() * 0.1, label='L5e')
        axes[1, 1].plot(firing_rates[i, :, 0, idx_col1 + 6].detach().numpy(), label='L6e')

        axes[1, 2].plot(final_column.detach().numpy())
        axes[1, 2].set_title('Final column')

        plt.tight_layout(pad=3.0)
        fig.subplots_adjust(left=0.15)
        plt.savefig('../results/png/firing_rates_{:02d}_{:1d}'.format(train_iter + 1, i))
        plt.close(fig)


def visualize_weights(network, train_iter):

    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')

    for name, param in network.named_parameters():
        param_data = param.detach().cpu().numpy()

        if np.sum(param_data) != 0:
            fig, ax = plt.subplots(figsize=(13, 7))

            if param_data.ndim == 2:  # 2D weight matrix → use heatmap
                heatmap = ax.imshow(param_data, cmap="viridis", interpolation="nearest")
                fig.colorbar(heatmap, ax=ax)
                ax.set_title(f"Weight Matrix: {name}")
            elif param_data.ndim == 1:  # 1D bias vector → use bar plot
                ax.bar(np.arange(len(param_data)), param_data, color="slateblue")
                ax.set_title(f"Bias Vector: {name}")
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")

            # Clean filename (remove problematic characters)
            clean_name = name.replace('.', '_')
            plt.savefig('../results/png/{}_{:02d}'.format(clean_name, train_iter + 1))
            plt.close(fig)


def mask_weights(network):
    '''
    Apply mask to grad to make sure no illegal updates are made.
    '''
    network.output_weights.grad *= network.output_mask  # output weights

    network.areas['0'].input_weights.grad *= network.areas['0'].input_mask  # input weights

    for area_idx in range(1, network.nr_areas):  # feedforward weights, skip first
        network.areas[str(area_idx)].feedforward_weights.grad *= network.areas[str(area_idx)].feedforward_mask

    for area_idx in range(network.nr_areas - 1):  # lateral weights, skip last
        network.areas[str(area_idx)].lateral_weights.grad *= network.areas[str(area_idx)].lateral_mask

    for area_idx in range(network.nr_areas - 1):  # feedback weights, skip last
        network.areas[str(area_idx)].feedback_weights.grad *= network.areas[str(area_idx)].feedback_mask


def train_parity_ode(nr_inputs, nr_samples, batch_size, device):

    network, time_vec, initial_state = init_network(device)
    num_populations = network.network_as_area.num_populations

    with open('../parity_pre_training.pkl', 'wb') as f:
        pickle.dump(network, f)

    optimizer = torch.optim.RMSprop(network.parameters(), lr=1e-4, alpha=0.95)

    # for name, param in network.named_parameters():
    #     print(name)
    #     # print(param)

    nr_batches = int(nr_samples/batch_size)

    for batch_itr in range(nr_batches):
        optimizer.zero_grad()

        train_set, _ = make_ds(nr_inputs, nr_samples, batch_size)

        # Storing results
        batch_output = torch.Tensor(batch_size, len(time_vec), 1, num_populations*3)  # *3 bc 3 state variables
        batch_output = batch_output.to(device)

        # Run neural ODE on train samples
        for itr, train_stim in enumerate(train_set):

            # print(itr)
            # print(train_stim)

            # train_stim = torch.tensor([15., 15., 15., 15.])

            stim_ode = prep_stim_ode(train_stim, time_vec, network.areas['0'].num_columns)
            stim_ode = stim_ode.to(device).to(torch.float32)
            network.stim = stim_ode

            ode_output = odeint(network, initial_state, time_vec)
            # ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'})

            # for i in range(104):
            #     print(i)
            #     # plt.plot(ode_output[:, :, -8].detach().numpy())
            #     # plt.plot(ode_output[:, :, 64+0+i].detach().numpy())
            #     plt.plot(ode_output[:, :, 64+208+i].detach().numpy())
            #     plt.show()

            batch_output[itr, :, :, :] = ode_output

        # Compute loss and update weights
        final_fr = batch_output[:, -1, 0, -8:]  # final firing rate of output column
        final_fr_summed = torch.sum((final_fr * network.output_weights) / network.output_scale , dim=-1)

        parity_targets = (train_set.sum(dim=1) % 30 == 0).float()
        parity_targets = parity_targets * 20.

        parity_targets = parity_targets.to(device).to(torch.float32)
        loss = torch.mean(abs(final_fr_summed - parity_targets))
        loss.backward()

        print('Iter {:02d} | Total Loss {:.5f}'.format(batch_itr + 1, loss.item()))

        # for name, param in network.named_parameters():
        #     print(name)
        #     print(param.grad)

        for name, param in network.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"None gradient in {name}")
                print(param)
            if param.requires_grad and torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            if param.requires_grad and torch.norm(param.grad) > 1e4:
                print(f"Large gradient in {name}: {torch.norm(param.grad)}")

        mask_weights(network)

        optimizer.step()

        for name, param in network.named_parameters():
            param.data.clamp_(min=0.0)  # weights can not be negative
            if 'output' in name:
                param.data.clamp_(min=0.0, max=network.output_scale)

        with torch.no_grad():
            visualize_results(network, batch_output, train_set, loss.item(), batch_itr)
            visualize_weights(network, batch_itr)

            if batch_itr % 5 == 0:
                with open('../parity_post_training.pkl', 'wb') as f:
                    pickle.dump(network, f)




if __name__ == '__main__':

    nr_inputs = 4 # 16 combinations
    nr_samples = 1600
    batch_size = 16
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")

    train_parity_ode(nr_inputs, nr_samples, batch_size, device)

