import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import torch

from src.xor_columns import ColumnsXOR
from src.coupled_columns import ColumnNetwork
from src.utils import *
from torchsde import sdeint



def vis_xor_results(firing_rates, stim, train_loss, iter1, iter2):
    '''
    Visualizes the training process of the XOR classification task. Plots the
    firing rates of column A, B and C and reports the training loss, the input
    condition (XOR or AND) and the final firing rate of column C, what determines
    the network's output.
    Images are saved in ../results/png.
    '''

    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if stim[2] != stim[10]:
        condition = "diff input - XOR"
    else:
        condition = "same input - AND"

    fig.text(0.2, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.5, 0.03, f"Input: {condition}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Final FR: {firing_rates[-1, 16]:.2f}", ha='center', fontsize=10, fontweight='bold')

    axes[0].plot(firing_rates[:, 0], label='col A')
    axes[0].plot(firing_rates[:, 8], label='col B')
    axes[0].set_ylim(0.0, 3.0)
    axes[0].set_title("Firing rates L2/3e in column A and B")
    fig.legend(loc="upper left")

    axes[1].plot(firing_rates[:, 16], label='col C')
    axes[1].set_ylim(0.0, 3.0)
    axes[1].set_title("Firing rates L2/3e in column C")

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig('../results/png/{:02d}_{:1d}'.format(iter1+1, iter2))
    plt.close(fig)

def make_stim():
    '''
    Creates a set of 4 stimuli, one for each XOR condition (0,0), (0,1),
    (1,0), (1,1). 1s are randomly set between 0.975 and 1.025.
    '''

    stims = torch.Tensor(4, 16)
    conditions = torch.tensor([[0., 0.],
                               [1., 0.],
                               [0., 1.],
                               [1., 1.]])
    for i in range(4):
        rand_stim = conditions[i] * torch.empty(1).uniform_(0.975, 1.025)
        stim = create_feedforward_input(16, rand_stim[0], rand_stim[1])
        stims[i, :] = stim

    # Shuffle along the first dimension
    stims_shuffled = stims[torch.randperm(stims.size(0))]
    return stims_shuffled

def prep_stim_ode(stim_raw, time_vec):
    '''
    Prepare the raw stimulus for the ODE. Outputs a tensor of length
    time steps with a pre-stimulus and a stimulus period.
    Contains two mirrored stimulus instances to input both stimuli
    to both input columns.
    '''
    empty_stim = torch.zeros(stim_raw.shape)

    phase_length = int(len(time_vec) / 2)
    empty_stim_phase = empty_stim.expand(phase_length, -1)
    stim_phase = stim_raw.expand(phase_length, -1)

    whole_stim_phase = torch.cat((empty_stim_phase, stim_phase), dim=0)

    # Double the stim to input it to both columns
    mirror_stim_phase = torch.cat((whole_stim_phase[:, 8:], whole_stim_phase[:, :8]), dim=1)
    return torch.stack((whole_stim_phase, mirror_stim_phase), dim=1)  # (time steps, 2, num populations)

def run_four_xor_samples(network, initial_state, time_vec, time_steps, batch_size=4, mode="training"):
    '''
    Runs the network once on all four XOR conditions, using a neural ODE.
    Can be used for either training or testing the model (set mode).
    '''

    # Storing results
    batch_output = torch.Tensor(batch_size, time_steps, 1, 48)
    stim_batch = torch.Tensor(batch_size, 16)

    # Run neural ODE on four samples
    for batch_iter in range(int(batch_size / 4)):
        four_stims = make_stim()

        for stim_iter, stim in enumerate(four_stims):
            itr = (batch_iter * 4) + stim_iter
            stim_batch[itr, :] = stim
            stim_ode = prep_stim_ode(stim, time_vec)

            network.stim = stim_ode  # new!

            # ode_output = network.run_ode_network(initial_state, time_vec)  # torchdiffeq
            ode_output = sdeint(network, initial_state, time_vec)

            batch_output[itr, :, :, :] = ode_output

    # Compute firing rates
    firing_rates = compute_firing_rate_torch(batch_output[:, :, :, :24] - batch_output[:, :, :, 24:])
    firing_rates = firing_rates.squeeze(dim=2)

    # Use final firing rate to compute model output
    final_fr_C = firing_rates[:, -1, 16]  # final firing rates column C
    xor_output = min_max(final_fr_C)
    # xor_output = fr_to_binary(final_fr_C)

    # Compute loss with binary targets
    xor_targets = (stim_batch[:, 2] != stim_batch[:, 10]).int()
    loss = torch.mean(abs(xor_output - xor_targets))

    if mode == "training":
        return xor_output, loss
    elif mode == "testing":
        return firing_rates, stim_batch, xor_targets


def train_xor_ode(nr_samples, nr_test_samples, batch_size):
    '''
    Train an ODE to solve the XOR problem
    '''

    # Initialize the network
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')

    network_input = {'nr_areas': 2, 'areas': ['mt', 'mt'], 'nr_columns_per_area': [2, 1], 'nr_input_units': 2}
    network = ColumnNetwork(col_params, network_input)

    # Initial state is always the same
    membrane = torch.tensor(sim_params['initial_conditions']['membrane_potential'])
    membrane = torch.tile(membrane, (3,))  # extent to three columns
    adaptation = torch.tensor(sim_params['initial_conditions']['adaptation'])
    adaptation = torch.tile(adaptation, (3,))

    initial_state = torch.concat((membrane, adaptation))
    initial_state = initial_state.unsqueeze(0)  # shape (1,48) for sde

    # Initialize the optimizer and add weights as learnable parameters
    optimizer = torch.optim.RMSprop([
                                    network.feedforward_weights[0][0],
                                    network.feedforward_weights[0][1],
                                    network.feedforward_weights[1][0],
                                    network.feedforward_weights[1][1]
                                     ], lr=1e-4, alpha=0.95)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  # higher gamma = slower decay

    nr_batches = int(nr_samples/batch_size)
    time_steps = int(sim_params['protocol']['stimulus_duration'] * 2 / sim_params['time_step'])
    time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)

    network.time_vec = time_vec  # new!

    # Store results
    accuracy_sigmoid = []
    accuracy_binary = []

    for itr in range(nr_batches):
        optimizer.zero_grad()

        xor_output, loss = run_four_xor_samples(network, initial_state, time_vec, time_steps,
                                                batch_size=batch_size, mode="training")
        loss.backward()

        with torch.no_grad():  # use masks to make sure not to update illegal connections
            network.feedforward_weights[0][0].grad *= torch.tile(network.ff_target_mask, (2,))
            network.feedforward_weights[0][1].grad *= torch.tile(network.ff_target_mask, (2,))
            network.feedforward_weights[1][0].grad *= network.ff_target_mask
            network.feedforward_weights[1][1].grad *= network.ff_target_mask

        optimizer.step()
        scheduler.step()

        print('Iter {:02d} | Total Loss {:.5f}'.format(itr + 1, loss.item()))

        # Test ODE model and visualize results
        with torch.no_grad():

            final_fr_rates = []
            xor_targets = []

            for i in range(int(nr_test_samples / 4)):
                fr_rates, stims, targets = run_four_xor_samples(network, initial_state, time_vec, time_steps,
                                                       batch_size=4, mode="testing")

                for j in range(4):  # Store results
                    final_fr_rates.append(fr_rates[j, -1, 16].item())
                    xor_targets.append(targets[j].item())

            for test_itr in range(4):  # Visualize last 4 test samples and save figures
                vis_xor_results(fr_rates[test_itr], stims[test_itr], loss.item(), itr, test_itr)

            # Threshold the firing rates using the mean ~ binary predictions
            threshold = np.mean(final_fr_rates)
            xor_preds_binary = [1 if x > threshold else 0 for x in final_fr_rates]

            # Sigmoid-classify the firing rates ~ non-binary predictions
            xor_preds_sigmoid = fr_to_binary(torch.tensor(final_fr_rates)).numpy()

            # Compute accuracies
            TP_sigmoid = [xor_preds_sigmoid[k] if xor_preds_sigmoid[k] > 0.5 and xor_targets[k] == 1 else 0 for k in range (nr_test_samples)]
            TN_sigmoid = [1 - (xor_preds_sigmoid[k]) if xor_preds_sigmoid[k] < 0.5 and xor_targets[k] == 0 else 0 for k in range(nr_test_samples)]
            TP_binary = [1 if xor_preds_binary[k] == 1 and xor_targets[k] == 1 else 0 for k in range (nr_test_samples)]
            TN_binary = [1 if xor_preds_binary[k] == 0 and xor_targets[k] == 0 else 0 for k in range(nr_test_samples)]

            accuracy_sigmoid.append((np.sum(TP_sigmoid) + np.sum(TN_sigmoid)).item() / nr_test_samples)
            accuracy_binary.append((np.sum(TP_binary) + np.sum(TN_binary)).item() / nr_test_samples)

            # print('     FF weights input 1: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(odefunc.ff_weights_1[2],
            #                                                                        odefunc.ff_weights_1[3],
            #                                                                        odefunc.ff_weights_1[10],
            #                                                                        odefunc.ff_weights_1[11]))
            # print('     FF weights input 2: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(odefunc.ff_weights_2[2],
            #                                                                        odefunc.ff_weights_2[3],
            #                                                                        odefunc.ff_weights_2[10],
            #                                                                        odefunc.ff_weights_2[11]))
            # print('     FF weights col A to C: {:.4f}, {:.4f}'.format(odefunc.ff_weights_AC[2],
            #                                                                        odefunc.ff_weights_AC[3]))
            # print('     FF weights col B to C: {:.4f}, {:.4f}'.format(odefunc.ff_weights_BC[2],
            #                                                                        odefunc.ff_weights_BC[3]))

    return accuracy_sigmoid, accuracy_binary


if __name__ == '__main__':
    nr_samples = 80
    nr_test_samples = 4
    batch_size = 4

    acc1, acc2 = train_xor_ode(nr_samples, nr_test_samples, batch_size)



# import matplotlib.pyplot as plt
# import os
# import pickle
# import numpy as np
# import torch
#
# from src.xor_columns import ColumnsXOR
# from src.utils import *
#
#
#
# def vis_xor_results(firing_rates, stim, train_loss, iter1, iter2):
#     '''
#     Visualizes the training process of the XOR classification task. Plots the
#     firing rates of column A, B and C and reports the training loss, the input
#     condition (XOR or AND) and the final firing rate of column C, what determines
#     the network's output.
#     Images are saved in ../results/png.
#     '''
#
#     if not os.path.exists('../results/png'):
#         os.makedirs('../results/png')
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
#     if stim[2] != stim[10]:
#         condition = "diff input - XOR"
#     else:
#         condition = "same input - AND"
#
#     fig.text(0.2, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')
#     fig.text(0.5, 0.03, f"Input: {condition}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
#     fig.text(0.8, 0.03, f"Final FR: {firing_rates[-1, 16]:.2f}", ha='center', fontsize=10, fontweight='bold')
#
#     axes[0].plot(firing_rates[:, 0], label='col A')
#     axes[0].plot(firing_rates[:, 8], label='col B')
#     axes[0].set_ylim(0.0, 3.0)
#     axes[0].set_title("Firing rates L2/3e in column A and B")
#     fig.legend(loc="upper left")
#
#     axes[1].plot(firing_rates[:, 16], label='col C')
#     axes[1].set_ylim(0.0, 3.0)
#     axes[1].set_title("Firing rates L2/3e in column C")
#
#     plt.tight_layout(pad=3.0)
#     fig.subplots_adjust(left=0.15)
#     plt.savefig('../results/png/{:02d}_{:1d}'.format(iter1+1, iter2))
#     plt.close(fig)
#
# def make_stim():
#     '''
#     Creates a set of 4 stimuli, one for each XOR condition (0,0), (0,1),
#     (1,0), (1,1). 1s are randomly set between 0.975 and 1.025.
#     '''
#
#     stims = torch.Tensor(4, 16)
#     conditions = torch.tensor([[0., 0.],
#                                [1., 0.],
#                                [0., 1.],
#                                [1., 1.]])
#     for i in range(4):
#         rand_stim = conditions[i] * torch.empty(1).uniform_(0.975, 1.025)
#         stim = create_feedforward_input(16, rand_stim[0], rand_stim[1])
#         stims[i, :] = torch.tensor(stim)
#
#     # Shuffle along the first dimension
#     stims_shuffled = stims[torch.randperm(stims.size(0))]
#     return stims_shuffled
#
# def run_four_xor_samples(odefunc, initial_state, time_vec, time_steps, batch_size=4, mode="training"):
#     '''
#     Runs the network once on all four XOR conditions, using a neural ODE.
#     Can be used for either training or testing the model (set mode).
#     '''
#
#     # Storing results
#     batch_output = torch.Tensor(batch_size, time_steps, 2, 24)
#     stim_batch = torch.Tensor(batch_size, 16)
#
#     # Run neural ODE on four samples
#     for batch_iter in range(int(batch_size / 4)):
#         four_stims = make_stim()
#
#         for stim_iter, stim in enumerate(four_stims):
#             itr = (batch_iter * 4) + stim_iter
#             stim_batch[itr, :] = stim
#
#             ode_output = odefunc.run_ode_xor(initial_state, stim, time_vec, num_stim_phases=2)
#             batch_output[itr, :, :, :] = ode_output
#
#     # Compute firing rates
#     firing_rates = compute_firing_rate_torch(batch_output[:, :, 0, :] - batch_output[:, :, 1, :])
#
#     # Use final firing rate to compute model output
#     final_fr_C = firing_rates[:, -1, 16]  # final firing rates column C
#     xor_output = min_max(final_fr_C)
#     # xor_output = fr_to_binary(final_fr_C)
#
#     # Compute loss with binary targets
#     xor_targets = (stim_batch[:, 2] != stim_batch[:, 10]).int()
#     loss = torch.mean(abs(xor_output - xor_targets))
#
#     if mode == "training":
#         return xor_output, loss
#     elif mode == "testing":
#         return firing_rates, stim_batch, xor_targets
#
#
# def train_xor_ode(nr_samples, nr_test_samples, batch_size):
#     '''
#     Train an ODE to solve the XOR problem
#     '''
#
#     # Initialize the ODE function
#     col_params = load_config('../config/model.toml')
#     sim_params = load_config('../config/simulation.toml')
#     odefunc = ColumnsXOR(col_params, 'MT')
#
#     # Initial state is always the same
#     membrane = torch.tensor(sim_params['initial_conditions']['membrane_potential'])
#     adaptation = torch.tensor(sim_params['initial_conditions']['adaptation'])
#     initial_state = torch.stack((membrane, adaptation))
#     initial_state = torch.cat((initial_state, initial_state[:, :8]), dim=-1)  # extent to three columns
#
#     # Initialize the optimizer and add weights as learnable parameters
#     optimizer = torch.optim.RMSprop([odefunc.ff_weights_1,
#                                     odefunc.ff_weights_2,
#                                     odefunc.ff_weights_AC,
#                                     odefunc.ff_weights_BC
#                                      ], lr=1e-4, alpha=0.95)
#
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  # higher gamma = slower decay
#
#     nr_batches = int(nr_samples/batch_size)
#     time_steps = int(sim_params['protocol']['stimulus_duration'] * 2 / sim_params['time_step'])
#     time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)
#
#     # Store results
#     accuracy_sigmoid = []
#     accuracy_binary = []
#
#     for itr in range(nr_batches):
#         optimizer.zero_grad()
#
#         xor_output, loss = run_four_xor_samples(odefunc, initial_state, time_vec, time_steps,
#                                                 batch_size=batch_size, mode="training")
#         loss.backward()
#
#         with torch.no_grad():  # make sure not to update illegal connections by using masks
#             odefunc.ff_weights_1.grad *= odefunc.ff_weights_mask
#             odefunc.ff_weights_2.grad *= odefunc.ff_weights_mask
#             odefunc.ff_weights_AC.grad *= odefunc.ff_weights_mask[:8]
#             odefunc.ff_weights_BC.grad *= odefunc.ff_weights_mask[:8]
#
#         optimizer.step()
#         scheduler.step()
#
#         print('Iter {:02d} | Total Loss {:.5f}'.format(itr + 1, loss.item()))
#
#         # Test ODE model and visualize results
#         with torch.no_grad():
#
#             final_fr_rates = []
#             xor_targets = []
#
#             for i in range(int(nr_test_samples / 4)):
#                 fr_rates, stims, targets = run_four_xor_samples(odefunc, initial_state, time_vec, time_steps,
#                                                        batch_size=4, mode="testing")
#
#                 for j in range(4):  # Store results
#                     final_fr_rates.append(fr_rates[j, -1, 16].item())
#                     xor_targets.append(targets[j].item())
#
#             for test_itr in range(4):  # Visualize last 4 test samples and save figures
#                 vis_xor_results(fr_rates[test_itr], stims[test_itr], loss.item(), itr, test_itr)
#
#             # Threshold the firing rates using the mean ~ binary predictions
#             threshold = np.mean(final_fr_rates)
#             xor_preds_binary = [1 if x > threshold else 0 for x in final_fr_rates]
#
#             # Sigmoid-classify the firing rates ~ non-binary predictions
#             xor_preds_sigmoid = fr_to_binary(torch.tensor(final_fr_rates)).numpy()
#
#             # Compute accuracies
#             TP_sigmoid = [xor_preds_sigmoid[k] if xor_preds_sigmoid[k] > 0.5 and xor_targets[k] == 1 else 0 for k in range (nr_test_samples)]
#             TN_sigmoid = [1 - (xor_preds_sigmoid[k]) if xor_preds_sigmoid[k] < 0.5 and xor_targets[k] == 0 else 0 for k in range(nr_test_samples)]
#             TP_binary = [1 if xor_preds_binary[k] == 1 and xor_targets[k] == 1 else 0 for k in range (nr_test_samples)]
#             TN_binary = [1 if xor_preds_binary[k] == 0 and xor_targets[k] == 0 else 0 for k in range(nr_test_samples)]
#
#             accuracy_sigmoid.append((np.sum(TP_sigmoid) + np.sum(TN_sigmoid)).item() / nr_test_samples)
#             accuracy_binary.append((np.sum(TP_binary) + np.sum(TN_binary)).item() / nr_test_samples)
#
#             # print('     FF weights input 1: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(odefunc.ff_weights_1[2],
#             #                                                                        odefunc.ff_weights_1[3],
#             #                                                                        odefunc.ff_weights_1[10],
#             #                                                                        odefunc.ff_weights_1[11]))
#             # print('     FF weights input 2: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(odefunc.ff_weights_2[2],
#             #                                                                        odefunc.ff_weights_2[3],
#             #                                                                        odefunc.ff_weights_2[10],
#             #                                                                        odefunc.ff_weights_2[11]))
#             # print('     FF weights col A to C: {:.4f}, {:.4f}'.format(odefunc.ff_weights_AC[2],
#             #                                                                        odefunc.ff_weights_AC[3]))
#             # print('     FF weights col B to C: {:.4f}, {:.4f}'.format(odefunc.ff_weights_BC[2],
#             #                                                                        odefunc.ff_weights_BC[3]))
#
#     return accuracy_sigmoid, accuracy_binary
#
#
# if __name__ == '__main__':
#     nr_samples = 80
#     nr_test_samples = 4
#     batch_size = 4
#
#     acc1, acc2 = train_xor_ode(nr_samples, nr_test_samples, batch_size)