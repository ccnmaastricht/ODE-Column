import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from src.coupled_columns import CoupledColumns
from src.xor_columns import ColumnsXOR
from src.utils import load_config, create_feedforward_input


def vis_xor_results(firing_rates, stim, train_loss, iter1, iter2):
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if stim[2] != stim[10]:
        message = "diff input - XOR"
    else:
        message = "same input - AND"

    fig.text(0.5, 0.03, f"Input: {message}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Final FR: {firing_rates[-1, 16]:.2f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.2, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')

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

def make_stim(layer_4_indices):
    stims = torch.Tensor(4, 16)
    conditions = torch.tensor([[0., 0.],
                               [1., 0.],
                               [0., 1.],
                               [1., 1.]])
    for i in range(4):
        rand_fr = conditions[i] * torch.empty(1).uniform_(0.875, 1.125)
        stim = create_feedforward_input(16, rand_fr[0], rand_fr[1])
        stims[i, :] = torch.tensor(stim)

    # Shuffle along the first dimension
    stims_shuffled = stims[torch.randperm(stims.size(0))]
    return stims_shuffled

def min_max(x):
    max_val = torch.max(x)
    min_val = torch.min(x)
    return (x - min_val) / (max_val - min_val)

def run_four_xor_samples(odefunc, initial_state, time_vec, time_steps, layer_4_indices, batch_size=4, mode="training"):
    batch_output = torch.Tensor(batch_size, time_steps, 2, 24)
    stim_batch = torch.Tensor(batch_size, 16)

    for batch_iter in range(int(batch_size / 4)):
        four_stims = make_stim(layer_4_indices)

        for stim_iter, stim in enumerate(four_stims):
            itr = (batch_iter * 4) + stim_iter
            stim_batch[itr, :] = stim

            ode_output = odefunc.run_ode_xor(initial_state, stim, time_vec, num_stim_phases=2)
            batch_output[itr, :, :, :] = ode_output

    # Train on xor classification
    # Compute firing rates
    firing_rates = odefunc.compute_firing_rate_torch(batch_output[:, :, 0, :] - batch_output[:, :, 1, :])

    # for i in range(batch_size):
    #     print(stim_batch[i])
    #     plt.plot(firing_rates[i, :, 0].detach().numpy())
    #     plt.plot(firing_rates[i, :, 8].detach().numpy())
    #     plt.plot(firing_rates[i, :, 16].detach().numpy())
    #     plt.show()

    # pass final firing rate to loss functions
    final_fr_C = firing_rates[:, -1, 16]  # final firing rates column C
    xor_output = min_max(final_fr_C)
    # Compute loss with binary targets
    xor_targets = (stim_batch[:, 2] != stim_batch[:, 10]).int()
    loss = torch.mean(abs(xor_output - xor_targets))

    if mode == "training":
        return xor_output, loss
    elif mode == "testing":
        return firing_rates, stim_batch, xor_targets


def train_xor_ode(nr_samples, batch_size):
    '''
    Train an ODE to solve the XOR problem
    '''

    # Initialize the ODE function
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')
    odefunc = ColumnsXOR(col_params, 'MT')

    layer_4_indices = col_params['layer_4_indices']

    # Initial state is always the same
    membrane = torch.tensor(sim_params['initial_conditions']['membrane_potential'])
    adaptation = torch.tensor(sim_params['initial_conditions']['adaptation'])
    initial_state = torch.stack((membrane, adaptation))
    initial_state = torch.cat((initial_state, initial_state[:, :8]), dim=-1)  # extent to three columns

    # Initialize the optimizer and add weights as learnable parameters
    optimizer = torch.optim.RMSprop([odefunc.ff_weights_1,
                                     odefunc.ff_weights_2,
                                     odefunc.ff_weights_AC,
                                     odefunc.ff_weights_BC,
                                     odefunc.connection_weights
                                     ], lr=1e-4, alpha=0.95)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # higher gamma = slower decay

    nr_batches = int(nr_samples/batch_size)
    time_steps = int(sim_params['protocol']['stimulus_duration'] * 2 / sim_params['time_step'])
    time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)

    accuracy = []

    for itr in range(nr_batches):
        optimizer.zero_grad()

        xor_output, loss = run_four_xor_samples(odefunc, initial_state, time_vec, time_steps,
                                                layer_4_indices, batch_size=batch_size, mode="training")
        loss.backward()

        # print("Gradient weights input 1: ", torch.norm(odefunc.ff_weights_1.grad))
        # print("Gradient weights input 1: ", torch.norm(odefunc.ff_weights_2.grad))
        # print("Gradient weights A-C: ", torch.norm(odefunc.ff_weights_AC.grad))
        # print("Gradient weights B-C: ", torch.norm(odefunc.ff_weights_BC.grad))
        # print("Gradient connection weights: ", torch.norm(odefunc.connection_weights.grad))

        with torch.no_grad():  # make sure not to update illegal connections
            odefunc.ff_weights_1.grad *= odefunc.ff_weights_mask
            odefunc.ff_weights_2.grad *= odefunc.ff_weights_mask
            odefunc.ff_weights_AC.grad *= odefunc.ff_weights_mask[:8]
            odefunc.ff_weights_BC.grad *= odefunc.ff_weights_mask[:8]
            odefunc.connection_weights.grad[:16, :16] *= odefunc.lat_mask

        optimizer.step()
        scheduler.step()  # adjust learning rate

        print('Iter {:02d} | Total Loss {:.5f}'.format(itr + 1, loss.item()))
        # Test ODE model and visualize results
        with torch.no_grad():
            fr_rates, stims, _ = run_four_xor_samples(odefunc, initial_state, time_vec, time_steps,
                                                   layer_4_indices, batch_size=4, mode="testing")
            for test_itr in range(4):
                vis_xor_results(fr_rates[test_itr], stims[test_itr], loss.item(), itr, test_itr)

            # Visualizing accuracy results
            # with torch.no_grad():
            #     final_fr_rates = []
            #     xor_targets = []
            #     for i in range(25):
            #         fr_rates, stims, targets = run_four_xor_samples(odefunc, initial_state, time_vec, time_steps,
            #                                                layer_4_indices, batch_size=4, mode="testing")
            #         for j in range(4):
            #             final_fr_rates.append(fr_rates[j, -1, 16].item())
            #             xor_targets.append(targets[j].item())
            #     # Define a threshold as the mean final firing rate
            #     threshold = np.mean(final_fr_rates)
            #     xor_preds = [1 if x > threshold else 0 for x in final_fr_rates]
            #     # Determine confusion matrix based on threshold
            #     hits = [1 if xor_preds[k] == xor_targets[k] and xor_targets[k] == 1 else 0 for k in range(100)]
            #     rejections = [1 if xor_preds[k] == xor_targets[k] and xor_targets[k] == 0 else 0 for k in range(100)]
            #     misses = [1 if xor_preds[k] != xor_targets[k] and xor_targets[k] == 1 else 0 for k in range(100)]
            #     false_alarms = [1 if xor_preds[k] != xor_targets[k] and xor_targets[k] == 0 else 0 for k in range(100)]
            #     print(threshold)
            #
            #     print(np.sum(hits))
            #     print(np.sum(rejections))
            #     print(np.sum(misses))
            #     print(np.sum(false_alarms))
            #     accuracy.append((np.sum(hits) + np.sum(rejections)).item() / 100.)
            #     print(accuracy)

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
        # print('     Lat weights between A and B: {:.4f}, {:.4f}'.format(odefunc.connection_weights[1, 8],
        #                                                                 odefunc.connection_weights[9, 0]))

    return accuracy


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
    return classifier


if __name__ == '__main__':
    nr_samples = 100
    batch_size = 4

    acc = train_xor_ode(nr_samples, batch_size)

    # acc = [0.5, 0.5, 0.5, 0.5, 0.5, 0.87, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    print(acc)
    fig = plt.plot(acc)
    plt.xlabel('Number of batches (batch size=4)')
    plt.ylabel('Accuracy')
    plt.savefig('../results/acc_xor')
