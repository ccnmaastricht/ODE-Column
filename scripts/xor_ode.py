import os
from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from src.coupled_columns import *
from src.utils import *

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'




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

def make_stim(shuffle=True):
    '''
    Creates a set of 4 stimuli, one for each XOR condition (0,0), (0,1),
    (1,0), (1,1).
    '''
    stims = torch.Tensor(4, 16)
    conditions = torch.tensor([[20.,  0.],
                               [ 0., 20.],
                               [20., 20.],
                               [ 0.,  0.]])
    for i in range(4):
        rand_stim = conditions[i]
        stim = torch.zeros(16)
        stim[2], stim[3]      = rand_stim[0], rand_stim[0]
        stim[10], stim[11]    = rand_stim[1], rand_stim[1]
        stims[i, :] = stim

    # Shuffle order of stims
    stims_shuffled = stims[torch.randperm(stims.size(0))]
    if shuffle is True:
        return stims_shuffled
    return stims

def prep_stim_ode(stim_raw, time_vec):
    '''
    Prepare the raw stimulus for the ODE. Outputs a tensor of length
    time steps with a pre-stimulus and a stimulus period. Contains two
    stimulus instances to input both stimuli to both input columns.
    '''
    empty_stim = torch.zeros(stim_raw.shape)

    phase_length = int(len(time_vec) / 2)
    empty_stim_phase = empty_stim.expand(phase_length, -1)
    stim_phase = stim_raw.expand(phase_length, -1)

    whole_stim_phase = torch.cat((empty_stim_phase, stim_phase), dim=0)

    # Double the stim to input it to both columns
    mirror_stim_phase = torch.cat((whole_stim_phase[:, 8:], whole_stim_phase[:, :8]), dim=1)
    return torch.stack((whole_stim_phase, mirror_stim_phase), dim=1)  # shape = (time steps, 2, num populations)

def run_four_xor_samples(network, initial_state, time_vec, time_steps, batch_size=4, mode="training"):
    '''
    Runs the network once on all four XOR conditions, using a neural ODE.
    Can be used for either training or testing the model (set mode).
    '''

    # Storing results
    batch_output = torch.Tensor(batch_size, time_steps, 1, 72)
    stim_batch = torch.Tensor(batch_size, 16)

    # Run neural ODE on four samples
    for batch_iter in range(int(batch_size / 4)):
        four_stims = make_stim()

        for stim_iter, stim in enumerate(four_stims):
            itr = (batch_iter * 4) + stim_iter
            stim_batch[itr, :] = stim
            stim_ode = prep_stim_ode(stim, time_vec)

            network.stim = stim_ode

            ode_output = odeint(network, initial_state, time_vec)
            # ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

            batch_output[itr, :, :, :] = ode_output

    # Compute firing rates
    firing_rates = compute_firing_rate(batch_output[:, :, :, :24] - batch_output[:, :, :, 24:48]).squeeze(dim=2)

    # Use final firing rate to compute model output
    final_fr_C = firing_rates[:, -1, 16:]  # final firing rates column C
    final_fr_C = torch.sum(final_fr_C * network.ff_source_mask, dim=1)
    xor_output = min_max(final_fr_C)

    # Compute loss with binary targets
    xor_targets = (stim_batch[:, 2] != stim_batch[:, 10]).int()
    xor_targets = torch.tensor([1.0 if i == 1 else 0.25 for i in xor_targets])
    loss = torch.mean(abs(final_fr_C - xor_targets))

    if mode == "training":
        return xor_output, loss
    elif mode == "testing":
        return firing_rates, stim_batch, xor_targets

def init_xor():
    '''
    Initialize the network, initial state and time variables for
    XOR training.
    '''
    # Initialize the network
    col_params = load_config('../config/model.toml')
    network_input = {'nr_areas': 2, 'areas': ['mt', 'mt'], 'nr_columns_per_area': [2, 1], 'nr_input_units': 2}
    network = ColumnNetworkXOR(col_params, network_input)

    # Initial state
    initial_state = torch.zeros(72)  # 8 * 3 * 3
    initial_state = initial_state.unsqueeze(0)  # shape (1,72) for sde

    # Time
    dt = 1e-3
    stim_duration = 0.5
    time_steps = int(stim_duration * 2 / dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)
    network.time_vec = time_vec

    return network, initial_state, time_vec, time_steps


def train_xor_ode(nr_samples, nr_test_samples, batch_size):
    '''
    Train an ODE to perform XOR classification
    '''
    network, initial_state, time_vec, time_steps = init_xor()

    optimizer = torch.optim.RMSprop(network.parameters(), lr=0.5, alpha=0.95)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  # higher gamma = slower decay

    nr_batches = nr_samples // batch_size

    for itr in range(nr_batches):
        optimizer.zero_grad()

        xor_output, loss = run_four_xor_samples(network, initial_state, time_vec, time_steps,
                                                batch_size=batch_size, mode="training")
        loss.backward()

        with torch.no_grad():  # use masks to make sure not to update illegal connections
            network.feedforward_target_weights['0'][0].grad *= torch.tile(network.ff_target_mask, (2,))
            network.feedforward_target_weights['0'][1].grad *= torch.tile(network.ff_target_mask, (2,))
            network.feedforward_target_weights['1'][0].grad *= network.ff_target_mask
            network.feedforward_target_weights['1'][1].grad *= network.ff_target_mask

        optimizer.step()
        scheduler.step()

        print('Iter {:02d} | Total Loss {:.5f}'.format(itr + 1, loss.item()))

        # Test ODE model and visualize results
        with torch.no_grad():

            for i in range(int(nr_test_samples / 4)):
                fr_rates, stims, targets = run_four_xor_samples(network, initial_state, time_vec, time_steps,
                                                       batch_size=4, mode="testing")

            for test_itr in range(4):  # Visualize last 4 test samples and save figures
                vis_xor_results(fr_rates[test_itr], stims[test_itr], loss.item(), itr, test_itr)



if __name__ == '__main__':
    nr_samples = 160
    nr_test_samples = 4
    batch_size = 4

    train_xor_ode(nr_samples, nr_test_samples, batch_size)


