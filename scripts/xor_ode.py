import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import torch

from src.coupled_columns import *
from src.utils import *
from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint



class ColumnNetworkXOR(torch.nn.Module):

    '''
    Concatenates a number of areas (each consisting of a number
    of columns) to form a larger network. Within an area, only
    lateral connections between columns are allowed. Across areas
    only feedforward- and feedback connections are allowed.
    '''

    def __init__(self, column_parameters, network_dict):
        super().__init__()

        self.noise_type = "diagonal"  # sde params
        self.sde_type = "ito"

        self._initialize_areas(column_parameters, network_dict)

        self.network_as_area = ColumnArea(column_parameters, 'mt', sum(network_dict['nr_columns_per_area']))
        self.nr_input_units = network_dict['nr_input_units']
        self.nr_columns_per_area = network_dict['nr_columns_per_area']

        self._initialize_lateral_weights()
        self._initialize_ff_fb_masks()
        self._initialize_feedforward_weights()

    def _initialize_areas(self, column_parameters, network_dict):
        self.areas = nn.ModuleDict({})
        for area_idx in range(network_dict['nr_areas']):

            area_name = network_dict['areas'][area_idx]
            num_columns = network_dict['nr_columns_per_area'][area_idx]

            area = ColumnArea(column_parameters, area_name, num_columns)
            self.areas[str(area_idx)] = area

    def _initialize_lateral_weights(self):
        '''
        Sets external recurrent weights of all areas to zero, to make sure
        all lateral connectivity is removed.
        '''
        for idx, area in self.areas.items():
            recurr_weights = area.recurrent_weights
            area.recurrent_weights = recurr_weights * area.internal_mask  # set any existing external connectivity to zero

    def _initialize_ff_fb_masks(self):
        '''
        Specify from which population the feedforward flow comes
        (source) and which population it targets (target).
        '''
        # Source of ff is L2/3e (and L5e)
        ff_source_mask = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        self.ff_source_mask = ff_source_mask

        # Target of ff is L4e and L4i
        ff_target_mask = torch.tensor([0., 0., 1., 1., 0., 0., 0., 0.])
        self.ff_target_mask = ff_target_mask

    def _initialize_feedforward_weights(self):
        '''
        Initialize the feedforward weights as learnable weights. Both target
        and source weights are initialized.
        '''
        feedforward_target_weights = nn.ModuleDict({})

        feedforward_target_weights['0'] = nn.ParameterList()
        feedforward_target_weights['0'].append(torch.tensor([0.0000, 0.0000, 0.0140, 0.0036, 0.0000, 0.0000, 0.0000, 0.0000,
                                                             0.0000, 0.0000, 0.0131, 0.0066, 0.0000, 0.0000, 0.0000, 0.0000]))
        feedforward_target_weights['0'].append(torch.tensor([0.0000, 0.0000, 0.0162, 0.0063, 0.0000, 0.0000, 0.0000, 0.0000,
                                                             0.0000, 0.0000, 0.0046, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]))
        feedforward_target_weights['1'] = nn.ParameterList()
        feedforward_target_weights['1'].append(torch.tensor([0.0000, 0.0000, 0.0157, 0.0044, 0.0000, 0.0000, 0.0000, 0.0000]))
        feedforward_target_weights['1'].append(torch.tensor([0.0000, 0.0000, 0.0116, 0.0094, 0.0000, 0.0000, 0.0000, 0.0000]))

        # feedforward_target_weights['0'] = nn.ParameterList()
        # feedforward_target_weights['0'].append(torch.tensor([0.0000, 0.0000, 0.0138, 0.0072, 0.0000, 0.0000, 0.0000, 0.0000,
        #                                                      0.0000, 0.0000, 0.0151, 0.0066, 0.0000, 0.0000, 0.0000, 0.0000]))
        # feedforward_target_weights['0'].append(torch.tensor([0.0000, 0.0000, 0.0143, 0.0070, 0.0000, 0.0000, 0.0000, 0.0000,
        #                                                      0.0000, 0.0000, 0.0148, 0.0064, 0.0000, 0.0000, 0.0000, 0.0000]))
        # feedforward_target_weights['1'] = nn.ParameterList()
        # # feedforward_target_weights['1'].append(torch.tensor([0.0000, 0.0000, 0.0135, 0.0075, 0.0000, 0.0000, 0.0000, 0.0000]))
        # # feedforward_target_weights['1'].append(torch.tensor([0.0000, 0.0000, 0.0147, 0.0072, 0.0000, 0.0000, 0.0000, 0.0000]))
        # feedforward_target_weights['1'].append(torch.tensor([0.0000, 0.0000, 0.0131, 0.0079, 0.0000, 0.0000, 0.0000, 0.0000]))
        # feedforward_target_weights['1'].append(torch.tensor([0.0000, 0.0000, 0.0143, 0.0076, 0.0000, 0.0000, 0.0000, 0.0000]))
        #
        # # feedforward_target_weights['0'] = nn.ParameterList()
        # # feedforward_target_weights['0'].append(torch.tensor([0.0000, 0.0000, 0.0142, 0.0068, 0.0000, 0.0000, 0.0000, 0.0000,
        # #                                                      0.0000, 0.0000, 0.0155, 0.0062, 0.0000, 0.0000, 0.0000, 0.0000]))
        # # feedforward_target_weights['0'].append(torch.tensor([0.0000, 0.0000, 0.0147, 0.0066, 0.0000, 0.0000, 0.0000, 0.0000,
        # #                                                      0.0000, 0.0000, 0.0152, 0.0060, 0.0000, 0.0000, 0.0000, 0.0000]))
        # # feedforward_target_weights['1'] = nn.ParameterList()
        # # feedforward_target_weights['1'].append(torch.tensor([0.0000, 0.0000, 0.0135, 0.0075, 0.0000, 0.0000, 0.0000, 0.0000]))
        # # feedforward_target_weights['1'].append(torch.tensor([0.0000, 0.0000, 0.0147, 0.0072, 0.0000, 0.0000, 0.0000, 0.0000]))

        # # CCN paper weights
        # feedforward_target_weights['0'] = nn.ParameterList()
        # feedforward_target_weights['0'].append(torch.tensor([0., 0., 295., 105., 0., 0., 0., 0.,
        #                                                      0., 0., 285., 160., 0., 0., 0., 0.]) / 22692)
        # feedforward_target_weights['0'].append(torch.tensor([0., 0., 350., 160., 0., 0., 0., 0.,
        #                                                      0., 0., 230., 125., 0., 0., 0., 0.]) / 22692)
        # feedforward_target_weights['1'] = nn.ParameterList()
        # feedforward_target_weights['1'].append(torch.tensor([0., 0., 335., 120., 0., 0., 0., 0.]) / 22692)
        # feedforward_target_weights['1'].append(torch.tensor([0., 0., 295., 180., 0., 0., 0., 0.]) / 22692)

        # for area_idx, area in self.areas.items():
        #
        #     feedforward_target_weights[area_idx] = nn.ParameterList()
        #
        #     if area_idx == '0':   # if first area, check how many external inputs it receives
        #         nr_ff_weights = self.nr_input_units
        #     else:               # for subsequent areas, check how many inputs from previous area
        #         key_prev_area = str(int(area_idx)-1)
        #         nr_ff_weights = self.areas[key_prev_area].num_columns
        #
        #     # Weight initialization should not be too small, otherwise no ff flow and no training possible
        #     original_target_weights = area.feedforward_weights.clone().detach() * area.synapse_time_constant
        #     std_W = area.synapse_time_constant
        #
        #     for i in range(nr_ff_weights):
        #
        #         rand_weights_target = abs(torch.normal(mean=original_target_weights, std=std_W))
        #         rand_weights_target = rand_weights_target * torch.tile(self.ff_target_mask, (area.num_columns,))
        #         ff_weights_target = nn.Parameter(rand_weights_target, requires_grad=True)
        #         feedforward_target_weights[area_idx].append(ff_weights_target)

        self.feedforward_target_weights = feedforward_target_weights

    def set_time_vec(self, time_vec):
        '''
        Set the time_vec as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.time_vec = time_vec

    def set_stim(self, stim):
        '''
        Set the stimulus as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.stim = stim

    def partition_firing_rates(self, firing_rate):
        '''
        Organizes the firing rates into a dict of separate areas.
        This allows easy access to previous area's firing rates.
        '''
        fr_per_area = {}
        idx = 0
        for area_idx, area in self.areas.items():
            fr_area = firing_rate[idx : idx + area.num_populations]
            fr_area_reshape = fr_area.reshape(area.num_columns, 8)
            fr_per_area[area_idx] = fr_area_reshape
            idx = idx + area.num_populations
        return fr_per_area

    def compute_currents(self, ext_ff_rate, fr_per_area, t):
        '''
        Compute the current for each area separately. The total current
        consists of feedforward current (stimulus-driven and/or from other
        brain areas), background current and recurrent current.
        '''
        total_current = torch.Tensor()

        for area_idx, area in self.areas.items():

            # Compute feedforward current of each area, based on
            # area=0: external input or area>0: the previous area's firing rate
            feedforward_current = torch.zeros(area.num_populations)

            for ff_idx, ff_target_weight in enumerate(self.feedforward_target_weights[area_idx]):
            # Multiply each input with each corresponding ff weights
                if area_idx == '0':  # first area gets external input
                    feedforward_current += ext_ff_rate[ff_idx] * ff_target_weight

                elif area_idx > '0':  # subsequent areas receive previous area's firing rate
                    key_prev_area = str(int(area_idx) - 1)
                    prev_area_fr = fr_per_area[key_prev_area][ff_idx] * self.ff_source_mask
                    prev_area_fr_sum = torch.sum(prev_area_fr)
                    prev_area_fr_sum *= 10.  # pump up firing rates
                    feedforward_current += prev_area_fr_sum * ff_target_weight

            # Background and recurrent current
            background_current = area.background_weights * area.background_drive
            recurrent_current = torch.matmul(area.recurrent_weights, fr_per_area[area_idx].flatten())

            # Total current of this area
            # Notice that ff an fb are not scaled down by synapse time constant bc they are already scaled down for training
            total_current_area = feedforward_current + (background_current + recurrent_current) * area.synapse_time_constant

            total_current = torch.cat((total_current, total_current_area), dim=0)
        return total_current

    def forward(self, t, state):
        '''
        State dynamics updating the membrane potential and adaptation;
        ODE should learn these dynamics and update the weights accordingly.
        '''

        # Prepare the state (membrane, adaptation, firing rate)
        state = state.squeeze(0)  # lose extra dim
        mem_adap_split = len(state) // 3
        adap_rate_split = len(state) // 3 * 2
        membrane_potential, adaptation = state[:mem_adap_split], state[mem_adap_split:adap_rate_split]

        firing_rate = compute_firing_rate_torch(membrane_potential - adaptation)

        # Partition firing rate per area
        fr_per_area = self.partition_firing_rates(firing_rate)

        # Get current stimulus (external ff rate) based on current time t and the time vector time_vec
        ext_ff_rate = torch_interp(t, self.time_vec, self.stim)
        ext_ff_rate = ext_ff_rate * 20.  # input in 1Hz range, so scale up

        # Compute input current
        total_current = self.compute_currents(ext_ff_rate, fr_per_area, t)

        # Compute derivative membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
            total_current * self.network_as_area.resistance) / self.network_as_area.membrane_time_constant
        delta_adaptation = (-adaptation + self.network_as_area.adaptation_strength *
                            firing_rate) / self.network_as_area.adapt_time_constant

        # Compute derivative firing rate
        prev_firing_rate = state[adap_rate_split:]
        delta_firing_rate = (-prev_firing_rate + firing_rate) / self.network_as_area.synapse_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation, delta_firing_rate))

        return state.unsqueeze(0)

    def diffusion(self, t, y):
        noise_std = 3.0
        g = torch.zeros_like(y)
        split = (len(y[0]) // 3)
        g[:, :split] = noise_std  # membrane gets noise
        return g



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

def vis_xor_results_layer5(firing_rates, network, stim, train_loss, iter1, iter2):
    '''
    Visualizes the training process of the XOR classification task. Plots the
    firing rates of column A, B and C and reports the training loss, the input
    condition (XOR or AND) and the final firing rate of column C, what determines
    the network's output. Includes layer 5 in addition to layer 2/3
    Images are saved in ../results/png.
    '''

    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    if stim[2] != stim[10]:
        condition = "diff input - XOR"
    else:
        condition = "same input - AND"

    axes[0].plot(firing_rates[:, 0], label='col A')
    axes[0].plot(firing_rates[:, 8], label='col B')
    # axes[0].set_ylim(0.0, 3.0)
    axes[0].set_title("Firing rates L2/3e in column A and B")

    axes[1].plot(firing_rates[:, 4], label='col A')
    axes[1].plot(firing_rates[:, 12], label='col B')
    # axes[1].set_ylim(0.0, 50.0)
    axes[1].set_title("Firing rates L5e in column A and B")

    axes[2].plot(firing_rates[:, 16], label='col C L23')
    axes[2].plot(firing_rates[:, 20], label='col C L5')
    columnC_weighted = torch.sum(firing_rates[:, 16:] * network.ff_source_mask, dim=1)
    axes[2].plot(columnC_weighted, label='col C mean')
    # axes[2].set_ylim(0.0, 50.0)
    axes[2].set_title("Firing rates in column C")

    fig.text(0.2, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.5, 0.03, f"Input: {condition}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Final FR: {columnC_weighted[-1]:.2f}", ha='center', fontsize=10, fontweight='bold')

    fig.legend(loc="upper left")
    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig('../results/png/{:02d}_{:1d}'.format(iter1+1, iter2))
    plt.close(fig)

def make_stim(shuffle=True):
    '''
    Creates a set of 4 stimuli, one for each XOR condition (0,0), (0,1),
    (1,0), (1,1). 1s are randomly set between 0.975 and 1.025.
    '''

    stims = torch.Tensor(4, 16)
    conditions = torch.tensor([[1., 0.],
                               [0., 1.],
                               [1., 1.],
                               [0., 0.]])
    for i in range(4):
        rand_stim = conditions[i] * torch.empty(1).uniform_(0.975, 1.025)
        stim = create_feedforward_input(16, rand_stim[0], rand_stim[1])
        stims[i, :] = stim

    # Shuffle along the first dimension
    stims_shuffled = stims[torch.randperm(stims.size(0))]
    if shuffle is True:
        return stims_shuffled
    else:
        return stims

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
            # ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'})

            batch_output[itr, :, :, :] = ode_output

    # # Compute firing rates
    # firing_rates = compute_firing_rate_torch(batch_output[:, :, :, :24] - batch_output[:, :, :, 24:48])
    # firing_rates = firing_rates[0, :, :].detach().numpy()
    #
    # # plot
    # membrane = batch_output[0, :, :, :24].detach().numpy()
    # adaptation = batch_output[0, :, :, 24:48].detach().numpy()
    # firing_rates = batch_output[0, :, :, 48:].detach().numpy()
    #
    # for i in range (24):
    #     plt.plot(membrane[:, :, i])
    #     plt.plot(adaptation[:, :, i])
    #     plt.plot(firing_rates[:, :, i])
    #     plt.show()

    firing_rates = batch_output[:, :, :, 48:].squeeze(dim=2)

    # Use final firing rate to compute model output
    final_fr_C = firing_rates[:, -1, 16:]  # final firing rates column C
    final_fr_C = torch.sum(final_fr_C * network.ff_source_mask, dim=1)
    xor_output = min_max(final_fr_C)
    # xor_output = fr_to_binary(final_fr_C)

    # Compute loss with binary targets
    xor_targets = (stim_batch[:, 2] != stim_batch[:, 10]).int()
    xor_targets = torch.tensor([1.0 if i == 1 else 0.25 for i in xor_targets])
    loss = torch.mean(abs(final_fr_C - xor_targets))

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
    network = ColumnNetworkXOR(col_params, network_input)

    # Initial state is always the same
    membrane = torch.tensor(sim_params['initial_conditions']['membrane_potential'])
    membrane = torch.tile(membrane, (3,))  # extent to three columns
    adaptation = torch.tensor(sim_params['initial_conditions']['adaptation'])
    adaptation = torch.tile(adaptation, (3,))
    rate = adaptation  # start firing rate is just zeros

    initial_state = torch.concat((membrane, adaptation, rate))
    initial_state = initial_state.unsqueeze(0)  # shape (1,72) for sde

    optimizer = torch.optim.RMSprop(network.parameters(), lr=1e-4, alpha=0.95)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  # higher gamma = slower decay

    nr_batches = int(nr_samples/batch_size)
    time_steps = int(sim_params['protocol']['stimulus_duration'] * 2 / sim_params['time_step'])
    time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)

    network.time_vec = time_vec

    # Store results
    accuracy_sigmoid = []
    accuracy_binary = []

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

        pprint(network.feedforward_target_weights['0'][0])
        pprint(network.feedforward_target_weights['0'][1])
        pprint(network.feedforward_target_weights['1'][0])
        pprint(network.feedforward_target_weights['1'][1])

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

            # # Threshold the firing rates using the mean ~ binary predictions
            # threshold = np.mean(final_fr_rates)
            # xor_preds_binary = [1 if x > threshold else 0 for x in final_fr_rates]
            #
            # # Sigmoid-classify the firing rates ~ non-binary predictions
            # xor_preds_sigmoid = fr_to_binary(torch.tensor(final_fr_rates)).numpy()
            #
            # # Compute accuracies
            # TP_sigmoid = [xor_preds_sigmoid[k] if xor_preds_sigmoid[k] > 0.5 and xor_targets[k] == 1 else 0 for k in range (nr_test_samples)]
            # TN_sigmoid = [1 - (xor_preds_sigmoid[k]) if xor_preds_sigmoid[k] < 0.5 and xor_targets[k] == 0 else 0 for k in range(nr_test_samples)]
            # TP_binary = [1 if xor_preds_binary[k] == 1 and xor_targets[k] == 1 else 0 for k in range (nr_test_samples)]
            # TN_binary = [1 if xor_preds_binary[k] == 0 and xor_targets[k] == 0 else 0 for k in range(nr_test_samples)]
            #
            # accuracy_sigmoid.append((np.sum(TP_sigmoid) + np.sum(TN_sigmoid)).item() / nr_test_samples)
            # accuracy_binary.append((np.sum(TP_binary) + np.sum(TN_binary)).item() / nr_test_samples)

    return accuracy_sigmoid, accuracy_binary


def run_xor_timecourse():

    # Initialize the network
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')

    network_input = {'nr_areas': 2, 'areas': ['mt', 'mt'], 'nr_columns_per_area': [2, 1], 'nr_input_units': 2}
    network = ColumnNetworkXOR(col_params, network_input)

    initial_state = torch.zeros(72)
    initial_state = initial_state.unsqueeze(0)  # shape (1,72) for sde

    time_steps = int(sim_params['protocol']['stimulus_duration'] * 2 / sim_params['time_step'])
    time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)

    time_course = torch.Tensor(time_steps*4, 24)  # 4 stimuli * N time steps, 24 populations
    stim_time_course = torch.Tensor(time_steps*4, 2)

    four_stims = make_stim(shuffle=False)

    with torch.no_grad():
        for stim_iter, stim in enumerate(four_stims):
            stim_ode = prep_stim_ode(stim, time_vec)

            network.time_vec = time_vec
            network.stim = stim_ode
            # ode_output = odeint(network, initial_state, time_vec)
            ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'})

            initial_state = ode_output[-1, :, :]

            # Store results
            time_course[stim_iter*time_steps:(stim_iter+1)*time_steps, :] = ode_output[:, 0, 48:]
            stim_time_course[stim_iter*time_steps:(stim_iter+1)*time_steps, 0] = stim_ode[:, 0, 2]  # idx2 = layer 4
            stim_time_course[stim_iter*time_steps:(stim_iter+1)*time_steps, 1] = stim_ode[:, 1, 2]

        # Set the figure size (wide, not too tall)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 2.5, 0.5, 0.5]})

        ax1.plot(time_course[:, 0], label='Column A', color='deepskyblue', linewidth=2)
        ax1.plot(time_course[:, 8], label='Column B', color='coral', linewidth=2)
        ax1.set_title('L2/3e firing rates in columns A & B', fontsize=14)
        ax1.set_ylabel('Firing Rate', fontsize=12)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.plot(time_course[:, 16], label='Column C', color='limegreen', linewidth=2)
        ax2.set_title('L2/3e firing rates in column C', fontsize=14)
        ax2.set_ylabel('Firing Rate', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)

        ax3.plot(stim_time_course[:, 0]*20., label='Input 1', color='black', linewidth=2)
        ax3.set_title('Input 1', fontsize=14)
        ax3.set_ylabel('Hz', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.5)

        ax4.plot(stim_time_course[:, 1]*20., label='Input 2', color='black', linewidth=2)
        ax4.set_title('Input 2', fontsize=14)
        ax4.set_xlabel('Time Step', fontsize=12)
        ax4.set_ylabel('Hz', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.5)

        # Layout adjustment
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    nr_samples = 160
    nr_test_samples = 4
    batch_size = 4

    # acc1, acc2 = train_xor_ode(nr_samples, nr_test_samples, batch_size)

    run_xor_timecourse()

