import matplotlib.pyplot as plt
import os
import torch
from pprint import pprint

from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from src.coupled_columns import ColumnNetwork
from src.utils import *




def make_ds(nr_inputs, nr_samples, batch_size):
    all_combinations = torch.tensor([[(i >> bit) & 1 for bit in reversed(range(nr_inputs))]
                             for i in range(2 ** nr_inputs)], dtype=torch.float32)

    train_set = all_combinations[torch.randperm(all_combinations.size(0))][:batch_size]
    test_set = all_combinations[torch.randperm(all_combinations.size(0))][:1]

    return train_set, test_set


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
    stim_phase = all_inputs.unsqueeze(0).repeat(phase_length, 1, 1)

    empty_stim_phase = torch.zeros(stim_phase.shape)

    return torch.cat((empty_stim_phase, stim_phase), dim=0)  # (time steps, num inputs, num populations)


def init_initial_state(sim_params, num_columns):
    membrane = torch.tensor(sim_params['initial_conditions']['membrane_potential'])
    membrane = torch.tile(membrane, (num_columns,))  # extent to number of columns
    adaptation = torch.tensor(sim_params['initial_conditions']['adaptation'])
    adaptation = torch.tile(adaptation, (num_columns,))
    rate = adaptation  # start firing rate is just zeros

    initial_state = torch.concat((membrane, adaptation, rate))
    return initial_state.unsqueeze(0)


def train_parity_ode(nr_inputs, nr_samples, batch_size):

    # Initialize the network
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')

    network_input = {'nr_areas': 3,
                     'areas': ['mt', 'mt', 'mt'],
                     'nr_columns_per_area': [8, 4, 1],
                     'nr_input_units': nr_inputs}
    network = ColumnNetwork(col_params, network_input)
    num_columns = sum(network_input['nr_columns_per_area'])

    time_steps = int(sim_params['protocol']['stimulus_duration'] * 2 / sim_params['time_step'])
    time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)
    network.time_vec = time_vec

    initial_state = init_initial_state(sim_params, num_columns)

    optimizer = torch.optim.RMSprop(network.parameters(), lr=1e-5, alpha=0.95)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  # higher gamma = slower decay

    nr_batches = int(nr_samples/batch_size)

    for batch_itr in range(nr_batches):

        optimizer.zero_grad()

        train_set, test_set = make_ds(nr_inputs, nr_samples, batch_size)

        # Storing results
        batch_output = torch.Tensor(batch_size, time_steps, 1, num_columns*8*3)  # last *3 bc mem, adap and fr

        conn_weights = network.areas['0'].recurrent_weights
        ff_weights = network.feedforward_weights

        # Run neural ODE on train samples
        for itr, train_stim in enumerate(train_set):

            print(itr)

            # train_stim = torch.tensor([1., 1., 1., 1.])

            stim_ode = prep_stim_ode(train_stim, time_vec, network.areas['0'].num_columns)
            network.stim = stim_ode

            # ode_output = odeint(network, initial_state, time_vec)
            ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'}, adaptive=True)

            # print(train_stim)
            # for i in range(104):
            #     print(i)
            #     # plt.plot(ode_output[:, :, -8].detach().numpy())
            #     plt.plot(ode_output[:, :, 0+i].detach().numpy())
            #     plt.plot(ode_output[:, :, 208+i].detach().numpy())
            #     plt.show()

            batch_output[itr, :, :, :] = ode_output

        # Compute loss and update weights
        final_fr = batch_output[:, -1, 0, -8:]  # final firing rate of output column
        final_fr_summed = torch.sum(final_fr * network.ff_source_mask, dim=-1)

        parity_targets = (train_set.sum(dim=1) % 2 == 0).float()
        parity_targets = parity_targets * 30.

        pprint(final_fr_summed)
        pprint(parity_targets)

        loss = torch.mean(abs(final_fr_summed - parity_targets))
        # loss.backward()

        with torch.autograd.set_detect_anomaly(True):
            loss.backward()

        print('Iter {:02d} | Total Loss {:.5f}'.format(batch_itr + 1, loss.item()))

        for name, param in network.named_parameters():
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            if torch.norm(param.grad) > 1e4:
                print(f"Large gradient in {name}: {torch.norm(param.grad)}")

        optimizer.step()
        scheduler.step()

        # pprint(final_fr_summed)
        # pprint(train_set)
        #
        # for i in range(8):
        #     final_column = torch.sum(batch_output[i, :, 0, -8:] * network.ff_source_mask, dim=-1)
        #     plt.plot(final_column.detach().numpy())
        #     plt.plot(batch_output[i, :, 0, -8].detach().numpy())
        #     plt.plot(batch_output[i, :, 0, -4].detach().numpy())
        #     plt.show()




if __name__ == '__main__':

    nr_inputs = 4 # 16 combinations
    nr_samples = 1600
    batch_size = 4

    train_parity_ode(nr_inputs, nr_samples, batch_size)
