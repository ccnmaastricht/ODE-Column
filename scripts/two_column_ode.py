import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.coupled_columns import CoupledColumns, ColumnODEFunc
from src.utils import load_config

from torchdiffeq import odeint



def make_ds_dmf(ds_file, nr_samples):
    if os.path.exists(ds_file):
        with open(ds_file, 'rb') as f:
            ds = pickle.load(f)
    else:
        # Initialize two columns
        col_params = load_config('../config/model.toml')
        sim_params = load_config('../config/simulation.toml')
        columns = CoupledColumns(col_params, 'MT')

        # Make ds dict with two tensors states and stims
        protocol, dt = sim_params['protocol'], sim_params['time_step']
        time_steps = (protocol['pre_stimulus_period'] + protocol['post_stimulus_period'] + protocol['stimulus_duration']) / dt
        ds = {
            'states': torch.Tensor(nr_samples, int(time_steps), 3, columns.num_populations),
            'stims': torch.Tensor(nr_samples, columns.num_populations)
        }

        # Generate training data and store in ds dict
        for i in range(nr_samples):
            state, stim = columns.run_single_sim(sim_params, col_params, rand_input=True)
            ds['states'][i, :, :, :] = torch.tensor(state)   # membrane, adaptation and firing rate
            ds['stims'][i, :] = torch.tensor(stim)  # stimulus input used

        # Save ds dict as pickle
        with open(ds_file, 'wb') as f:
            pickle.dump(ds, f)
    return ds['states'], ds['stims']


if __name__ == '__main__':

    nr_samples = 10
    batch_size = 2

    # Get dataset
    states, stims = make_ds_dmf('../pickled_ds/test_ds.pkl', nr_samples)

    # Prepare train and test sets
    split = int(nr_samples * 0.9)
    train_states, test_states = states[:split, :, :2, :], states[split:, :, :2, :]  # lose the firing rate
    train_stims, test_stims = stims[:split, :], stims[split:, :]

    train_dataset = TensorDataset(train_states, train_stims)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the ODE function
    col_params = load_config('../config/model.toml')
    odefunc = ColumnODEFunc(col_params, 'MT')

    # Initialize the optimizer and add connection weights as learnable parameter
    optimizer = torch.optim.RMSprop([odefunc.connection_weights], lr=0.001, alpha=0.99)

    for iter, (state_batch, stim_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        nr_batch_samples = len(state_batch)
        time_steps = len(state_batch[0])
        time_vec = torch.linspace(0., time_steps * 1e-4, time_steps)

        results_nn = torch.Tensor(nr_batch_samples, time_steps, 2, odefunc.num_populations)

        for batch_iter in range(nr_batch_samples):
            stim = stim_batch[batch_iter]
            input_state = state_batch[batch_iter, 0, :, :]

            output_state = odeint(lambda t, y: odefunc.dynamics_ode(t, input_state, stim), input_state, time_vec)

            results_nn[batch_iter, :, :, :] = output_state

            # for t in range(time_steps):
            #     output_state = odefunc.dynamics_ode(t, input_state, stim)
            #
            #     results_nn[batch_iter, t, :, :] = output_state
            #     input_state = output_state  # next input for nn is its current output

            plt.plot(results_nn[batch_iter, :, 0, 0].detach().numpy())
            plt.plot(state_batch[batch_iter, :, 0, 0].detach().numpy())
            plt.show()

