import torch
import numpy as np
import os
import pickle

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from src.brain_network import BrainNetwork
from src.ww_model import DM
from src.utils.loss_functions import huber_loss_wta
from src.utils.set_seed import set_seed



def make_input_pairs(test_stride=10):
    """ Creates a grid of input pairs (A, B) such that each
    input rate is always between 15.0 and 45.0 and the two have
    a difference of at least 5.0 and at most 10.0."""
    mu_values = np.linspace(15, 45, 100)

    train_pairs = []
    test_pairs = []

    for i, a in enumerate(mu_values):
        for j, b in enumerate(mu_values):

            if not (5 <= abs(a - b) <= 10):
                continue

            if (i + j) % test_stride == 0:
                test_pairs.append([a, b])
            else:
                train_pairs.append([a, b])

    return train_pairs, test_pairs

def make_ds_ww(ds_file, mode, input_pairs, time_steps):
    """ Make Wang-Wong dataset from provided input pairs."""

    if not os.path.exists('../data'):
        os.makedirs('../data')

    # Caching depends on nr of input_pairs
    cache_key = ds_file.replace('.pkl', f'_{mode}.pkl')

    if os.path.exists(cache_key):
        with open(cache_key, 'rb') as f:
            ds = pickle.load(f)

    else:
        nr_samples = len(input_pairs)

        ds = {
            'states': torch.Tensor(nr_samples, time_steps, 2),
            'stims': torch.Tensor(nr_samples, 2)
        }

        dm = DM()  # Wang-Wong model

        for i in range(nr_samples):
            muA, muB = input_pairs[i]

            R = dm.run_sim(muA, muB)
            R = R[:, ::10]
            R = R[:, :time_steps]

            ds['states'][i] = torch.tensor(R).transpose(0, 1)
            ds['stims'][i] = torch.tensor([muA, muB])

        with open(cache_key, 'wb') as f:
            pickle.dump(ds, f)

    return ds['states'], ds['stims']

def get_data(batch_size, network_time_params, fn):
    """ Gets the training dataset made with Wang-Wong model,
    scales it down to match our L23e firing rates."""
    # Determine number of time steps for the target data
    dt = network_time_params['dt']
    sim_time = network_time_params['sim_time']
    time_steps = int(round(sim_time / dt))

    train_pairs, test_pairs = make_input_pairs()

    train_states, train_stims = make_ds_ww(fn, 'train', train_pairs, time_steps)
    test_states, test_stims = make_ds_ww(fn, 'test', test_pairs, time_steps)

    # Scale down Wang-Wong firing rates to match with our L23e
    train_states = train_states / 30.
    test_states = test_states / 30.

    ds = TensorDataset(train_states, train_stims)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_states, test_stims

def initialize_self_excitation_connection(network_params):
    """
    Initializer function for self-excitation weights (within column).
    """
    init = torch.tensor(network_params['model_params']['connection_inits']['self_excitation'])
    mask = torch.tensor(network_params['model_params']['connection_masks']['self_excitation'])

    size_area = network_params['source'].num_columns

    init = torch.tile(init, (size_area, size_area))
    init_weights = abs(torch.normal(mean=init, std=0.1))

    mask = torch.tile(mask, (size_area, size_area))
    mask *= network_params['source'].internal_mask

    weights = init_weights * mask
    return weights, mask



if __name__ == '__main__':


    # Hyper params
    batch_size          = 32
    num_epochs          = 3
    test_freq           = 10
    train_with_adjoint  = False
    train_with_noise    = True
    seed                = 1
    fn_target_data      = '../../data/ds_wta.pkl'
    device              = torch.device('cpu')

    set_seed(seed)


    # Build network
    config_path = '../../config/wta_params.toml'
    network = BrainNetwork.from_toml(config_path)

    network.add_area('mt', 2)

    network.add_input_connection('mt', 2, receptive_field_size=1, stride=1, trainable=False, std=0.0)
    network.add_lateral_connection('mt', trainable=True)
    network.add_output_connection('mt', trainable=False)

    network.add_custom_connection(connection_name='self_excitation',
                                  source='mt',
                                  target='mt',
                                  initializer=initialize_self_excitation_connection)


    # Prepare train and test data, and optimizer
    train_loader, test_states, test_stims = get_data(batch_size, network.params['model']['time_params'], fn_target_data)
    test_states = test_states.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=10.0) # torch.optim.RMSprop(network.parameters(), lr=10.0, alpha=0.9)

    # Store losses
    train_losses = []
    test_losses = []

    # Start training loop
    for epoch in range(num_epochs):

        train_loss_sum = 0.0

        for itr, (true_states, stim_batch) in enumerate(train_loader):

            optimizer.zero_grad()
            true_states = true_states.to(device)

            output = network.run(stim_batch, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)

            # Compute loss between predicted and true states
            pred_states = network.read_out(output, mode='trajectory')
            loss = huber_loss_wta(pred_states, true_states)

            loss.backward()
            optimizer.step()

            train_loss_sum = train_loss_sum / len(train_loader)

            # Test
            if itr % test_freq == 0:
                with torch.no_grad():

                    output = network.run(test_stims, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)
                    pred_states = network.read_out(output, mode='trajectory')
                    test_loss = huber_loss_wta(pred_states, test_states)

                    print('Epoch {:02d} | Iter {:02d} | Train Loss {:.4f} | Test Loss {:.4f}'.format(epoch, itr//10, loss.item(), test_loss.item()))

                    train_losses.append(loss.item())
                    test_losses.append(test_loss.item())


    # Store training history and trained network
    history = {'train_losses': train_losses,
               'test_losses': test_losses}

    torch.save(history, 'wta_history.pt')
    network.save('wta.pt')


    # weights = network.connections['recurrent_mt_mt'].weights + network.connections['lateral_mt_mt'].weights + network.connections[
    #     'self_excitation_mt_mt'].weights
    #
    # plt.imshow(weights.detach().numpy(), cmap="viridis", interpolation="nearest")
    # plt.show()

    # firing_rates = network.get_firing_rates(output)
    # # network.analysis.plot_firing_rates(firing_rates)
    # for i in range(len(test_stims)):
    #     print(test_stims[i])
    #     plt.plot(test_states[i, :, 0], linestyle='--', label='true_1')
    #     plt.plot(test_states[i, :, 1], linestyle='--', label='true_2')
    #     plt.plot(firing_rates[:, i, 0], label='pred_1')
    #     plt.plot(firing_rates[:, i, 8], label='pred_2')
    #     plt.legend()
    #     plt.show()
