import torch
import numpy as np
import os

from torch.utils.data import TensorDataset, DataLoader

from src.brain_network import BrainNetwork
from src.ww_model import DM
from src.utils.paths import config_path, data_path, models_path
from src.utils.loss_functions import huber_loss_wta
from src.utils.set_seed import set_seed



def make_input_pairs(test_stride=10):
    """
    Creates a grid of input pairs (A, B) such that each
    input rate is always between 15.0 and 45.0 and the two have
    a difference of at least 5.0 and at most 10.0.
    """
    mu_values = np.linspace(15, 45, 100)

    input_pairs = []

    for a in mu_values:
        for b in mu_values:

            if 5 <= abs(a - b) <= 10:
                input_pairs.append([a, b])

    return input_pairs

def make_ds_ww(ds_file, time_steps):
    """
    Generate or load the complete Wang-Wong dataset.
    """
    if os.path.exists(ds_file):
        ds = torch.load(ds_file, weights_only=False)

    else:
        print("Generating Wang-Wong dataset...")

        input_pairs = make_input_pairs()
        nr_samples = len(input_pairs)

        ds = {
            'states': torch.empty(nr_samples, time_steps, 2),
            'stims': torch.empty(nr_samples, 2)
        }

        dm = DM()

        for i, (muA, muB) in enumerate(input_pairs):
            R = dm.run_sim(muA, muB)
            R = R[:, ::10]
            R = R[:, :time_steps]

            ds['states'][i] = torch.tensor(R).transpose(0, 1)
            ds['stims'][i] = torch.tensor([muA, muB])

        torch.save(ds, ds_file)

    return ds['states'], ds['stims']

def split_dataset(states, stims, seed, test_fraction=0.1):
    """
    Randomly partition the dataset into train and test sets.
    """
    generator = torch.Generator().manual_seed(seed)

    n = len(states)
    perm = torch.randperm(n, generator=generator)

    n_test = int(round(test_fraction * n))

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return (
        states[train_idx],
        stims[train_idx],
        states[test_idx],
        stims[test_idx])

def get_data(batch_size, network_time_params, fn, seed):
    """
    Gets the Wang-Wong dataset and creates a random train/test split.
    """
    dt = network_time_params['dt']
    sim_time = network_time_params['sim_time']
    time_steps = int(round(sim_time / dt))

    states, stims = make_ds_ww(fn, time_steps)

    # Scale down Wang-Wong firing rates
    states = states / 30.

    train_states, train_stims, test_states, test_stims = split_dataset(states, stims, seed)

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

def train_wta(
        fn_target_data,
        seed,
        batch_size=32,
        num_epochs=3,
        test_freq=10,
        train_with_adjoint=True,
        train_with_noise=True,
        device=torch.device('cpu')):
    """
    Train a BrainNetwork to perform winner-take-all decision-making.
    """
    set_seed(seed)

    # Build network
    config = config_path('wta_params.toml')
    general_config = config_path('general_params_wta.toml')
    network = BrainNetwork.from_toml(config, general_config)

    network.add_area('mt', 2)

    network.add_input_connection('mt', 2, receptive_field_size=1, stride=1, trainable=False, std=0.0)
    network.add_lateral_connection('mt', trainable=True)
    network.add_output_connection('mt', trainable=False)

    network.add_custom_connection(connection_name='self_excitation',
                                  source='mt',
                                  target='mt',
                                  initializer=initialize_self_excitation_connection)

    # Prepare train and test data, and optimizer
    train_loader, test_states, test_stims = get_data(batch_size, network.params['model']['time_params'], fn_target_data, seed)
    test_states = test_states.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=10.0)

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

                    print('Epoch {:02d} | Iter {:02d} | Train Loss {:.4f} | Test Loss {:.4f}'.format(epoch, itr // test_freq, loss.item(), test_loss.item()))

                    train_losses.append(loss.item())
                    test_losses.append(test_loss.item())

    # Store training history and trained network
    history = {'train_losses': train_losses,
               'test_losses': test_losses}

    torch.save(history, models_path('wta', f'wta_history_{seed}.pt'))
    network.save(models_path('wta', f'wta_{seed}.pt'))



if __name__ == '__main__':

    fn_target_data      = data_path('ds_wta.pt')
    batch_size          = 32
    num_epochs          = 3
    test_freq           = 10
    train_with_adjoint  = False
    train_with_noise    = True
    device              = torch.device('cpu')

    for seed in range(1, 11):
        print('Seed:', seed)

        train_wta(
        fn_target_data,
        seed,
        batch_size=batch_size,
        num_epochs=num_epochs,
        test_freq=test_freq,
        train_with_adjoint=train_with_adjoint,
        train_with_noise=train_with_noise,
        device=device)
