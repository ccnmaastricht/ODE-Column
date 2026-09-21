import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from src.brain_network import BrainNetwork
from src.utils.paths import config_path, models_path
from src.utils.set_seed import set_seed



def visualize_results(raw_output, network, stims, loss, iter):
    """"""
    firing_rates = network.get_firing_rates(raw_output, area='v2', population='L23e')

    for i in range(len(stims)):

        fig = plt.figure()

        plt.plot(firing_rates[:, i, 0], label='col1')
        plt.plot(firing_rates[:, i, 1], label='col2')

        fig.legend(loc="upper left")

        fig.text(0.2, 0.03, f"Loss: {loss:.2f}", ha='center', fontsize=10, fontweight='bold')
        fig.text(0.5, 0.03, f"Input: {stims[i].reshape(1, len(stims[i]))}",
                 ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')

        plt.savefig('./results/fr_{:02d}_{:1d}'.format(iter, i))
        plt.close(fig)

def make_input_pairs(min_val=20.0, max_val=35.0, min_diff=5.0, max_diff=10.0, nr_steps=100):
    """"""
    mu_values = np.linspace(min_val, max_val, nr_steps)

    input_pairs = []

    for a in mu_values:
        for b in mu_values:

            if min_diff <= abs(a - b) <= max_diff:
                input_pairs.append([a, b])

    return input_pairs

def get_data(batch_size, seed, test_fraction=0.1):
    """"""
    input_pairs = make_input_pairs()
    input_pairs = torch.tensor(input_pairs, dtype=torch.float32)
    labels = torch.argmax(input_pairs, dim=1)
    labels = torch.nn.functional.one_hot(labels, num_classes=2).float()

    generator = torch.Generator().manual_seed(seed)
    n = len(input_pairs)
    perm = torch.randperm(n, generator=generator)

    n_test = int(round(test_fraction * n))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_stims = input_pairs[train_idx]
    train_labels = labels[train_idx]
    test_stims = input_pairs[test_idx]
    test_labels = labels[test_idx]

    ds = TensorDataset(train_stims, train_labels)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_stims, test_labels

def initialize_lat_in_connection(network_params):
    """
    Initializer function for lateral inhibition weights.
    """
    init_raw = torch.tensor(network_params['model_params']['connection_inits']['lateral_inhibition'])
    mask = torch.tensor(network_params['model_params']['connection_masks']['lateral_inhibition'])

    size_area = network_params['source'].num_columns
    init = torch.tile(init_raw, (size_area, size_area))

    mask = torch.tile(mask, (size_area, size_area))
    mask *= network_params['source'].external_mask

    weights = init * mask
    return weights, mask

def initialize_self_excitation_connection(network_params):
    """
    Initializer function for self-excitation weights (within column).
    """
    init_raw = torch.tensor(network_params['model_params']['connection_inits']['self_excitation'])
    mask = torch.tensor(network_params['model_params']['connection_masks']['self_excitation'])

    size_area = network_params['source'].num_columns
    init = torch.tile(init_raw, (size_area, size_area))

    mask = torch.tile(mask, (size_area, size_area))
    mask *= network_params['source'].internal_mask

    weights = init * mask
    return weights, mask

def train_memory(
        seed,
        batch_size=32,
        lr=1e-2,
        test_freq=10,
        train_with_adjoint=False,
        train_with_noise=False,
        device=torch.device('cpu')):
    """"""
    set_seed(seed)

    # Initialize network
    config = config_path('memory_params.toml')
    network = BrainNetwork.from_toml(config)

    network.add_area('v1', 20)
    network.add_area('v2', 2)

    network.add_input_connection('v1', 2, receptive_field_size=1, stride=1, std=0.0)
    network.add_feedforward_connection('v1', 'v2', std=0.0)
    network.add_lateral_connection('v1', std=0.0)
    network.add_custom_connection(connection_name='lateral_inhibition', source='v2', target='v2',
                                  initializer=initialize_lat_in_connection, trainable=False)
    network.add_custom_connection(connection_name='self_excitation', source='v2', target='v2',
                                  initializer=initialize_self_excitation_connection, trainable=False)
    network.add_output_connection('v2')


    # Dataset, optimizer and loss function
    trainloader, test_stims, test_labels = get_data(batch_size, seed)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.MSELoss()


    # Save history
    history = {
        'train_losses': [],
        'test_losses': []}


    # Training loop
    for itr, (train_stims, train_labels) in enumerate(trainloader):
        optimizer.zero_grad()

        output = network.run(train_stims, adjoint=train_with_adjoint,
                             stochastic=train_with_noise, device=device)

        model_activations = network.read_out(output, mode='classification')
        loss = criterion(model_activations, train_labels)

        # print(train_stims)
        # print(train_labels)
        # print(model_activations)
        # print(loss.item())
        # network.analysis.plot_firing_rates(output, area='v2', population='L23e')

        loss.backward()
        optimizer.step()

        print('Train Loss {:.4f}'.format(loss.item()))
        history['train_losses'].append(loss.item())

        # Test
        if itr % test_freq == 0:
            with torch.no_grad():

                output = network.run(test_stims, adjoint=train_with_adjoint,
                                     stochastic=train_with_noise, device=device)

                model_activations = network.read_out(output, mode='classification')
                test_loss = criterion(model_activations, test_labels)

                visualize_results(output, network, test_stims, test_loss, itr)

                # network.analysis.visualize_weights()
                # if itr > 2:
                #     network.analysis.plot_firing_rates(output, area='v1')

                print('Test Loss {:.4f}'.format(test_loss.item()))
                history['test_losses'].append(test_loss.item())

    torch.save(history, models_path('memory', f'memory_history_{seed}.pt'))
    network.save(models_path('memory', f'memory_{seed}.pt'))




if __name__ == '__main__':

    seed = 1

    train_memory(seed)

    # TODO: might need to set method to euler to avoid ODE failures
