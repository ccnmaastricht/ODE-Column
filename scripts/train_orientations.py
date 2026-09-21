import torch

from src.brain_network import BrainNetwork
from src.utils.paths import config_path, models_path
from src.utils.set_seed import set_seed



def get_twelve_orientations(flatten=False):
    """
    All 12 possible orientation on a 7x7 grid.
    """
    orientations = torch.tensor([
        [[0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.]],
        [[0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 1., 1., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 1., 1., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 1.],
         [0., 0., 1., 1., 1., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0.],
         [0., 0., 1., 1., 1., 0., 0.],
         [0., 0., 0., 0., 0., 1., 1.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0.],
         [0., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 0.],
         [0., 0., 0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 0., 0., 0.]],
        [[1., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 0., 0., 1.]],
        [[0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0.]],
        [[0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0.]]
    ])

    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    if flatten:
        return torch.flatten(orientations, start_dim=1), labels

    return orientations, labels

def initialize_orientation_rfs(network_params, size_input):
    """"""
    init = torch.tensor(network_params['model_params']['connection_inits']['input'])
    mask = torch.tensor(network_params['model_params']['connection_masks']['input'])
    init = torch.transpose(init.unsqueeze(0), 0, 1)
    mask = torch.transpose(mask.unsqueeze(0), 0, 1)

    size_target_area = network_params['target'].num_columns
    mask = torch.tile(mask, (size_target_area, size_input))

    # Initialize random weights
    init *= network_params['general_params']['synaptic_strength']['baseline']
    init = torch.tile(init, (size_target_area, size_input))
    rand_weights = abs(torch.normal(mean=init, std=0.1))

    # Make orientation-specific receptive field masks
    rf_mask, _ = get_twelve_orientations(flatten=True)
    rf_mask_full = rf_mask.repeat_interleave(8, dim=0)
    mask *= rf_mask_full

    weights = rand_weights * mask
    return weights, mask

def train_orientations(
    seed,
    nr_iterations,
    lr=1e+0):
    """
    Train a network of V1 orientation columns.
    """
    set_seed(seed)

    # Initialize network
    config = config_path('orientation_params.toml')
    network = BrainNetwork.from_toml(config)

    network.add_area('v1', 12)
    network.add_custom_connection('input', source='input', target='v1',
                                  initializer=initialize_orientation_rfs, trainable=True,
                                  size_input=49)
    network.add_lateral_connection('v1')
    network.add_output_connection('v1')


    # Optimizer and loss function
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()


    # Save training history
    history = {
        'train_losses': []
    }


    # Training loop
    for itr in range(nr_iterations):
        optimizer.zero_grad()

        orientations, labels = get_twelve_orientations(flatten=True)

        output = network.run(orientations)
        model_activations = network.read_out(output, mode='classification')

        loss = criterion(model_activations, labels)

        loss.backward()
        optimizer.step()

        print('Train Loss {:.4f}'.format(loss.item()))
        history['train_losses'].append(loss.item())

        # testing ...


    # Save history and network
    torch.save(history, models_path('orientations', f'orientations_history_{seed}.pt'))
    network.save(models_path('orientations', f'orientations_{seed}.pt'))




if __name__ == '__main__':

    seed = 1
    nr_iterations = 100

    train_orientations(seed, nr_iterations)

    # 12 columns all receiving 7x7 input
    # Each column's receptive field should reflect its preferred orientation
        # Learn ff connectivity (i.e. receptive fields) or not
        # Learn lateral connectivity (i.e. Mexican hat shape) or not

    # Ch 1: orientation input V
    # Ch 2: orientation specific receptive fields V
    # Ch 3: Mexican hat-shaped lateral connectivity

    # Important: labels/loss should be circular, i.e. 0 and 11 should be close together
