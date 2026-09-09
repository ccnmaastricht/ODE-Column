import torch

from src.utils.set_seed import set_seed
from src.brain_network import BrainNetwork



def make_xor_ds():
    """
    Makes one mini XOR dataset with the 4 possible combinations.
    Returns the shuffled datasets and their target classification.
    """
    xor_combos = torch.tensor([[20.,  0.],
                               [ 0., 20.],
                               [20., 20.],
                               [ 0.,  0.]])
    xor_shuffled = xor_combos[torch.randperm(xor_combos.size(0))]

    xor_targets = (xor_shuffled[:, 0] != xor_shuffled[:, 1]).float()
    return xor_shuffled, xor_targets.unsqueeze(1)

def run_xor_batch(train_with_adjoint, train_with_noise, batch_size, device):
    """
    Runs multiple XOR batches through the network and computes
    the average loss over all batches.
    """
    nr_batches = batch_size//4
    total_loss = 0.0

    for _ in range(nr_batches):
        stim_batch, true_labels = make_xor_ds()
        true_labels = true_labels.to(device)

        # Run simulation
        output = network.run(
            stim_batch,
            adjoint=train_with_adjoint,
            stochastic=train_with_noise,
            device=device
        )
        model_read_out = network.read_out(output, mode='classification')

        loss = ((model_read_out - true_labels) ** 2).mean()  # mse loss
        total_loss += loss

    avg_loss = total_loss / nr_batches
    return avg_loss



if __name__ == '__main__':

    # Training params
    batch_size              = 4
    nr_epochs               = 100
    lr                      = 1.0
    train_with_adjoint      = False
    train_with_noise        = False
    seed                    = 1
    device                  = torch.device('cpu')

    set_seed(seed)

    # Building the network
    config_path = '../config/example_params.toml'
    network = BrainNetwork.from_toml(config_path)

    network.add_area(area_name='v1', size=2)
    network.add_area(area_name='v2', size=1)

    network.add_input_connection(target_area='v1', input_size=2, trainable=True, scale=0.5)
    network.add_feedforward_connection(source='v1', target='v2', trainable=True, scale=10.0)
    network.add_output_connection(source_area='v2', trainable=False)

    # Training setup
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)


    # Start training loop
    for itr in range(nr_epochs):
        optimizer.zero_grad()

        # Run simulation and compute loss
        loss = run_xor_batch(train_with_adjoint, train_with_noise, batch_size, device)

        loss.backward()
        optimizer.step()

        # Test
        with torch.no_grad():
            test_loss = run_xor_batch(train_with_adjoint, train_with_noise, 4, device)
            print('Iter {:02d} | Train Loss {:.4f} | Test Loss {:.4f}'.format(
                itr + 1, loss.item(), test_loss.item()))

