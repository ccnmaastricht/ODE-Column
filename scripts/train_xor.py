import torch
import time

from src.utils.set_seed import set_seed
from src.brain_network import BrainNetwork



def make_xor_ds():
    """ Makes one mini XOR dataset with the 4 possible combinations.
     Returns the shuffled datasets and their target classification. """
    xor_combos = torch.tensor([[20.,  0.],
                               [ 0., 20.],
                               [20., 20.],
                               [ 0.,  0.]])
    xor_shuffled = xor_combos[torch.randperm(xor_combos.size(0))]

    xor_targets = (xor_shuffled[:, 0] != xor_shuffled[:, 1]).float()
    return xor_shuffled, xor_targets

def run_xor_batch(train_with_adjoint, train_with_noise, device):
    """ Runs one batch of XOR samples through the network and computes
    the loss between the model predictions and training targets. """
    stim_batch, true_labels = make_xor_ds()
    true_labels = true_labels.to(device)

    start = time.time()

    # Run simulation
    output = network.run(stim_batch, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)
    model_read_out = network.read_out(output, mode='classification')

    # print(time.time() - start)

    # firing_rates = network.get_firing_rates(output, area='v1')
    # network.analysis.plot_firing_rates(firing_rates)
    #
    # print(model_read_out)
    # print(true_labels.unsqueeze(1))

    # Compute loss
    # loss = torch.mean(torch.abs(model_read_out - true_labels.unsqueeze(1))) # mae
    loss = ((model_read_out - true_labels.unsqueeze(1)) ** 2).mean() # mse
    return loss




if __name__ == '__main__':

    # Params
    nr_epochs               = 100
    lr                      = 0.5
    train_with_adjoint      = False
    train_with_noise        = False
    seed                    = 1
    device                  = torch.device('cpu')

    set_seed(seed)

    # Building the network
    network = BrainNetwork()

    network.add_area(area_name='v1', size=2)
    network.add_area(area_name='v2', size=1)

    network.add_input_connection(target_area='v1', input_size=2, trainable=True)
    network.add_feedforward_connection(source='v1', target='v2', trainable=True, scale=10.0)
    network.add_output_connection(source_area='v2', trainable=False)

    # Training setup
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)


    # Start training loop
    for itr in range(nr_epochs):
        optimizer.zero_grad()

        # Run simulation and compute loss
        loss = run_xor_batch(train_with_adjoint, train_with_noise, device)

        loss.backward()
        optimizer.step()

        # Test
        with torch.no_grad():
            test_loss = run_xor_batch(train_with_adjoint, train_with_noise, device)
            print('Iter {:02d} | Train Loss {:.4f} | Test Loss {:.4f}'.format(itr + 1, loss.item(), test_loss.item()))

