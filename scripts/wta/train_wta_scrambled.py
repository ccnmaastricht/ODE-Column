import torch
import numpy as np
from scipy.linalg import block_diag

from train_wta import build_wta_network, get_data

from src.utils.paths import data_path, models_path
from src.utils.loss_functions import huber_loss_wta
from src.utils.set_seed import set_seed



def make_scrambled_connectivity(area, perturb_param):
    """
    Scrambles the recurrent connectivity of the specified BrainArea. Specifically,
    scrambles the synapse counts and then applies the recurrent synapse strength to
    the resulting matrix.
    """
    raw_connectivity = area.recurrent_synapse_counts[:8, :8].detach().cpu().numpy()

    # Separate diagonal connections to not scramble these
    diagonal = np.diag(np.diag(raw_connectivity))
    non_diagonal = raw_connectivity - diagonal

    row_sums = non_diagonal.sum(axis=1)
    col_sums = non_diagonal.sum(axis=0)

    rng = np.random.default_rng()
    scrambled_non_diagonal = non_diagonal * (1 + perturb_param * rng.normal(size=non_diagonal.shape))
    scrambled_non_diagonal = np.clip(scrambled_non_diagonal, 1e-12, None)

    # Sinkhorn algorithm
    for _ in range(5000):
        scrambled_non_diagonal *= row_sums[:, None] / scrambled_non_diagonal.sum(axis=1, keepdims=True)
        scrambled_non_diagonal *= col_sums[None, :] / scrambled_non_diagonal.sum(axis=0, keepdims=True)

    scrambled = scrambled_non_diagonal + diagonal
    blocks = [scrambled] * 2  # * 2 columns
    scrambled_synapse_counts = block_diag(*blocks)

    # Multiply with synapse strength
    scrambled_synapse_counts = torch.tensor(scrambled_synapse_counts, dtype=torch.float32)
    scrambled_synapse_weights = scrambled_synapse_counts * area.recurrent_synaptic_strength
    return scrambled_synapse_weights

def train_wta_scrambled_connectivity(
        fn_target_data,
        seed,
        scramble_perturb_param,
        batch_size=32,
        lr=1e+1,
        num_epochs=3,
        test_freq=10,
        train_with_adjoint=True,
        train_with_noise=True,
        device=torch.device('cpu')):
    """
    Train a BrainNetwork to perform winner-take-all decision-making with
    scrambled recurrent (column-intrinsic) connectivity.
    """
    set_seed(seed)

    # Build network
    network = build_wta_network()

    # Reset the recurrent weights with scrambled weights
    scrambled_weights = make_scrambled_connectivity(network.areas['mt'], scramble_perturb_param)
    with torch.no_grad():
        network.connections['recurrent_mt_mt'].weights.copy_(scrambled_weights)

    # Prepare train and test data, and optimizer
    train_loader, test_states, test_stims = get_data(batch_size, network.params['model']['time_params'], fn_target_data, seed)
    test_states = test_states.to(device)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=lr)

    # Store losses
    train_losses = []
    test_losses = []

    # Start training loop
    for epoch in range(num_epochs):

        for itr, (true_states, stim_batch) in enumerate(train_loader):

            optimizer.zero_grad()
            true_states = true_states.to(device)

            output = network.run(stim_batch, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)

            # Compute loss between predicted and true states
            pred_states = network.read_out(output, mode='trajectory')
            loss = huber_loss_wta(pred_states, true_states)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Test
            if itr % test_freq == 0:
                with torch.no_grad():

                    output = network.run(test_stims, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)

                    pred_states = network.read_out(output, mode='trajectory')
                    test_loss = huber_loss_wta(pred_states, test_states)

                    print('Epoch {:02d} | Iter {:02d} | Train Loss {:.4f} | Test Loss {:.4f}'.format(
                        epoch, itr // test_freq, loss.item(), test_loss.item()))

                    train_losses.append(loss.item())
                    test_losses.append(test_loss.item())

    # Store training history and trained network
    history = {'train_losses': train_losses,
               'test_losses': test_losses}

    torch.save(history, models_path('wta_scrambled', f'wta_scrambled_history_{scramble_perturb_param}_{seed}.pt'))
    network.save(models_path('wta_scrambled', f'wta_scrambled_{scramble_perturb_param}_{seed}.pt'))



if __name__ == '__main__':

    fn_target_data      = data_path('ds_wta.pt')
    num_epochs          = 3
    test_freq           = 10
    train_with_adjoint  = True
    train_with_noise    = True
    device              = torch.device('cpu')

    perturb_params = [1e-1, 5e-1, 1e+0]

    for pp in perturb_params:
        for seed in range(1, 11):

            print(f'Perturb param: {pp} || Seed: {seed}')

            train_wta_scrambled_connectivity(
            fn_target_data,
            seed=seed,
            scramble_perturb_param=pp,
            num_epochs=num_epochs,
            test_freq=test_freq,
            train_with_adjoint=train_with_adjoint,
            train_with_noise=train_with_noise,
            device=device)
