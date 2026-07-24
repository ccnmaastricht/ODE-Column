import torch

from train_wta import build_wta_network, get_data

from src.utils.paths import data_path, models_path
from src.utils.loss_functions import huber_loss_wta
from src.utils.set_seed import set_seed



def compute_deviation(network, original_connectivity):
    """
    Computes the mean absolute error between the original connectivity profile
    and the current connectivity profile.
    """
    with torch.no_grad():
        current_connectivity = network.connections['recurrent_mt_mt'].W.detach()

        abs_diff = abs(current_connectivity - original_connectivity)
        return torch.mean(abs_diff)

def train_wta_optimized_connectivity(
        fn_target_data,
        seed,
        batch_size=32,
        num_epochs=3,
        test_freq=10,
        train_with_adjoint=True,
        train_with_noise=True,
        device=torch.device('cpu')):
    """
    Train a BrainNetwork to perform winner-take-all decision-making, with trainable
    recurrent (column-intrinsic) connectivity.
    """
    set_seed(seed)

    # Build network
    network = build_wta_network(optimize_connectivity=True)

    # Store the original connectivity profile
    with torch.no_grad():
        original_connectivity = network.connections['recurrent_mt_mt'].weights.clone().detach()

    # Prepare train and test data, and optimizer
    train_loader, test_states, test_stims = get_data(batch_size, network.params['model']['time_params'], fn_target_data, seed)
    test_states = test_states.to(device)
    optimizer = torch.optim.Adam([{'params': network.connections['lateral_mt_mt'].weights, 'lr': 10.0},
                                  {'params': network.connections['self_excitation_mt_mt'].weights, 'lr': 10.0},
                                  {'params': network.connections['recurrent_mt_mt'].weights, 'lr': 3.0}])

    # Store losses
    train_losses = []
    test_losses = []
    connectivity_deviations = []

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

                    connectivity_deviation = compute_deviation(network, original_connectivity)

                    print('Epoch {:02d} | Iter {:02d} | Train Loss {:.4f} | Test Loss {:.4f} | Deviation {:.4f}'.format(
                        epoch,
                        itr // test_freq,
                        loss.item(),
                        test_loss.item(),
                        connectivity_deviation))

                    train_losses.append(loss.item())
                    test_losses.append(test_loss.item())
                    connectivity_deviations.append(connectivity_deviation)

    # Store training history and trained network
    history = {'train_losses': train_losses,
               'test_losses': test_losses,
               'connectivity_deviations': connectivity_deviations}

    torch.save(history, models_path('wta_optimized', f'wta_optimized_history_{seed}.pt'))
    network.save(models_path('wta_optimized', f'wta_optimized_{seed}.pt'))


if __name__ == '__main__':

    fn_target_data = data_path('ds_wta.pt')
    seed = 1

    train_wta_optimized_connectivity(fn_target_data, seed)

