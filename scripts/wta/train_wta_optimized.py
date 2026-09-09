import torch

from train_wta import build_wta_network, get_data

from src.utils.paths import data_path, models_path
from src.utils.loss_functions import huber_loss_wta, compute_fr_ceiling_penalty
from src.utils.set_seed import set_seed



def compute_deviation(network, original_connectivity):
    """
    Compute the mean absolute error between the original connectivity profile
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
        lr_lateral=1e+1,
        lr_recurrent=3e+0,
        lambda_fr_reg=1e-6,
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
    optimizer = torch.optim.RMSprop([{'params': network.connections['lateral_mt_mt'].weights, 'lr': lr_lateral},
                                     {'params': network.connections['self_excitation_mt_mt'].weights, 'lr': lr_lateral},
                                     {'params': network.connections['recurrent_mt_mt'].weights, 'lr': lr_recurrent}])

    # Store losses
    train_losses = []
    train_fr_reg = []
    test_losses = []
    test_fr_reg = []
    connectivity_deviations = []

    # Start training loop
    for epoch in range(num_epochs):

        for itr, (true_states, stim_batch) in enumerate(train_loader):

            optimizer.zero_grad()
            true_states = true_states.to(device)

            output = network.run(stim_batch, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)

            pred_states = network.read_out(output, mode='trajectory')
            huber_loss = huber_loss_wta(pred_states, true_states)

            firing_rates = network.get_firing_rates(output, return_as_np_array=False)
            fr_reg = compute_fr_ceiling_penalty(firing_rates) * lambda_fr_reg
            loss = huber_loss + fr_reg

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_fr_reg.append(fr_reg.item())

            # Test
            if itr % test_freq == 0:
                with torch.no_grad():
                    output = network.run(test_stims, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)

                    pred_states = network.read_out(output, mode='trajectory')
                    test_huber_loss = huber_loss_wta(pred_states, test_states)

                    firing_rates = network.get_firing_rates(output, return_as_np_array=False)
                    fr_reg_test = compute_fr_ceiling_penalty(firing_rates) * lambda_fr_reg
                    test_loss = test_huber_loss + fr_reg_test

                    connectivity_deviation = compute_deviation(network, original_connectivity)

                    print('Epoch {:02d} | Iter {:02d} | Train Loss {:.4f} | Test Loss {:.4f} | Deviation {:.4f}'.format(
                        epoch,
                        itr // test_freq,
                        loss.item(),
                        test_loss.item(),
                        connectivity_deviation))

                    test_losses.append(test_loss.item())
                    test_fr_reg.append(fr_reg_test.item())
                    connectivity_deviations.append(connectivity_deviation)

    # Store training history and trained network
    history = {'train_losses': train_losses,
               'train_fr_reg': train_fr_reg,
               'test_losses': test_losses,
               'test_fr_reg': test_fr_reg,
               'connectivity_deviations': connectivity_deviations}

    torch.save(history, models_path('wta_optimized', f'wta_optimized_history_{seed}.pt'))
    network.save(models_path('wta_optimized', f'wta_optimized_{seed}.pt'))



if __name__ == '__main__':

    fn_target_data      = data_path('ds_wta.pt')
    lr_recurrent        = 1e-1
    num_epochs          = 3
    test_freq           = 10
    train_with_adjoint  = True
    train_with_noise    = True
    device              = torch.device('cpu')

    for seed in range(1, 11):
        print('Seed:', seed)

        train_wta_optimized_connectivity(
        fn_target_data,
        seed=seed,
        lr_recurrent=lr_recurrent,
        num_epochs=num_epochs,
        test_freq=test_freq,
        train_with_adjoint=train_with_adjoint,
        train_with_noise=train_with_noise,
        device=device)

