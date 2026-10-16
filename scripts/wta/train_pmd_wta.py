import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from src.brain_network import BrainNetwork
from src.utils.paths import data_path, config_path, models_path
from src.utils.set_seed import set_seed
from src.utils.loss_functions import huber_loss_wta, compute_fr_ceiling_penalty

from train_wta import make_ds_ww, split_dataset, initialize_self_excitation_connection
from test_pmd_wta import plot_firing_rates_per_layer



def get_data(batch_size, network_time_params, fn, seed):
    """
    Get the Wang-Wong dataset and create a random train/test split.
    """
    dt = network_time_params['dt']
    sim_time = network_time_params['sim_time']
    time_steps = int(round(sim_time / dt))

    states, stims = make_ds_ww(fn, time_steps)

    # Scale down Wang-Wong firing rates
    states = states / 4.

    train_states, train_stims, test_states, test_stims = split_dataset(states, stims, seed)

    ds = TensorDataset(train_states, train_stims)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_states, test_stims

def train_wta_in_pmd(
        seed,
        fn_target_data,
        batch_size=32,
        lr=1e+0,
        lambda_fr_reg=1e-6,
        num_epochs=3,
        test_freq=20,
        train_with_adjoint=True,
        train_with_noise=True,
        device=torch.device("cpu")):
    """"""
    set_seed(seed)

    # Build network
    model_config_path = config_path('wta_pmd_params.toml')
    general_config_path = config_path('general_params_agranular.toml')

    network = BrainNetwork.from_toml(model_config_path, general_config_path)
    network.add_area('pmd', 2)

    network.add_input_connection('pmd', 2, receptive_field_size=1,
                                 stride=1, trainable=False, std=0.0)
    network.add_output_connection('pmd')

    network.add_lateral_connection('pmd', trainable=True)
    network.add_custom_connection(connection_name='self_excitation',
                                  source='pmd',
                                  target='pmd',
                                  initializer=initialize_self_excitation_connection)

    plot_firing_rates_per_layer(network)

    # with torch.no_grad():
    #     nan_to_zero_weights = torch.nan_to_num(network.connections['recurrent_pmd_pmd'].weights)
    #     network.connections['recurrent_pmd_pmd'].weights.copy_(nan_to_zero_weights)

    # Prepare train and test data, and optimizer
    train_loader, test_states, test_stims = get_data(batch_size, network.params['model']['time_params'],
                                                     fn_target_data, seed)
    test_states = test_states.to(device)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop([
    #     {'params': network.connections['lateral_pmd_pmd'].weights, 'lr': lr},
    #     {'params': network.connections['self_excitation_pmd_pmd'].weights, 'lr': 1e+1}
    # ])


    # Store losses
    train_losses = []
    train_fr_reg = []
    test_losses = []
    test_fr_reg = []


    def run_wta_pmd_batch(stim_batch, true_states, plot=False):
        output = network.run(stim_batch, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)

        # network.analysis.plot_firing_rates(output)

        firing_rates = network.get_firing_rates(output)

        pred_states = network.read_out(output, mode='trajectory')
        huber_loss = huber_loss_wta(pred_states, true_states)

        firing_rates = network.get_firing_rates(output, return_as_np_array=False)
        fr_reg = compute_fr_ceiling_penalty(firing_rates) * lambda_fr_reg
        loss = huber_loss + fr_reg

        if plot:
            return loss, fr_reg, firing_rates

        return loss, fr_reg


    # Training loop
    for epoch in range(num_epochs):

        for itr, (train_states, stim_batch) in enumerate(train_loader):

            optimizer.zero_grad()
            train_states = train_states.to(device)

            loss, fr_reg = run_wta_pmd_batch(stim_batch, train_states)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_fr_reg.append(fr_reg.item())

            # Test
            if itr % test_freq == 0:
                with torch.no_grad():

                    test_loss, fr_reg_test, firing_rates = run_wta_pmd_batch(test_stims, test_states, plot=True)

                    # for i in range(10):
                    #     print(test_stims[i])
                    #
                    #     plt.plot(firing_rates[:, i, 0], color='blue')
                    #     plt.plot(firing_rates[:, i, 6], color='orange')
                    #     plt.plot(test_states[i, :, 0], linestyle='--', color='blue')
                    #     plt.plot(test_states[i, :, 1], linestyle='--', color='orange')
                    #     plt.plot(firing_rates[:, i, 2], color='royalblue')
                    #     plt.plot(firing_rates[:, i, 8], color='red')
                    #     plt.show()
                    #
                    # network.analysis.visualize_weights(conn_name='lateral_pmd_pmd')
                    # network.analysis.visualize_weights(conn_name='self_excitation_pmd_pmd')

                    print('Epoch {:02d} | Iter {:02d} | Train Loss {:.4f} | Test Loss {:.4f}'.format(
                        epoch, itr // test_freq, loss.item(), test_loss.item()))

                    test_losses.append(test_loss.item())
                    test_fr_reg.append(fr_reg_test.item())

    # Store training history and trained network
    history = {'train_losses': train_losses,
               'train_fr_reg': train_fr_reg,
               'test_losses': test_losses,
               'test_fr_reg': test_fr_reg}

    torch.save(history, models_path('wta_pmd', f'wta_pmd_history_{seed}.pt'))
    network.save(models_path('wta_pmd', f'wta_pmd_{seed}.pt'))

    network.analysis.visualize_weights(conn_name='lateral_pmd_pmd')
    network.analysis.visualize_weights(conn_name='self_excitation_pmd_pmd')
    plot_firing_rates_per_layer(network)




if __name__ == '__main__':

    seed = 1
    fn_target_data = data_path('ds_wta.pt')

    train_wta_in_pmd(seed, fn_target_data, train_with_adjoint=False, train_with_noise=True)

