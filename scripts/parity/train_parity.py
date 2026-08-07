import torch
import matplotlib.pyplot as plt

from src.brain_network import BrainNetwork
from src.utils.set_seed import set_seed
from src.utils.loss_functions import compute_fr_volatility_penalty
from src.utils.paths import config_path, models_path



def visualize_results(raw_output, network, stims, loss, epoch):
    """
    Visualize parity results during training.
    """
    firing_rates = network.get_firing_rates(raw_output)

    for i in range(len(stims)):

        fig, axes = plt.subplots(4, 4, figsize=(13, 12))

        axes_indices = [(0, 0), (0, 1), (0, 2), (0, 3),
                        (1, 0), (1, 1), (1, 2), (1, 3),
                        (2, 0), (2, 1), (2, 2), (2, 3)]

        for j in range(network.num_columns):
            axes[axes_indices[j]].plot(firing_rates[:, i, (j * 8) + 0], label='L23e')
            axes[axes_indices[j]].plot(firing_rates[:, i, (j * 8) + 4], label='L5e')
            axes[axes_indices[j]].plot(firing_rates[:, i, (j * 8) + 6], label='L6e')

            if j > network.num_columns - 3:
                axes[axes_indices[j]].set_ylim(0.0, 2.0)

        fig.legend(loc="upper left")

        fig.text(0.2, 0.03, f"Training loss: {loss:.2f}", ha='center', fontsize=10, fontweight='bold')
        fig.text(0.5, 0.03, f"Input: {stims[i].reshape(1, len(stims[i]))}",
                 ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')

        plt.tight_layout(pad=3.0)
        fig.subplots_adjust(left=0.15)
        plt.savefig('./results/fr_{:02d}_{:1d}'.format(epoch, i))
        plt.close(fig)

def visualize_weights(network, epoch):
    """
    Visualize learnable weights during training.
    """
    for name, conn in network.connections.items():
        if conn.trainable:

            fig, ax = plt.subplots(figsize=(13, 7))

            weights = network.analysis.get_weights(name)

            heatmap = ax.imshow(weights, cmap="viridis", interpolation="nearest")
            fig.colorbar(heatmap, ax=ax)

            # Clean filename (remove problematic characters)
            clean_name = name.replace('.', '_')
            plt.savefig('./results/{}_{:02d}'.format(clean_name, epoch))
            plt.close(fig)

def make_ds(batch_size=8, noise_std=0.0):
    """
    Make a dataset of all parity inputs with fixed position.
    """
    all_inputs = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1.],
                                     [0., 0., 0., 0., 0., 0., 1., 1.],
                                     [0., 0., 0., 0., 0., 1., 1., 1.],
                                     [0., 0., 0., 0., 1., 1., 1., 1.],
                                     [0., 0., 0., 1., 1., 1., 1., 1.],
                                     [0., 0., 1., 1., 1., 1., 1., 1.],
                                     [0., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.]], dtype=torch.float32)
    all_inputs *= 15.

    tile = batch_size // len(all_inputs)
    inputs_tiled = torch.tile(all_inputs, (tile, 1))
    inputs = inputs_tiled[torch.randperm(inputs_tiled.size(0))]

    is_even = (inputs.sum(dim=1) % 30 == 0)
    labels = is_even.long()

    if noise_std > 0.0:
        mask = (inputs == 15.0).float()
        noise = torch.normal(mean=0.0, std=noise_std, size=inputs.shape)
        noise = noise * mask

        # Masked noise sums to zero
        n_masked = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        mean_noise = noise.sum(dim=-1, keepdim=True) / n_masked
        noise = (noise - mean_noise) * mask
        inputs = inputs + noise

    return inputs, labels

def train_parity(
        seed,
        batch_size=8,
        lr=5e-1,
        lambda_volatility=1e-1,
        input_perturbation=0.0,
        num_epochs=1000,
        test_freq=10,
        train_with_adjoint=True,
        train_with_noise=False,
        device=torch.device('cpu')):
    """
    Train a BrainNetwork to learn parity (odd/even) classification on 8 binary inputs.
    """
    set_seed(seed)

    # Build network
    config = config_path('parity_params.toml')
    network = BrainNetwork.from_toml(config)

    network.add_area('v1', 8)
    network.add_area('v2', 2)

    network.add_input_connection('v1', 8, receptive_field_size=1, stride=1)
    network.add_feedforward_connection('v1', 'v2', scale=2.0)
    network.add_output_connection('v2')

    network.add_lateral_connection('v1')
    network.add_lateral_connection('v2')

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    criterion = torch.nn.CrossEntropyLoss()

    def run_batch(stims, labels):
        """
        Run one parity batch through the network and compute the loss.
        """
        output = network.run(stims, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)
        model_read_out = network.read_out(output, mode='classification')

        # Compute loss with CE
        loss = criterion(model_read_out, labels)

        # Add firing rates volatility penalty
        firing_rates = network.get_firing_rates(output, return_as_np_array=False)
        volatility_penalty = compute_fr_volatility_penalty(firing_rates, max_mean_sq_d=0.0)
        loss += (volatility_penalty * lambda_volatility)

        # Compute the accuracy
        acc = (labels == torch.argmax(model_read_out, dim=1)).float().mean()

        return output, loss, (volatility_penalty * lambda_volatility), acc

    # Track history
    losses = []
    volatility = []
    test_losses = []
    test_volatility = []
    acc_no_noise = []
    acc_low_noise = []
    acc_high_noise = []

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        train_inputs, train_labels = make_ds(batch_size, noise_std=input_perturbation)
        raw_output, loss, volatility_penalty, _ = run_batch(train_inputs.to(device), train_labels.to(device))

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        volatility.append(volatility_penalty.item())

        # Test with no noise, low noise and high noise in inputs
        if epoch % test_freq == 0:

            test_accuracy = []

            for noise in [0.0, 0.2, 0.5]:

                test_inputs, test_labels = make_ds(batch_size*8, noise_std=noise)
                test_output, test_loss, test_vol, acc = run_batch(test_inputs.to(device), test_labels.to(device))

                if noise == input_perturbation:
                    test_losses.append(test_loss.item())
                    test_volatility.append(test_vol.item())

                    # Visualize
                    visualize_results(test_output, network, test_inputs, test_loss.item(), epoch)
                    visualize_weights(network, epoch)

                test_accuracy.append(acc)

            acc_no_noise.append(test_accuracy[0])
            acc_low_noise.append(test_accuracy[1])
            acc_high_noise.append(test_accuracy[2])

            print('Epoch {:02d} | Train Loss {:.4f} | No noise {:.2f} | Low noise {:.2f} | High noise {:.2f}'.format(
                epoch, loss.item(), test_accuracy[0], test_accuracy[1], test_accuracy[2]))

    # Store training history and trained network
    history = {'train_losses': losses,
               'volatility': volatility,
               'accuracy_no_noise': acc_no_noise,
               'accuracy_low_noise': acc_low_noise,
               'accuracy_high_noise': acc_high_noise}

    torch.save(history, models_path('parity', f'parity_history_{seed}.pt'))
    network.save(models_path('parity', f'parity_{seed}.pt'))



if __name__ == '__main__':

    num_epochs          = 501
    test_freq           = 50
    train_with_adjoint  = False
    train_with_noise    = False
    device              = torch.device('cpu')

    for seed in range(8, 11):
        print('Seed:', seed)

        train_parity(
            seed,
            num_epochs=num_epochs,
            test_freq=test_freq,
            train_with_adjoint=train_with_adjoint,
            train_with_noise=train_with_noise,
            device=device)
