import torch
import numpy as np
import matplotlib.pyplot as plt

from src.brain_network import BrainNetwork
from src.utils.set_seed import set_seed
from src.utils.loss_functions import compute_ei_ratio_penalty
from src.utils.paths import config_path, models_path



def visualize_results(raw_output, network, stims, loss, epoch):
    """
    Look at results during training.
    """
    firing_rates = network.get_firing_rates(raw_output)

    for i in range(len(stims)):

        fig, axes = plt.subplots(3, 4, figsize=(13, 12))

        axes_indices = [(0, 0), (0, 1), (0, 2), (0, 3),
                        (1, 0), (1, 1), (1, 2), (1, 3),
                        (2, 0), (2, 1), (2, 2), (2, 3)]

        for j in range(network.num_columns):
            axes[axes_indices[j]].plot(firing_rates[:, i, (j * 8) + 0], label='L23e')
            axes[axes_indices[j]].plot(firing_rates[:, i, (j * 8) + 4], label='L5e')
            axes[axes_indices[j]].plot(firing_rates[:, i, (j * 8) + 6], label='L6e')

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
    Visualize learnable weights during training
    """
    for name, param in network.named_parameters():
        param_data = param.detach().cpu().numpy()

        if param.requires_grad:
            fig, ax = plt.subplots(figsize=(13, 7))

            if param_data.ndim == 2:  # 2D weight matrices: use heatmap
                heatmap = ax.imshow(param_data, cmap="viridis", interpolation="nearest")
                fig.colorbar(heatmap, ax=ax)

            # Clean filename (remove problematic characters)
            clean_name = name.replace('.', '_')
            plt.savefig('./results/{}_{:02d}'.format(clean_name, epoch))
            plt.close(fig)

def make_ds(batch_size=4):
    """
    Make a dataset of all parity inputs with fixed position.
    """

    all_combinations = torch.tensor([[0., 0., 0., 1.],
                                     [0., 0., 1., 1.],
                                     [0., 1., 1., 1.],
                                     [1., 1., 1., 1.]
                                    ], dtype=torch.float32)
    all_combinations *= 15.

    tile = batch_size // len(all_combinations)
    combinations_tiled = torch.tile(all_combinations, (tile, 1))

    ds = combinations_tiled[torch.randperm(combinations_tiled.size(0))]
    return ds

def train_parity(
        seed,
        batch_size=4,
        num_epochs=1000,
        test_freq=10,
        train_with_adjoint=False,
        train_with_noise=False,
        ei_weight=1e-1,
        device=torch.device('cpu')):
    """
    Train a BrainNetwork to learn parity (odd/even) classification.
    """
    set_seed(seed)

    # Build network
    config = config_path('parity_params.toml')
    network = BrainNetwork.from_toml(config)

    network.add_area('v1', 4)
    network.add_area('v2', 2)

    network.add_input_connection('v1', 4, receptive_field_size=1, stride=1)
    network.add_feedforward_connection('v1', 'v2', scale=2.0)
    network.add_output_connection('v2')

    network.add_lateral_connection('v1')
    network.add_lateral_connection('v2')

    for name, param in network.named_parameters():
        weights = param.detach().numpy()
        stop = 0

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Track history
    losses = []

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        train_set = make_ds(batch_size)
        train_set = train_set.to(device)

        output = network.run(train_set, adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)
        model_read_out = network.read_out(output, mode='classification')

        # Compute loss with MSE
        is_even = (train_set.sum(dim=1) % 30 == 0)
        # parity_targets = torch.stack([
        #     is_even.float(),  # even
        #     (~is_even).float()  # odd
        # ], dim=1)
        # parity_targets *= 20.
        # loss = ((model_read_out - parity_targets) ** 2).mean()

        # Or compute loss with CE
        parity_targets = is_even.long()
        loss = criterion(model_read_out, parity_targets)

        # Add E/I ratio penalty
        ei_penalty = compute_ei_ratio_penalty(network)
        loss += (ei_penalty * ei_weight)

        loss.backward()
        optimizer.step()

        print('Epoch {:02d} | Train Loss {:.4f}'.format(epoch, loss.item()))
        losses.append(loss.item())

        # Visualize
        if epoch % test_freq == 0:
            visualize_results(output, network, train_set, loss.item(), epoch)
            visualize_weights(network, epoch)

    # Store training history and trained network
    history = {'losses': losses}

    torch.save(history, models_path('parity', f'parity_history_{seed}.pt'))
    network.save(models_path('parity', f'parity_{seed}.pt'))



if __name__ == '__main__':

    seed = 1

    train_parity(seed)
