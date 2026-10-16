import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.brain_network import BrainNetwork
from src.utils.paths import config_path, models_path, results_path



def plot_firing_rates_per_layer(network):

    # inputs = torch.tensor(
    #     [[40., 20.], [40., 22.], [40., 24.], [40., 26.], [40., 28.],
    #      [40., 30.], [40., 32.], [40., 34.], [40., 36.], [40., 38.],
    #      [40., 40.]])
    inputs = torch.tensor(
        [[40., 25.], [40., 26.], [40., 27.], [40., 28.], [40., 29.],
         [40., 30.], [40., 31.], [40., 32.], [40., 33.], [40., 34.],
         [40., 35.]])

    resting_state = network.run(torch.zeros_like(inputs), dt=1e-4, sim_time=0.1, input_window=[0.05, 0.1])
    output = network.run(inputs, dt=1e-4, sim_time=0.1, input_window=[0.05, 0.1], reset_state=False)
    firing_rates = network.get_firing_rates(output)

    # network.analysis.plot_firing_rates(output)

    # Plot
    colors = plt.get_cmap('rainbow', len(inputs))
    pop_labels = ['L23e', 'L23i', 'L5e', 'L5i', 'L6e', 'L6i']

    fig, axes = plt.subplots(3, 2, figsize=(8, 8))
    axes = axes.flatten()

    for pop in range(6):
        for sample in range(len(inputs)):
            axes[pop].plot(
                firing_rates[:, sample, pop],
                color=colors(sample)
            )
            axes[pop].plot(
                firing_rates[:, sample, pop + 6],
                color=colors(sample),
                linestyle='--'
            )

        axes[pop].set_title(f'Population {pop_labels[pop]}')
        axes[pop].set_xlabel('Time')
        axes[pop].set_ylabel('Firing rate')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    network_path = models_path('wta_pmd', f'wta_pmd_1.pt')
    network = BrainNetwork.load(network_path)

    plot_firing_rates_per_layer(network)

