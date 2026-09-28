import torch
import numpy as np
import matplotlib.pyplot as plt

from src.brain_network import BrainNetwork
from src.utils.paths import models_path
from train_orientations import get_twelve_orientations



def plot_history(history_path):
    history = torch.load(history_path, weights_only=False)

    for key, values in history.items():
        print(key)
        plt.plot(values)
        plt.show()

def plot_lateral_connectivity(network):
    lateral_weights = network.analysis.get_weights(conn_name='lateral_v1_v1')

    for source_col in range(12):

        lat_weight_profile = []

        for target_col in range(12):

            if source_col != target_col:

                excitatory_L23_weight = lateral_weights[target_col * 8, source_col * 8]
                inhibitory_L23_weight = lateral_weights[(target_col * 8) + 1, source_col * 8]

                weight_diff_L23 = excitatory_L23_weight - inhibitory_L23_weight
                lat_weight_profile.append(weight_diff_L23)

        plt.plot(lat_weight_profile)
        plt.show()

def plot_centered_lateral_connectivity(network):
    """
    Plots all 12 lateral connectivity profiles centered on each column's preferred orientation (0°)
    in a single plot, along with a thick mean profile line to highlight Mexican-hat tuning.
    """
    lateral_weights = network.analysis.get_weights(conn_name='lateral_v1_v1')

    # Relative offsets in columns (-6 to +6) and corresponding degree offsets (-90° to +90°)
    column_offsets = np.arange(-6, 7)
    degree_offsets = column_offsets * 15

    all_profiles = []

    plt.figure(figsize=(9, 5))

    for source_col in range(12):
        lat_weight_profile = []

        for offset in column_offsets:
            target_col = (source_col + offset) % 12

            excitatory_L23_weight = lateral_weights[target_col * 8, source_col * 8]
            inhibitory_L23_weight = lateral_weights[(target_col * 8) + 1, source_col * 8]

            weight_diff_L23 = excitatory_L23_weight - inhibitory_L23_weight

            if hasattr(weight_diff_L23, 'item'):
                weight_diff_L23 = weight_diff_L23.item()

            lat_weight_profile.append(weight_diff_L23)

        all_profiles.append(lat_weight_profile)
        plt.plot(degree_offsets, lat_weight_profile, alpha=0.4, linewidth=1.5, label=f'Col {source_col}')

    # Calculate and plot the mean profile across all 12 columns
    mean_profile = np.mean(all_profiles, axis=0)
    plt.plot(degree_offsets, mean_profile, color='black', linewidth=3.5, label='Mean Profile', zorder=10)

    # Reference lines and formatting
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    plt.xlabel('Relative Orientation Offset (degrees)')
    plt.ylabel('Effective L2/3 Weight (Excitatory - Inhibitory)')
    plt.title('Centered Lateral Connectivity Profiles (Mexican-Hat Alignment)')
    plt.xticks(degree_offsets, [f'{d}°' for d in degree_offsets])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_orientation_activations(network):

    orientations, labels = get_twelve_orientations(flatten=True)
    output = network.run(orientations)

    network.analysis.plot_firing_rates(output, area='v1', population='L23e')





if __name__ == '__main__':

    network_path = models_path('orientations', f'orientations_1.pt')
    history_path = models_path('orientations', f'orientations_history_1.pt')

    # plot_history(history_path)

    network = BrainNetwork.load(network_path)

    plot_centered_lateral_connectivity(network)

    # plot_orientation_activations(network)

    # network.analysis.visualize_weights()

    input_weights = network.analysis.get_weights(conn_name='input_input_v1')
    lateral_weights = network.analysis.get_weights(conn_name='lateral_v1_v1')
    stop = 0

