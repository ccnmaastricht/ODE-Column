import torch
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

def plot_orientation_activations(network):

    orientations, labels = get_twelve_orientations(flatten=True)
    output = network.run(orientations)

    network.analysis.plot_firing_rates(output, area='v1', population='L23e')





if __name__ == '__main__':

    network_path = models_path('orientations', f'orientations_1.pt')
    history_path = models_path('orientations', f'orientations_history_1.pt')

    # plot_history(history_path)

    network = BrainNetwork.load(network_path)

    # plot_lateral_connectivity(network)

    # plot_orientation_activations(network)

    # network.analysis.visualize_weights()

    input_weights = network.analysis.get_weights(conn_name='input_input_v1')
    lateral_weights = network.analysis.get_weights(conn_name='lateral_v1_v1')
    stop = 0
