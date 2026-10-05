import torch
import matplotlib.pyplot as plt

from src.brain_network import BrainNetwork
from src.utils.paths import config_path



if __name__ == '__main__':

    model_config_path = config_path('wta_pmd_params.toml')
    general_config_path = config_path('general_params_agranular.toml')

    network = BrainNetwork.from_toml(model_config_path, general_config_path)
    network.add_area('pmd', 1, unique_id='pmd_a')
    network.add_area('pmd', 1, unique_id='pmd_b')
    network.add_input_connection('pmd_a', input_size=1)
    network.add_feedforward_connection(source='pmd_a', target='pmd_b')
    network.add_output_connection('pmd_b')

    output = network.run([10.])

    firing_rates = network.analysis.plot_firing_rates(output)
