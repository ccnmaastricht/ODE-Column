from src.brain_network import BrainNetwork



if __name__ == '__main__':
    config_path = '../config/wta_params.toml'
    network = BrainNetwork.from_toml(config_path)

    network.add_area('v1', 4)
    network.add_input_connection('v1', 4, unique_id='bottom_up', receptive_field_size=1, stride=1)  # each column gets one bottom-up input
    network.add_input_connection('v1', 2, unique_id='context')  # fully connected
    # Todo: Add recurrent connection: with learnable L23e self-excitation weights
    network.add_lateral_connection('v1')
    network.add_output_connection('v1')


