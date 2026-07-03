from src.brain_network import BrainNetwork



if __name__ == '__main__':
    config_path = '../config/parity_params.toml'
    network = BrainNetwork.from_toml(config_path)

    network.add_area('v1', 4)
    network.add_area('v2', 2)
    network.add_area('v4', 2)

    network.add_input_connection('v1', 8, receptive_field_size=2, stride=2)
    network.add_feedforward_connection('v1', 'v2')
    network.add_feedforward_connection('v2', 'v4')
    network.add_output_connection('v4')
