from src.brain_network import BrainNetwork



if __name__ == '__main__':
    config_path = '../config/parity_params.toml'
    network = BrainNetwork.from_toml(config_path)

    network.add_area('v1', 128)
    network.add_area('v2', 10)

    network.add_input_connection('v1', 100, receptive_field_size=3, stride=1, grid_organization=True)
    network.add_feedforward_connection('v1', 'v2')
    network.add_output_connection('v2')
