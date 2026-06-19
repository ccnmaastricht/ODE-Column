from src.brain_network import BrainNetwork



if __name__ == '__main__':

    # Building the network
    config_path = '../config/example_params.toml'
    network = BrainNetwork(config_path)

    # network.add_area(area_name='v1', unique_id='v1a', size=2)
    # network.add_area(area_name='v1', unique_id='v1b', size=1)
    #
    # network.add_feedforward_connection(source='v1a', target='v1b', trainable=True)
    # network.add_feedback_connection(source='v1b', target='v1a', trainable=True)
    # network.add_lateral_connection(area_name='v1a', trainable=True)
    #
    # network.add_input_connection(target_area='v1a', input_size=2, trainable=True)
    # network.add_output_connection(source_area='v1b', trainable=True)

    network.add_area(area_name='v1', size=2)
    network.add_area(area_name='v2', size=1)

    network.add_feedforward_connection(source='v1', target='v2', trainable=True)
    network.add_feedback_connection(source='v2', target='v1', trainable=True)
    network.add_lateral_connection(area_name='v1', trainable=True)

    network.add_input_connection(target_area='v1', input_size=2, trainable=True)
    network.add_output_connection(source_area='v2', trainable=True)

    # Stimulus batch (size = 3)
    stim = [[20.,  0.],
            [ 0., 20.],
            [10., 10.]]

    # Run simulation
    output = network.run(stim, adjoint=False, stochastic=True, device='cpu')
    model_read_out = network.read_out(output, mode='classification')

    # Plot all firing rates
    firing_rates = network.get_firing_rates(output)
    network.analysis.plot_firing_rates(firing_rates)
