from src.brain_network import BrainNetwork


# Todo: Check if runs; and think about anything else to showcase here


if __name__ == '__main__':

    # Build the network
    config_path = '../config/example_params.toml'
    network = BrainNetwork.from_toml(config_path)

    network.add_area(area_name='v1', size=2)
    network.add_area(area_name='v2', size=1)

    network.add_feedforward_connection(source='v1', target='v2', trainable=True)
    network.add_feedback_connection(source='v2', target='v1', trainable=True)
    network.add_lateral_connection(area_name='v1', trainable=True)

    network.add_input_connection(target_area='v1', input_size=2, trainable=True)
    network.add_output_connection(source_area='v2', trainable=True)

    # Check network
    network.analysis.summarize_connections()
    network.analysis.visualize_weights()

    # Set stimuli, shape = (batch_size, input_size)
    stim = [[ 0., 10.],
            [ 0., 20.],
            [ 0., 30.]]

    # Run the network simulation with the stimuli
    output = network.run(stim, adjoint=False, stochastic=False, device='cpu')
    model_read_out = network.read_out(output, mode='classification')

    # Plot all firing rates
    network.analysis.plot_firing_rates(output)
