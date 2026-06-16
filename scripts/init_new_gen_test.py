from src.network_new_gen import BrainNetwork, NetworkAnalyzer


# Set output
# Try training


if __name__ == '__main__':

    # Setting up
    network = BrainNetwork()
    analyzer = NetworkAnalyzer(network)

    network.add_area(area_name='v1', size=2)
    network.add_area(area_name='v2', size=1)

    network.add_feedforward_connection(source='v1', target='v2', trainable=True)
    network.add_feedback_connection(source='v2', target='v1', trainable=True)
    network.add_lateral_connection(area_name='v1', trainable=True)
    network.add_input_connection(target_area='v1', input_size=2, trainable=False)

    # Stimulus
    stim = [[20., 0.], [0., 20.]]

    # Run simulation
    output = network.run(stim, adjoint=False, stochastic=True, device='mps')

    # Plot firing rates
    firing_rates = analyzer.get_firing_rates(output)
    analyzer.plot_firing_rates(firing_rates)

