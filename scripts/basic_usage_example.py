from src.brain_network import BrainNetwork

# 1. Load configuration and initialize network
network = BrainNetwork.from_toml(
  "../config/example_params.toml",  # application-specific params (see Setting Parameters for more info)
  "../config/general_params.toml"   # general model params (see Setting Parameters for more info)
)

# 2. Add brain areas and connections
network.add_area(area_name="v1", size=2)
network.add_area(area_name="v2", size=1)

network.add_input_connection(target_area="v1", input_size=2)
network.add_feedforward_connection(source="v1", target="v2")
network.add_output_connection(source_area="v2")

# Optional: Check initialized network connections
network.analysis.summarize_connections()
network.analysis.visualize_weights()

# 3. Run simulation
stimulus = [[15.0, 25.0]]  # stim size = (batch_size, input_size)
output = network.run(stimulus, adjoint=False, stochastic=True)

# 4. Examine network output
# 4a. Obtain model predictions
readout = network.read_out(output, mode="classification")
# 4b. Transform raw output state (= membrane potential, adaptation) to firing rates
firing_rates = network.get_firing_rates(output)
# 4c. Plot firing rates
network.analysis.plot_firing_rates(output)