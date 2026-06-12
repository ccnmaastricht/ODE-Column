from torchdiffeq import odeint
import matplotlib.pyplot as plt

from src.utils import load_config, compute_firing_rate
from src.network_new_gen import BrainNetwork



if __name__ == '__main__':

    params = load_config('../config/params_new_gen.toml')

    network = BrainNetwork(params)

    network.add_area(area_name='v1', size=2)
    network.add_area(area_name='v2', size=1)

    network.add_projection(source='v1', target='v2', type='feedforward', trainable=True)
    network.add_projection(source='v1', target='v1', type='lateral', trainable=True)
    # network.add_input_area(area='v1', input_size=2, trainable=False)
    # network.add_output_area(area='v2', trainable=False)

    initial_state, time_vec = network.finalize(dt=1e-3, sim_time=1.0, batch_size=2)

    # Run
    ode_output = odeint(network, initial_state, time_vec)

    # Plot firing rates
    split = network.num_populations
    firing_rates = compute_firing_rate(ode_output[:, :, :split] - ode_output[:, :, split:(split * 2)]).detach().numpy()

    for i in range(firing_rates.shape[-1]):
        plt.plot(firing_rates[:, 0, i])
    plt.show()

