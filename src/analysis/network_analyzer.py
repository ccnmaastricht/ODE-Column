import matplotlib.pyplot as plt



class NetworkAnalyzer:

    def __init__(self, network):
        self.network = network
        self.layers = ['L23e, L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

    def get_firing_rates(self, raw_state, area=None, return_as_np_array=True):
        """
        Compute the firing rate from the raw state (= [membrane_potential, adaptation])
        Return as np.array unless specified otherwise.
        """
        return self.network.get_firing_rates(raw_state, area, return_as_np_array)

    def plot_firing_rates(self, firing_rates):
        """
        Plot firing rates, separately for each sample
        """
        # TODO: refine (more plotting options)
        for i in range(firing_rates.shape[1]):
            for j in range(firing_rates.shape[-1]):
                plt.plot(firing_rates[:, i, j])
            plt.show()
