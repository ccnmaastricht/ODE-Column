import numpy as np
import matplotlib.pyplot as plt



class NetworkAnalyzer:

    def __init__(self, network):
        self.network = network
        self.layer_labels = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

    def get_firing_rates(self, raw_state, area=None, return_as_np_array=True):
        """
        Compute the firing rate from the raw state (= [membrane_potential, adaptation])
        Return as np.array unless specified otherwise.
        """
        return self.network.get_firing_rates(raw_state, area, return_as_np_array)

    def plot_firing_rates(self, firing_rates, sample=None, area=None, column=None, population=None):
        """
        Plot firing rates
        """
        sample_list     = np.arange(firing_rates.shape[1])
        area_list       = self.network.areas.keys()
        column_list     = {area_id: np.arange(area.num_columns) for area_id, area in self.network.areas.items()}
        pop_list        = np.arange(8)

        if area is not None:
            area_slices = self.network.area_slices[area]
            area_list = [area]

            firing_rates = {area: firing_rates[:, :, area_slices]}
        else:
            firing_rates = {area_id: firing_rates[:, :, area_slice]
                           for area_id, area_slice in self.network.area_slices.items()}

        firing_rates = {area_id: fr.reshape(fr.shape[0],
                                         fr.shape[1],
                                         fr.shape[2] // 8,
                                         8)
                        for area_id, fr  in firing_rates.items()}

        if sample is not None:
            if isinstance(sample, int):
                sample = [sample]
            firing_rates = {area_id : fr[:, sample, :] for area_id, fr in firing_rates.items()}
            sample_list = sample

        if column is not None:
            if isinstance(column, int):
                column = [column]

            firing_rates = {area_id : fr[:, :, column, :] for area_id, fr in firing_rates.items()}
            column_list = {area_id : column for area_id in firing_rates.keys()}

        if population is not None:
            if not isinstance(population, list):
                population = [population]

            for i, pop in enumerate(population):
                if isinstance(pop, str):
                    population[i] = self.layer_labels.index(pop)

            firing_rates = {area : fr[:, :, :, population] for area, fr in firing_rates.items()}

            pop_list = population

        for sample_ in range(len(sample_list)):
            # fig = plt.figure(1, ((14, 7)))

            for area_ in area_list:
                for col in range(len(column_list[area_])):
                    for pop in range(len(pop_list)):

                        pop_name = self.layer_labels[pop_list[pop]]
                        label = f'{area_}_col{column_list[area_][col]}_{pop_name}'
                        plt.plot(firing_rates[area_][:, sample_, col, pop], label=label)

            plt.legend() # bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Sample {sample_list[sample_]}')
            plt.show()
