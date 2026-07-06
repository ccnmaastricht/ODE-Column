import torch
import numpy as np



class NetworkReadout:

    """
    Handles the network read-out post-simulation.
    """

    def __init__(self, network):

        self.network = network
        self.layer_labels = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

    def get_firing_rates(
            self,
            raw_state,
            sample=None,
            area=None,
            column=None,
            population=None,
            return_as_np_array=True,
            return_as_dict=False):
        """
        Compute the firing rate from the raw state (= [membrane_potential, adaptation])
        Return as np.array unless specified otherwise.
        """
        split = self.network.num_populations
        firing_rates = self.network.dynamics.compute_firing_rate(
            raw_state[:, :, :split] - raw_state[:, :, split:(split * 2)])

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

        if return_as_dict:
            if return_as_np_array:
                firing_rates = {area : fr.detach().cpu().numpy()
                                for area, fr in firing_rates.items()}
            return firing_rates, sample_list, area_list, column_list, pop_list

        # Flatten (column, population) -> (column * population)
        firing_rates = {area: fr.reshape(fr.shape[0], fr.shape[1], -1)
                        for area, fr in firing_rates.items()}

        # Concatenate areas in the original network order
        firing_rates = torch.cat([firing_rates[area]
                                  for area in self.network.areas.keys() if area in firing_rates], dim=2,)

        stop = 0

        if return_as_np_array:
            firing_rates = firing_rates.detach().cpu().numpy()

        return firing_rates


    def classification_read_out(self, fr_full_sim_time):
        """
        Use a classification time window to average network activity over time.
        """
        time_params = self.network.params['model']['time_params']
        assert 'classification_window' in time_params, (f"If mode is set to 'classification', please set "
                                                        f"a classification_window in model_params.toml under [time_params].")

        time_window = time_params['classification_window']
        sim_time = time_params['sim_time']
        dt = time_params['dt']

        start = time_window[0]
        end = time_window[1]
        assert start <= sim_time and end <= sim_time, (f"The input time window ({start}s, {end}s) "
                                                       f"exceeds total simulation time ({sim_time}s)")

        # Get time steps of classification window and slice the firing_rates
        start, end = int(start / dt), int(end / dt)
        fr_window_slice = fr_full_sim_time[start:end, :, :]

        return torch.mean(fr_window_slice, dim=0)


    def read_out(self, raw_output, mode, sum_per_col):
        """
        Read output from raw output; either return entire trajectory or average
        last x time steps, i.e. trajectory-based vs classification-based training procedure.
        """
        assert mode == 'trajectory' or mode == 'classification', f"Invalid mode for read-out. Acceptable modes are 'trajectory' or 'classification'."

        read_outs = {}

        for conn_name, output_conn in self.network.output_connections.items():
            fr_output_area = self.get_firing_rates(raw_output, area=output_conn.source_id, return_as_np_array=False)
            read_out = fr_output_area * output_conn.weights

            if sum_per_col:
                read_out_reshape = torch.reshape(read_out,
                                                 (read_out.shape[0], read_out.shape[1], read_out.shape[2] // 8, 8))
                read_out = torch.sum(read_out_reshape, dim=-1)

            if mode == 'classification':
                read_out = self.classification_read_out(read_out)

            if len(list(self.network.output_connections.keys())) == 1:
                return read_out

            read_outs[conn_name] = read_out
        return read_outs