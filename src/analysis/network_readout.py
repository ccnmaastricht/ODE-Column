import torch
import numpy as np


class NetworkReadout:
    """
    Handles post-simulation signal extraction, population firing rate slicing, and
    task readout calculations across cortical network areas.

    Args:
        network (BrainNetwork): Target brain network module.

    Attributes:
        network (BrainNetwork): Reference to the target brain network.
        layer_labels (list[str]): Names of laminar populations (`['L23e', 'L23i', ...]`).
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
        Extract population firing rates from raw simulation state tensors and slice
        by sample, area, column, or population layers.

        Args:
            raw_state (torch.Tensor): Output tensor from ODE/SDE simulation containing
                membrane potentials and adaptation, shape
                `(time_steps, batch_size, 2 * total_populations)`.
            sample (int | list[int] | None, optional): Sample index or indices to
                filter. Defaults to None (all samples).
            area (str | None, optional): Area name to slice. Defaults to None (all areas).
            column (int | list[int] | None, optional): Column index or indices to
                filter. Defaults to None (all columns).
            population (int | str | list | None, optional): Laminar population index or
                label to filter. Defaults to None (all populations).
            return_as_np_array (bool, optional): Whether to return firing rates as
                NumPy arrays instead of PyTorch Tensors. Defaults to True.
            return_as_dict (bool, optional): Whether to return a structured dictionary
                mapping areas to 4D tensors `(time_steps, samples, columns, populations)`
                along with index lists. Defaults to False.

        Returns:
            np.ndarray | torch.Tensor | tuple: Filtered firing rate tensor or
                dictionary tuple.
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

        if return_as_np_array:
            firing_rates = firing_rates.detach().cpu().numpy()

        return firing_rates

    def _classification_read_out(self, fr_full_sim_time):
        """
        Compute time-averaged population firing rates over a specified classification
        window interval.

        Args:
            fr_full_sim_time (torch.Tensor): Full time-series firing rate tensor, shape
                `(time_steps, batch_size, output_dims)`.

        Returns:
            torch.Tensor: Time-averaged firing rate tensor across classification
                window, shape `(batch_size, output_dims)`.

        Raises:
            AssertionError: Raised if classification window is missing or exceeds
                simulation duration.
        """
        time_params = self.network.params['model']['time_params']
        assert 'classification_window' in time_params, (f"If mode is set to 'classification', please set "
                                                        f"a classification_window in model_params.toml under [time_params].")

        time_window = time_params['classification_window']
        sim_time = time_params['sim_time']
        dt = time_params['dt']

        start = time_window[0]
        end = time_window[1]
        assert start <= sim_time and end <= sim_time, (f"The classification time window ({start}s, {end}s) "
                                                       f"exceeds total simulation time ({sim_time}s)")

        # Get time steps of classification window and slice the firing_rates
        start, end = int(start / dt), int(end / dt)
        fr_window_slice = fr_full_sim_time[start:end, :, :]

        return torch.mean(fr_window_slice, dim=0)

    def read_out(self, raw_output, mode, sum_per_col):
        """
        Compute task readout outputs from target output connections using either
        continuous trajectory or classification window averaging.

        Args:
            raw_output (torch.Tensor): Raw simulation output trajectory tensor of
                shape `(time_steps, batch_size, 2 * total_populations)`.
            mode (str): Evaluation mode, either `'trajectory'` (full time-series) or
                `'classification'` (window-averaged).
            sum_per_col (bool): Whether to sum firing rates across all 8 laminar
                populations within each column.

        Returns:
            torch.Tensor | dict[str, torch.Tensor]: Processed readout tensor for a
                single output connection or dictionary of readout tensors for multiple
                connections.

        Raises:
            AssertionError: Raised if readout mode is not `'trajectory'` or
                `'classification'`.
        """
        assert mode == 'trajectory' or mode == 'classification', (f"Invalid mode for read-out. "
                                                                  f"Acceptable modes are 'trajectory' or 'classification'.")

        output_conns = {name : conn for name, conn in self.network.connections.items() if conn.conn_type == 'output'}
        read_outs = {}

        for conn_name, output_conn in output_conns.items():

            fr_output_area = self.get_firing_rates(raw_output, area=output_conn.source_id, return_as_np_array=False)
            read_out = fr_output_area * output_conn.W

            if sum_per_col:
                read_out_reshape = torch.reshape(read_out, (read_out.shape[0], read_out.shape[1], read_out.shape[2] // 8, 8))
                read_out = torch.sum(read_out_reshape, dim=-1)

            if mode == 'classification':
                read_out = self._classification_read_out(read_out)

            read_outs[conn_name] = read_out

        if len(output_conns) == 1:
            return read_out
        return read_outs
