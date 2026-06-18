import torch

from src.utils.save_and_load import load_config
from src.structure.brain_area import BrainArea
from src.structure.connection import Connection
from src.simulation.network_simulator import NetworkSimulator
from src.analysis.network_analyzer import NetworkAnalyzer

from src.dynamics.network_dynamics import NetworkDynamics
from src.analysis.network_readout import NetworkReadout



class BrainNetwork(torch.nn.Module):

    """
    Network class that allows the initialization of BrainArea and Connection objects.
    Additionally, contains additional modules for network dynamics, simulation and analysis.
    """

    def __init__(self):
        super().__init__()

        col_params = load_config('../config/column_params.toml')
        model_params = load_config('../config/model_params.toml')

        self.params = {"column": col_params, "model": model_params}

        self.num_populations    = None
        self.num_columns        = None
        self.area_slices        = {}
        self.areas              = torch.nn.ModuleDict({})
        self.connections        = torch.nn.ModuleDict({})
        self.output_connections = torch.nn.ModuleDict({})

        self._initialize_general_parameters(col_params)
        self._initialize_additional_modules()

    def _initialize_general_parameters(self, params):
        """
        Initialize general parameters that apply for the entire network.
        """
        # Background drive
        bg_drive = torch.tensor(params['background_drive'], dtype=torch.float32)
        self.register_buffer("background_drive", bg_drive.unsqueeze(0))  # add extra dim

        # Firing rate functionality
        fr_params = params['firing_rate_params']
        self.register_buffer("gain", torch.tensor(fr_params['gain'], dtype=torch.float32))
        self.register_buffer("threshold", torch.tensor(fr_params['threshold'], dtype=torch.float32))
        self.register_buffer("noise_factor", torch.tensor(fr_params['noise_factor'], dtype=torch.float32))

        # Time constants
        time_constants = params['time_constants']
        self.register_buffer("synapse_time_constant", torch.tensor(time_constants['synapse'], dtype=torch.float32))
        self.register_buffer("membrane_time_constant", torch.tensor(time_constants['membrane'], dtype=torch.float32))
        self.register_buffer("adapt_time_constant", torch.tensor(time_constants['adaptation'], dtype=torch.float32))

        # Membrane resistance and adaptation strength
        resistance = time_constants['membrane'] / params['capacitance']
        self.register_buffer("resistance", torch.tensor(resistance, dtype=torch.float32))
        self.register_buffer("adaptation_strength", torch.tensor(
            self.params['model']['adaptation_strength'], dtype=torch.float32))

    def _initialize_additional_modules(self):
        """
        Initialize NetworkDynamics(), NetworkSimulator(), NetworkReadout()
        and NetworkAnalyzer().
        """
        self.dynamics   = NetworkDynamics(self)
        self.simulator  = NetworkSimulator(self)
        self.readout    = NetworkReadout(self)
        self.analysis   = NetworkAnalyzer(self)

    def _get_area(self, area_id):
        """
        Returns the Area object from the self.areas dict.
        """
        area_id = area_id.lower()
        assert area_id in self.areas.keys(), f"Area '{area_id}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."

        return self.areas[area_id]

    def add_area(self,
                 area_name,
                 size,
                 unique_id=None,
                 intrinsic_trainable=False,
                 background_trainable=False):
        """
        Initialize the specified area and its recurrent and background connections.

        Params:
        area_name (str):                The name of the to-be-modeled area, as specified in the .toml file (e.g. 'v1', 'v2', etc).
        size (int):                     The number of columns of the area.
        unique_id (str):                An optional user-specified id for the area. Useful when the network should contain more
                                        area modules with the same area configurations.
        intrinsic_trainable (bool):     If True, the recurrent, column-intrinsic connections can be updated during training.
        background_trainable (bool):    If True, the background connections can be updated during training.
        """
        area_name = area_name.lower()
        assert area_name in self.params['column']['population_size'], f"Population sizes of '{area_name}' not found in .toml file. "
        assert area_name in self.params['column']['background_synapse_counts'], f"Background synapse counts of '{area_name}' not found in .toml file. "

        if unique_id is None:
            unique_id = area_name

        area = BrainArea(self.params['column'], area_name, size, unique_id)
        self.areas[unique_id] = area

        # Add recurrent connectivity and background connectivity as connections
        recurrent_connection = Connection('recurrent', unique_id, unique_id, intrinsic_trainable)
        recurrent_connection.initialize_recurrent_weights(area)
        self.connections[recurrent_connection.get_name()] = recurrent_connection

        background_connection = Connection('background', 'background', unique_id, background_trainable)
        background_connection.initialize_background_weights(area)
        self.connections[background_connection.get_name()] = background_connection

    def add_feedforward_connection(self,
                                   source,
                                   target,
                                   trainable=True,
                                   std=0.1,
                                   scale=1.0):

        source_area = self._get_area(source)
        target_area = self._get_area(target)

        connection = Connection('feedforward', source, target, trainable)
        connection.initialize_feedforward_weights(self.params, source_area, target_area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_feedback_connection(self,
                                source,
                                target,
                                trainable=True,
                                std=0.1,
                                scale=1.0):

        source_area = self._get_area(source)
        target_area = self._get_area(target)

        connection = Connection('feedback', source, target, trainable)
        connection.initialize_feedback_weights(self.params, source_area, target_area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_lateral_connection(self,
                               area_name,
                               trainable=True,
                               std=0.1,
                               scale=1.0):

        area = self._get_area(area_name)

        connection = Connection('lateral', area_name, area_name, trainable)
        connection.initialize_lateral_weights(self.params, area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_input_connection(self,
                             target_area,
                             input_size,
                             trainable=True,
                             unique_id=None,
                             std=0.1,
                             scale=1.0):

        area = self._get_area(target_area)

        input_name = 'input'
        if unique_id is not None:
            input_name = unique_id

        connection = Connection('input', input_name, target_area, trainable)
        connection.initialize_input_weights(self.params, input_size, area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_output_connection(self,
                              source_area,
                              trainable=True,
                              unique_id=None,
                              std=0.0,
                              scale=1.0):

        area = self._get_area(source_area)

        output_name = 'output'
        if unique_id is not None:
            output_name = unique_id

        connection = Connection('output', source_area, output_name, trainable)
        connection.initialize_output_weights(self.params, area, std, scale)
        self.output_connections[connection.get_name()] = connection

    def finalize(self):
        """
        Finalizes the network after initializing all areas and connections by setting
        the total number of populations and columns, and setting slices to index the
        activity of each area. Also extends adaptation strength to entire network.
        """
        self.num_populations = sum(area.num_populations for area in self.areas.values())
        self.num_columns = self.num_populations // 8

        # Slices per area
        idx = 0
        for area_id, area in self.areas.items():
            self.area_slices[area_id] = slice(idx, idx + area.num_populations)
            idx += area.num_populations

        # Extend adaptation strength tensor to cover the entire network
        self.register_buffer("adaptation_strength_full", torch.tile(self.adaptation_strength,(self.num_columns,)))

    def constrain_weights(self):
        """
        Constrain all connection weights to not use any illegal connections.
        """
        all_connections = (list(self.connections.values())
                           + list(self.output_connections.values()))

        for connection in all_connections:
            connection.constrain(self.areas.keys())

    # def compute_firing_rate(self, x):
    #     """
    #     Compute the firing rates from (membrane potential - adaptation).
    #     """
    #     # TODO: look at ImageColumnModel for an updated version! Also, is soft clamp necessary?
    #
    #     x_nom = self.gain * x - self.threshold
    #     exp_input = -self.noise_factor * x_nom
    #     # exp_input = soft_clamp(exp_input)
    #     exp_term = torch.exp(exp_input)
    #
    #     denom = 1 - exp_term
    #     x_activ = x_nom / denom
    #     return x_activ
    #
    # def soft_clamp(self, x, max_val=80):
    #     return max_val * torch.tanh(x / max_val)
    #
    # def set_activities(self, t, fr_per_area, ext_input, input_windows):
    #     """
    #     Gather all activities in a dict; that includes the firing rates
    #     of all areas, background rate and external inputs.
    #     """
    #     # Add background drive
    #     activities = {'background': self.background_drive}
    #
    #     # Add firing rates of all network areas
    #     activities.update(fr_per_area)
    #
    #     # Add network-external input
    #     if ext_input is not None:
    #         for input_name, x in ext_input.items():
    #             # Present input if t is in input window
    #             start, end = input_windows[input_name]
    #             if start <= float(t) < end:
    #                 activities[input_name] = x
    #             else:
    #                 activities[input_name] = torch.zeros_like(x)
    #
    #     return activities
    #
    # def compute_currents(self, t, firing_rates, ext_input, input_windows):
    #     """
    #     For each area, compute the current based on all incoming connections.
    #     """
    #     fr_per_area = {area_id: firing_rates[:, area_slice]
    #                    for area_id, area_slice in self.area_slices.items()}
    #
    #     activities = self.set_activities(t, fr_per_area, ext_input, input_windows)
    #
    #     currents = {area_id: torch.zeros(firing_rates.shape[0], area.num_populations, device=firing_rates.device)
    #                 for area_id, area in self.areas.items()}
    #
    #     for connection in self.connections.values():
    #         conn_type = connection.conn_type
    #         source_fr = activities[connection.source_id]
    #         current = source_fr @ connection.W.T
    #         currents[connection.target_id] += current * self.synapse_time_constant
    #         stop = 0
    #
    #     total_current = torch.cat([currents[area_id] for area_id in self.areas], dim=1)  # TODO: check if there is no mess up of area order!
    #     return total_current
    #
    # def forward(self, t, state, ext_input, input_windows):
    #     """
    #     State dynamics computing the derivative of the membrane potential and adaptation at time t.
    #     """
    #     # Unpack the state (membrane, adaptation) and compute firing rate
    #     mem_adap_split = self.num_populations
    #     membrane_potential, adaptation = state[:, :mem_adap_split], state[:, mem_adap_split:]
    #
    #     firing_rate = self.compute_firing_rate(membrane_potential - adaptation)
    #
    #     # Compute current
    #     total_current = self.compute_currents(t, firing_rate, ext_input, input_windows)
    #
    #     # Compute derivative membrane potential and adaptation
    #     delta_membrane_potential = (-membrane_potential +
    #         total_current * self.resistance) / self.membrane_time_constant
    #     delta_adaptation = (-adaptation + self.adaptation_strength_full *
    #                         firing_rate) / self.adapt_time_constant
    #
    #     state = torch.concat((delta_membrane_potential, delta_adaptation), dim=1)
    #     return state
    #
    # def diffusion(self, t, state):
    #     '''
    #     Diffusion function used by SDE, noise is only applied to membrane potential.
    #     '''
    #     g = torch.zeros_like(state)
    #     n = self.num_populations
    #     g[:, :n] = 3.0
    #     return g

    def run(self, ext_input=None, input_window=None, adjoint=False, stochastic=False, device="cpu"):
        """
        Delegates running the network to the simulation engine.
        """
        return self.simulator.run(
            ext_input=ext_input,
            input_window=input_window,
            adjoint=adjoint,
            stochastic=stochastic,
            device=device)

    def get_firing_rates(self, raw_state, area=None, return_as_np_array=True):
        """
        Computes the firing rate from the raw state (= [membrane_potential, adaptation])
        Returns as np.array unless specified otherwise.
        """
        return self.readout.get_firing_rates(
            raw_state=raw_state,
            area=area,
            return_as_np_array=return_as_np_array)

    def read_out(self, raw_output, mode, sum_per_col=True):
        """
        Reads output from raw output; converts to firing rates and slices only
        the area(s) that are identified as output sources.
        Either returns entire trajectory (mode='trajectory') or average of
        last x time steps (mode='classification'), i.e. trajectory-based vs
        classification-based training procedure.
        """
        return self.readout.read_out(
            raw_output=raw_output,
            mode=mode,
            sum_per_col=sum_per_col)

    # def get_firing_rates(self, raw_state, area=None, return_as_np_array=True):
    #     """
    #     Compute the firing rate from the raw state (= [membrane_potential, adaptation])
    #     Return as np.array unless specified otherwise.
    #     """
    #     # TODO: refine (layer indices?)
    #     split = self.num_populations
    #     firing_rates = self.compute_firing_rate(raw_state[:, :, :split] - raw_state[:, :, split:(split * 2)])
    #
    #     if area is not None:
    #         area_slices = self.area_slices[area]
    #         firing_rates = firing_rates[:, :, area_slices]
    #
    #     if return_as_np_array:
    #         return firing_rates.detach().cpu().numpy()
    #     return firing_rates
    #
    # def classification_read_out(self, fr_full_sim_time):
    #     """
    #     Use a classification time window to average network activity over time.
    #     """
    #     time_params = self.params['model']['time_params']
    #     assert 'classification_window' in time_params, (f"If mode is set to 'classification', please set "
    #                                                     f"a classification_window in model_params.toml under [time_params].")
    #
    #     time_window     = time_params['classification_window']
    #     sim_time        = time_params['sim_time']
    #     dt              = time_params['dt']
    #
    #     start = time_window[0]
    #     end = time_window[1]
    #     assert start <= sim_time and end <= sim_time, (f"The input time window ({start}s, {end}s) "
    #                                                    f"exceeds total simulation time ({sim_time}s)")
    #
    #     # Get time steps of classification window and slice the firing_rates
    #     start, end = int(start / dt), int(end / dt)
    #     fr_window_slice = fr_full_sim_time[start:end, :, :]
    #
    #     return torch.mean(fr_window_slice, dim=0)
    #
    # def read_out(self, raw_output, mode, sum_per_col=True):
    #     """
    #     Read output from raw output; either return entire trajectory or last x time steps,
    #     i.e. trajectory-based vs classification-based training procedure...
    #     """
    #     assert mode == 'trajectory' or mode == 'classification', f"Invalid mode for read-out. Acceptable modes are 'trajectory' or 'classification'."
    #
    #     read_outs = {}
    #
    #     for conn_name, output_conn in self.output_connections.items():
    #         fr_output_area = self.get_firing_rates(raw_output, area=output_conn.source_id, return_as_np_array=False)
    #         read_out = fr_output_area * output_conn.weights
    #
    #         if sum_per_col:
    #             read_out_reshape = torch.reshape(read_out, (read_out.shape[0], read_out.shape[1], read_out.shape[2]//8, 8))
    #             read_out = torch.sum(read_out_reshape, dim=-1)
    #
    #         if mode == 'classification':
    #             read_out = self.classification_read_out(read_out)
    #
    #         if len(list(self.output_connections.keys())) == 1:
    #             return read_out
    #
    #         read_outs[conn_name] = read_out
    #     return read_outs