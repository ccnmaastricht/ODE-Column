import torch
import numpy as np
from contourpy.array import concat_points_or_none
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint, sdeint_adjoint

from src.utils import compute_firing_rate, load_config



class BrainArea(torch.nn.Module):

    """
    Areas denote groups of columns with the same column parameters (e.g. population counts,
    background synapse counts, etc). The recurrent activity profile determines the column-
    intrinsic dynamics.
    """

    def __init__(self, col_params, area_name, num_columns, unique_name):
        super().__init__()

        self.num_columns = num_columns
        self.area_name = area_name
        self.name = unique_name

        self._initialize_population_parameters(col_params)
        self._initialize_connection_probabilities(col_params)
        self._initialize_synapses(col_params)
        self._build_all_weights()

    def _initialize_population_parameters(self, column_parameters):
        """
        Initialize the population sizes for the columns.
        """
        self.population_sizes = np.array(
            column_parameters['population_size'][self.area_name])
        self.population_sizes = np.tile(self.population_sizes, self.num_columns)
        self.num_populations = len(self.population_sizes)

        self._make_in_ex_masks(self.num_columns)

    def _initialize_connection_probabilities(self, column_parameters):
        """
        Initialize the connection probabilities for the columns.
        """
        self.internal_connection_probabilities = torch.tensor(
            column_parameters['connection_probabilities']['internal'])

        # Copy internal connections n times along diagonal for n columns
        blocks = [self.internal_connection_probabilities] * self.num_columns
        self.connection_probabilities = block_diag(*blocks)

    def _initialize_synapses(self, column_parameters):
        """
        Initialize the synapse counts and synaptic strengths for the columns.
        """
        background_synapse_counts = torch.tensor(
            column_parameters['background_synapse_counts'][self.area_name])
        self.background_synapse_counts = torch.tile(
            background_synapse_counts, (self.num_columns,))

        self.baseline_synaptic_strength = column_parameters[
            'synaptic_strength']['baseline']

        self._compute_recurrent_synapse_counts()
        self._build_recurrent_synaptic_strength_matrix()

    def _compute_recurrent_synapse_counts(self):
        """
        Compute the number of synapses for recurrent connections based on the
        connection probabilities and population sizes.
        """
        log_numerator = np.log(1 - np.array(self.connection_probabilities))
        log_denominator = np.log(1 - 1 / np.array(np.outer(self.population_sizes, self.population_sizes)))

        recurrent_synapse_counts = log_numerator / log_denominator / self.population_sizes[:, None]
        self.recurrent_synapse_counts = torch.tensor(recurrent_synapse_counts, dtype=torch.float32)

    def _build_recurrent_synaptic_strength_matrix(self):
        """
        Build the synaptic strength matrix.
        """
        inhibitory_scaling_factor = torch.tensor([
            -num_excitatory / num_inhibitory
            for num_excitatory, num_inhibitory in zip(
                self.population_sizes[::2], self.population_sizes[1::2])
        ])

        synaptic_strength_column = torch.ones(self.num_populations) * self.baseline_synaptic_strength
        synaptic_strength_column[1::2] = inhibitory_scaling_factor * self.baseline_synaptic_strength

        self.recurrent_synaptic_strength = torch.tile(
            synaptic_strength_column, (self.num_populations, 1)) * self.internal_mask

    def _build_all_weights(self):
        """
        Build recurrent and background from synapse counts and synaptic strengths.
        """
        recurrent_weights = self.recurrent_synapse_counts * self.recurrent_synaptic_strength
        self.register_buffer("recurrent_weights", recurrent_weights)
        background_weights = self.background_synapse_counts * self.baseline_synaptic_strength
        self.register_buffer("background_weights", background_weights)

    def _make_in_ex_masks(self, num_columns):
        """
        Make an internal mask with ones for within column connections
        and an external mask with ones for across column connections.
        """
        column_size = self.num_populations // num_columns  # will likely always be 8

        mask = torch.zeros(self.num_populations, self.num_populations)

        for i in range(0, self.num_populations, column_size):
            idx1 = i
            idx2 = i + column_size
            mask[idx1:idx2, idx1:idx2] = 1.0

        internal_mask = mask
        self.register_buffer("internal_mask", internal_mask)
        external_mask = 1 - mask
        self.register_buffer("external_mask", external_mask)



class Connection(torch.nn.Module):

    """
    Connections allow source activity to influence target areas (between areas) or
    target populations (within areas). Connections include recurrent (intrinsic),
    background, feedforward, feedback, lateral and input.
    """

    def __init__(self, conn_type, source, target, trainable):
        super().__init__()

        self.conn_type = conn_type
        self.source = source
        self.target = target
        self.trainable = trainable

        self.weights = None
        self.mask = None

    def get_name(self):
        """ Returns the unique string specifying the connection."""
        return f'{self.conn_type}_{self.source}_{self.target}'

    def initialize_recurrent_weights(self, area):
        """ Initialize recurrent (i.e. column-intrinsic) weights within the same area."""
        self.weights = torch.nn.Parameter(area.recurrent_weights, requires_grad=self.trainable)

    def initialize_background_weights(self, area):
        """ Initialize background weights within an area."""
        self.weights = torch.nn.Parameter(area.background_weights, requires_grad=self.trainable)

    def initialize_feedforward_weights(self, params, source_area, target_area, std, scale):
        """
        Initialize feedforward weights between source area and target area.
        """
        size_source = source_area.num_columns
        size_target = target_area.num_columns

        init = torch.tensor(params['model']['connection_inits']['feedforward'])
        init *= params['column']['synaptic_strength']['baseline']
        init = torch.tile(init, (size_target, size_source))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        mask = torch.tensor(params['model']['connection_masks']['feedforward'])
        mask = torch.tile(mask, (size_target, size_source))

        weights = rand_weights * mask
        self.weights = torch.nn.Parameter(weights, requires_grad=self.trainable)

    def initialize_feedback_weights(self, params, source_area, target_area, std, scale):
        """
        Initialize feedback weights between source area and target area.
        """
        size_source = source_area.num_columns
        size_target = target_area.num_columns

        init = torch.tensor(params['model']['connection_inits']['feedback'])
        init *= params['column']['synaptic_strength']['baseline']
        init = torch.tile(init, (size_target, size_source))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        mask = torch.tensor(params['model']['connection_masks']['feedback'])
        mask = torch.tile(mask, (size_target, size_source))

        weights = rand_weights * mask
        self.weights = torch.nn.Parameter(weights, requires_grad=self.trainable)

    def initialize_lateral_weights(self, params, area, std, scale):
        """
        Initialize lateral weights within an area.
        """
        size_area = area.num_columns

        init = torch.tensor(params['model']['connection_inits']['lateral'])
        init *= params['column']['synaptic_strength']['baseline']
        init = torch.tile(init, (size_area, size_area))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        mask = torch.tensor(params['model']['connection_masks']['lateral'])
        mask = torch.tile(mask, (size_area, size_area)) * area.external_mask

        weights = rand_weights * mask
        self.weights = torch.nn.Parameter(weights, requires_grad=self.trainable)

    def initialize_input_weights(self, params, size_input, target_area, std, scale):
        """
        Initialize input weights targeting an area.
        """
        size_target_area = target_area.num_columns

        init = torch.tensor(params['model']['connection_inits']['input'])
        init *= params['column']['synaptic_strength']['baseline']
        init = torch.tile(init, (size_target_area, size_input))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        mask = torch.tensor(params['model']['connection_masks']['input'])
        mask = torch.tile(mask, (size_target_area, size_input))

        weights = rand_weights * mask
        self.weights = torch.nn.Parameter(weights, requires_grad=self.trainable)



class BrainNetwork(torch.nn.Module):

    """
    Network class that allows the initialization of BrainArea objects, and Connection objects.
    Contains the dynamics to simulate network activity.
    """

    def __init__(self):
        super().__init__()

        col_params = load_config('../config/column_params.toml')
        model_params = load_config('../config/model_params.toml')

        self.params = {"column": col_params, "model": model_params}

        self.simulator = NetworkSimulator(self, model_params)

        self.num_populations    = None
        self.num_columns        = None
        self.area_slices        = {}
        self.areas              = torch.nn.ModuleDict({})
        self.connections        = torch.nn.ModuleDict({})

        self._initialize_basic_parameters(col_params)

    def _initialize_basic_parameters(self, params):
        """
        Initialize basic parameters that apply for the entire network
        """
        # TODO: Handle background drive as source of input?
        # Basic parameters
        self.register_buffer("background_drive", torch.tensor(params['background_drive'], dtype=torch.float32))
        self.register_buffer("adaptation_strength", torch.tensor(params['adaptation_strength'], dtype=torch.float32))

        # Time constants and membrane resistance
        time_constants = params['time_constants']
        self.register_buffer("synapse_time_constant", torch.tensor(time_constants['synapse'], dtype=torch.float32))
        self.register_buffer("membrane_time_constant", torch.tensor(time_constants['membrane'], dtype=torch.float32))
        self.register_buffer("adapt_time_constant", torch.tensor(time_constants['adaptation'], dtype=torch.float32))
        resistance = time_constants['membrane'] / params['capacitance']
        self.register_buffer("resistance", torch.tensor(resistance, dtype=torch.float32))

    def add_area(self,
                 area_name,
                 size,
                 unique_name=None,
                 intrinsic_trainable=False,
                 background_trainable=False):
        """
        Initialize the specified area and its recurrent and background connections.

        Params:
        area_name (str):                The name of the to-be-modeled area, as specified in the .toml file (e.g. 'v1', 'v2', etc).
        size (int):                     The number of columns of the area.
        unique_name (str):              An optional user-specified name for the area. Useful when the network should contain more
                                        area modules with the same area configurations.
        intrinsic_trainable (bool):     If True, the recurrent, column-intrinsic connections can be updated during training.
        background_trainable (bool):    If True, the background connections can be updated during training.
        """
        area_name = area_name.lower()
        assert area_name in self.params['column']['population_size'], f"Population sizes of '{area_name}' not found in .toml file. "
        assert area_name in self.params['column']['background_synapse_counts'], f"Background synapse counts of '{area_name}' not found in .toml file. "

        if unique_name is None:
            unique_name = area_name

        area = BrainArea(self.params['column'], area_name, size, unique_name)
        self.areas[area_name] = area

        # Add recurrent connectivity and background connectivity as connections
        recurrent_connection = Connection('recurrent', area_name, area_name, intrinsic_trainable)
        recurrent_connection.initialize_recurrent_weights(area)
        self.connections[recurrent_connection.get_name()] = recurrent_connection

        background_connection = Connection('background', area_name, area_name, background_trainable)
        background_connection.initialize_background_weights(area)
        self.connections[background_connection.get_name()] = background_connection

    def add_feedforward_connection(self,
                                   source,
                                   target,
                                   trainable=True,
                                   std=0.1,
                                   scale=1.0):
        source, target = source.lower(), target.lower()
        assert source in self.areas.keys(), f"Source area '{source}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."
        assert target in self.areas.keys(), f"Target area '{target}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."

        source_area = self.areas[source]
        target_area = self.areas[target]

        connection = Connection('feedforward', source, target, trainable)
        connection.initialize_feedforward_weights(self.params, source_area, target_area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_feedback_connection(self,
                                source,
                                target,
                                trainable=True,
                                std=0.1,
                                scale=1.0):
        source, target = source.lower(), target.lower()
        assert source in self.areas.keys(), f"Source area '{source}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."
        assert target in self.areas.keys(), f"Target area '{target}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."

        source_area = self.areas[source]
        target_area = self.areas[target]

        connection = Connection('feedback', source, target, trainable)
        connection.initialize_feedback_weights(self.params, source_area, target_area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_lateral_connection(self,
                               area_name,
                               trainable=True,
                               std=0.1,
                               scale=1.0):
        area_name = area_name.lower()
        assert area_name in self.areas.keys(), f"Area '{area_name}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."
        area = self.areas[area_name]

        connection = Connection('lateral', area_name, area_name, trainable)
        connection.initialize_lateral_weights(self.params, area, std, scale)
        self.connections[connection.get_name()] = connection

    def add_input_connection(self,
                             target_area,
                             input_size,
                             trainable=True,
                             unique_name=None,
                             std=0.1,
                             scale=1.0):
        target_area = target_area.lower()
        assert target_area in self.areas.keys(), f"Area '{target_area}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."
        area = self.areas[target_area]

        input_name = 'input'
        if unique_name is not None:
            input_name = unique_name

        connection = Connection('input', input_name, target_area, trainable)
        connection.initialize_input_weights(self.params, input_size, area, std, scale)
        self.connections[connection.get_name()] = connection

    def finalize(self):
        """
        Finalizes the network after initializing all areas and connections by setting
        the total number of populations and columns, and setting slices to index the
        activity of each area.
        """
        # Population counts: total and slices per area
        self.num_populations = sum(area.num_populations for area in self.areas.values())
        self.num_columns = self.num_populations // 8

        idx = 0
        for area_name, area in self.areas.items():
            self.area_slices[area_name] = slice(idx, idx + area.num_populations)
            idx += area.num_populations

    # TODO: constraining function

    def compute_currents(self, t, firing_rates, ext_input, input_windows):
        """
        For each area, compute the current based on all incoming connections.
        """
        fr_per_area = {area_name: firing_rates[:, area_slice]
                       for area_name, area_slice in self.area_slices.items()}

        activities = {}
        if ext_input is not None:
            for input_name, x in ext_input.items():
                # Present input if t is in input window
                start, end = input_windows[input_name]
                if start <= t < end:
                    activities[input_name] = x
                else:
                    activities[input_name] = torch.zeros_like(x)
        activities.update(fr_per_area)

        currents = {area_name: torch.zeros(firing_rates.shape[0], area.num_populations, device=firing_rates.device)
                    for area_name, area in self.areas.items()}

        for connection in self.connections.values():
            conn_type = connection.conn_type
            if connection.conn_type == 'background':
                batch_size = activities[connection.source].shape[0]
                current = torch.tile(self.background_drive, (batch_size, 1)) * connection.weights
            else:
                source_fr = activities[connection.source]
                current = source_fr @ connection.weights.T
            currents[connection.target] += current * self.synapse_time_constant
            stop = 0

        total_current = torch.cat([currents[name] for name in self.areas], dim=1)
        return total_current

    def dynamics(self, t, state, ext_input, input_windows):
        """
        State dynamics computing the derivative of the membrane potential and adaptation at time t.
        """
        # Unpack the state (membrane, adaptation) and compute firing rate
        mem_adap_split = self.num_populations
        membrane_potential, adaptation = state[:, :mem_adap_split], state[:, mem_adap_split:]

        firing_rate = compute_firing_rate(membrane_potential - adaptation)

        # Compute current
        total_current = self.compute_currents(t, firing_rate, ext_input, input_windows)

        # Compute derivative membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
            total_current * self.resistance) / self.membrane_time_constant
        delta_adaptation = (-adaptation + torch.tile(self.adaptation_strength, (self.num_columns,)) *
                            firing_rate) / self.adapt_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation), dim=1)
        return state

    def diffusion(self, t, state):
        '''
        Diffusion function used by SDE, noise is only applied to membrane potential.
        '''
        g = torch.zeros_like(state)
        n = self.num_populations
        g[:, :n] = 3.0
        return g

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



class NetworkSimulator:

    """
    Handles the simulation of the network activity.
    """

    def __init__(self, network, model_params):

        self.network        = network

        self.dt             = model_params['time_params']['dt']
        self.sim_time       = model_params['time_params']['sim_time']
        self.input_window   = model_params['time_params']['input_window']

    def _infer_batch_size(self, ext_input):
        """
        Infer the batch size from the shape of the input tensors.
        """
        first_input = next(iter(ext_input.values()))
        return first_input.shape[0]

    def _validate_input_shapes(self, ext_input):
        """
        Check if input tensors have the same shape in the first dimension (i.e. same number of input samples.
        """
        batch_sizes = {tensor.shape[0]
                       for tensor in ext_input.values()}
        assert len(batch_sizes) == 1, f"Input tensors have inconsistent batch sizes: {batch_sizes}"

    def _make_input_dict(self, input_var):
        """
        Convert the input variable to a dictionary.
        """
        if not isinstance(input_var, dict):
            input_var = {"input": input_var}
        return input_var

    def _convert_to_tensors(self, ext_input, device):
        """
        Convert each input tensor to a tensor and put them on the specified device.
        """
        return {name: (
                tensor.to(device)
                if torch.is_tensor(tensor)
                else torch.tensor(tensor,dtype=torch.float32, device=device))
                for name, tensor in ext_input.items()}

    def _validate_network_inputs(self, ext_input):
        """
        Check if the inputs have an assigned connection that targets an area and if the
        input tensor has the correct shape in the second dimension (should match with connection.weights)
        """
        input_connections = {conn.source: conn
                         for conn in self.network.connections.values()
                         if conn.conn_type == "input"}

        assert ext_input.keys() == input_connections.keys(), (f"Mismatch found between specified inputs ({list(ext_input.keys())}) "
                                                              f"and specified input connection sources ({list(input_connections.keys())}).")

        for source, connection in input_connections.items():
            assert ext_input[source].shape[1] == connection.weights.shape[1], (f"Input tensor '{source}' shape (x, {ext_input[source].shape[1]}) "
                                                                               f"does not match with input weights shape (x, {connection.weights.shape[1]}).")

    def _validate_input_windows(self, input_window):
        """
        Check if input window does not exceed the simulation time.
        """
        for window_i, window_tuple in input_window.items():
            start = window_tuple[0]
            end = window_tuple[1]
            assert start <= self.sim_time and end <= self.sim_time, (f"The input time window ({start}s, {end}s) "
                                                                     f"exceeds total simulation time ({self.sim_time}s)")

    def _prepare_input_for_sim(self, ext_input, input_window, device):
        """
        Prepare the input samples and the input time window for simulation:
        they need to be formatted as dictionaries, inputs as tensors and on the device,
        check if compatible with network architecture.
        """
        if ext_input is None:
            return None, None, 1

        if input_window is None:
            input_window = self.input_window

        ext_input = self._make_input_dict(ext_input)
        input_window = self._make_input_dict(input_window)

        ext_input = self._convert_to_tensors(ext_input, device)

        self._validate_input_shapes(ext_input)
        self._validate_network_inputs(ext_input)
        self._validate_input_windows(input_window)

        assert ext_input.keys() == input_window.keys(), (f"Mismatch found between dictionaries of inputs ({list(ext_input.keys())}) "
                                                         f"and their time windows ({list(input_window.keys())}).")

        batch_size = self._infer_batch_size(ext_input)
        return ext_input, input_window, batch_size

    def _init_state(self, batch_size, device):
        """
        Set initial state of network as zeros; size = (batch_size, total_nr_populations * 2 state variables).
        """
        initial_state = torch.zeros(batch_size, self.network.num_populations * 2)
        return initial_state.to(device)

    def run(self, ext_input, input_window, adjoint, stochastic, device):
        """
        Runs the network simulation.
        """
        ext_input, input_window, batch_size = self._prepare_input_for_sim(ext_input, input_window, device)

        self.network = self.network.to(device)
        self.network.finalize()

        time_vec = torch.arange(0, self.sim_time, self.dt, device=device)
        initial_state = self._init_state(batch_size, device)
        sim_wrapper = NetworkOdeWrapper(self.network, ext_input, input_window)

        if not adjoint and not stochastic:
            return odeint(sim_wrapper, initial_state, time_vec)

        elif adjoint and not stochastic:
            return odeint_adjoint(sim_wrapper, initial_state, time_vec)

        elif not adjoint and stochastic:
            return sdeint(sim_wrapper, initial_state, time_vec,
                            names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

        elif adjoint and stochastic:
            return sdeint_adjoint(sim_wrapper, initial_state, time_vec,
                                    names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')




class NetworkOdeWrapper(torch.nn.Module):

    """
    Sets the external input with initialization and gets called by odeint/sdeint.
    """

    noise_type = "diagonal"  # sde params
    sde_type = "ito"

    def __init__(self, network, ext_input, input_windows):
        super().__init__()

        self.network = network
        self.ext_input = ext_input
        self.input_windows = input_windows

    def forward(self, t, state):
        return self.network.dynamics(
            t,
            state,
            self.ext_input,
            self.input_windows)

    def diffusion(self, t, state):
        return self.network.diffusion(
            t,
            state)



class NetworkAnalyzer:

    def __init__(self, network):
        self.network = network
        self.layers = ['L23e, L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

    def get_firing_rates(self, raw_state, area=None, return_as_np_array=True):
        """
        Compute the firing rate from the raw state (= [membrane_potential, adaptation])
        Return as np.array unless specified otherwise.
        """
        # TODO: refine (layer indices?)
        split = self.network.num_populations
        firing_rates = compute_firing_rate(raw_state[:, :, :split] - raw_state[:, :, split:(split * 2)])

        if area is not None:
            area_slices = self.network.area_slices[area]
            firing_rates = firing_rates[:, :, area_slices]

        if return_as_np_array:
            return firing_rates.detach().cpu().numpy()
        return firing_rates

    def plot_firing_rates(self, firing_rates):
        """
        Plot firing rates, separately for each sample
        """
        # TODO: refine (more plotting options)
        for i in range(firing_rates.shape[1]):
            for j in range(firing_rates.shape[-1]):
                plt.plot(firing_rates[:, i, j])
            plt.show()

