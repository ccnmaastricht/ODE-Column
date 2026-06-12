import torch
import numpy as np
from scipy.linalg import block_diag
from src.utils import compute_firing_rate



class BrainArea(torch.nn.Module):

    """
    ...
    """

    def __init__(self, params, area_name, num_columns, unique_name):
        super().__init__()

        self.num_columns = num_columns
        self.area_name = area_name
        self.name = unique_name

        self._initialize_population_parameters(params)
        self._initialize_connection_probabilities(params)
        self._initialize_synapses(params)
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



class Projection(torch.nn.Module):

    """
    ...
    """

    def __init__(self,
                 type,
                 source,
                 target,
                 trainable,
                 params,
                 init=None,
                 mask=None,
                 std=None,
                 scale=None):
        super().__init__()

        self.type = type
        self.source_obj = source
        self.target_obj = target
        self.source = source.name
        self.target = target.name
        self.trainable = trainable

        # TODO: should also set mask

        if type == 'recurrent':
            weights = self._initialize_recurrent_projection(params)
        elif type == 'background':
            weights = self._initialize_background_projection(params)
        elif type == 'feedforward':
            weights = self._initialize_feedforward_weights(params, init, mask, std, scale)
        elif type == 'feedback':
            weights = self._initialize_feedback_weights(params, init, mask, std, scale)
        elif type == 'lateral':
            weights = self._initialize_lateral_weights(params, init, mask, std, scale)

        if trainable:
            self.weights = torch.nn.Parameter(weights, requires_grad=True)
        else:
            self.weights = torch.nn.Parameter(weights, requires_grad=False)

    def get_name(self):
        """
        Returns the unique string specifying the projection.
        """
        return f'{self.type}_{self.source}_{self.target}'

    def _initialize_recurrent_projection(self, params):
        return self.source_obj.recurrent_weights

    def _initialize_background_projection(self, params):
        return self.source_obj.background_weights

    def _initialize_feedforward_weights(self, params, init, mask, std, scale):
        """
        Initialize feedforward weights between source area and target area.
        """
        size_source = self.source_obj.num_columns
        size_target = self.target_obj.num_columns

        if init is None:
            init = torch.tensor(params['connection_inits']['feedforward'])
        init *= params['synaptic_strength']['baseline']
        init = torch.tile(init, (size_target, size_source))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        if mask is None:
            mask = torch.tensor(params['connection_masks']['feedforward'])
        mask = torch.tile(mask, (size_target, size_source))

        weights = rand_weights * mask
        return weights

    def _initialize_feedback_weights(self, params, init, mask, std, scale):
        """
        Initialize feedback weights between source area and target area.
        """
        size_source = self.source_obj.num_columns
        size_target = self.target_obj.num_columns

        if init is None:
            init = torch.tensor(params['connection_inits']['feedback'])
        init *= params['synaptic_strength']['baseline']
        init = torch.tile(init, (size_target, size_source))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        if mask is None:
            mask = torch.tensor(params['connection_masks']['feedback'])
        mask = torch.tile(mask, (size_target, size_source))

        weights = rand_weights * mask
        return weights

    def _initialize_lateral_weights(self, params, init, mask, std, scale):
        """
        Initialize lateral weights
        """
        size_area = self.source_obj.num_columns

        if init is None:
            init = torch.tensor(params['connection_inits']['lateral'])
        init *= params['synaptic_strength']['baseline']
        init = torch.tile(init, (size_area, size_area))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        if mask is None:
            mask = torch.tensor(params['connection_masks']['lateral'])
        mask = torch.tile(mask, (size_area, size_area)) * self.source_obj.external_mask  # This is unique to the lateral weights!

        weights = rand_weights * mask
        return weights



class BrainNetwork(torch.nn.Module):

    """
    ...
    """

    def __init__(self, params):
        super().__init__()

        self.noise_type = "diagonal"  # sde params
        self.sde_type = "ito"

        self.params = params

        self.num_populations    = None
        self.num_columns        = None
        self.area_slices        = {}
        self.areas              = torch.nn.ModuleDict({})
        self.projections        = torch.nn.ModuleDict({})

        self._initialize_basic_parameters(params)

    def _initialize_basic_parameters(self, params):
        """
        Initialize basic parameters that apply for the entire network
        """
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
        Initialize the specified area and its recurrent and background projections.

        Params:
        area_name (str):                The name of the to-be-modeled area, as specified in the .toml file (e.g. 'v1', 'v2', etc).
        size (int):                     The number of columns of the area.
        unique_name (str):              An optional user-specified name for the area. Useful when the network should contain more
                                        area modules with the same area configurations.
        intrinsic_trainable (bool):     If True, the recurrent, column-intrinsic connections can be updated during training.
        background_trainable (bool):    If True, the background connections can be updated during training.
        """
        area_name = area_name.lower()
        assert area_name in self.params['population_size'], f"Population sizes of '{area_name}' not found in .toml file. "
        assert area_name in self.params['background_synapse_counts'], f"Background synapse counts of '{area_name}' not found in .toml file. "

        if unique_name is None:
            unique_name = area_name

        area = BrainArea(self.params, area_name, size, unique_name)
        self.areas[area_name] = area

        # Add recurrent connectivity and background connectivity as projections
        recurrent_projection = Projection('recurrent', area, area, intrinsic_trainable, self.params)
        self.projections[recurrent_projection.get_name()] = recurrent_projection
        background_projection = Projection('background', area, area, background_trainable, self.params)
        self.projections[background_projection.get_name()] = background_projection

    def add_projection(self,
                       source,
                       target,
                       type,
                       trainable=True,
                       mask=None,
                       init=None,
                       std=0.1,
                       scale=1.0):
        """
        Initialize the specified projection between the source and target area.

        Params:
        source (str):       The source area of the projection.
        target (str):       The target area of the projection.
        type (str):         The projection type; feedforward, feedback or lateral.
        trainable (bool):   If True, the projection weights can be adjusted during training.
        mask (tensor):      Optional user-specified mask for the projection.
        init (tensor):      Optional user-specified initialization for the projection.
        std (float):        The standard deviation for the weight initialization of the projection.
        scale (float):      The scale of the initialization of the projection.
        """
        source, target = source.lower(), target.lower()
        assert source in self.areas.keys(), f"Source area '{source}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."
        assert target in self.areas.keys(), f"Target area '{target}' is not yet initialized. Please use BrainNetwork.add_area(name, size)."

        source_area = self.areas[source]
        target_area = self.areas[target]

        assert type in ['feedforward', 'feedback', 'lateral'], (f"The projection type '{type}' does not exist. Acceptable types are 'feedforward', 'feedback' and 'lateral'. "
                                                                f"If you want to establish recurrent or background projections, note that these are established automatically with BrainNetwork.add_area(name, size). ")
        if type == 'feedforward':
            assert source != target, f"Feedforward projections can only be established between two non-identical areas."
        elif type == 'feedback':
            assert source != target, f"Feedback projections can only be established between two non-identical areas."
        elif type == 'lateral':
            assert source == target, f"Lateral projections can only be established within the same area (source_area == target_area)."

        projection = Projection(type, source_area, target_area, trainable, self.params, mask, init, std, scale)
        self.projections[projection.get_name()] = projection

    def finalize(self, dt, sim_time, batch_size=1):
        """
        Finalizes the network after initializing all areas and projections.
        Returns the initial state for all modelled neuronal populations and
        the time vector.
        """
        # Time vector
        time_steps = int(sim_time / dt)
        time_vec = torch.linspace(0., time_steps * dt, time_steps)

        # Population counts: total and slices per area
        self.num_populations = sum(area.num_populations for area in self.areas.values())
        self.num_columns = self.num_populations // 8

        idx = 0
        for area_name, area in self.areas.items():
            self.area_slices[area_name] = slice(idx, idx + area.num_populations)
            idx += area.num_populations

        # Initial state for simulation
        initial_state = torch.zeros(batch_size, self.num_populations * 2)  # *2 state variables
        return initial_state, time_vec

    # TODO: constraining function

    def compute_currents(self, fr_per_area):
        # TODO: introduce external input

        currents = {area_name: torch.zeros_like(fr_per_area[area_name])
                    for area_name in self.areas}

        for projection in self.projections.values():
            type = projection.type
            if projection.type == 'background':
                batch_size = fr_per_area[projection.source].shape[0]
                current = torch.tile(self.background_drive, (batch_size, 1)) * projection.weights
                currents[projection.target] += current
            else:
                source_fr = fr_per_area[projection.source]
                current = source_fr @ projection.weights.T
                currents[projection.target] += current
            stop = 0

        for area_name, area in self.areas.items():
            currents[area_name] *= self.synapse_time_constant

        total_current = torch.cat([currents[name] for name in self.areas], dim=1)
        return total_current

    def forward(self, t, state):
        """
        State dynamics computing the derivative of the membrane potential and adaptation at time t.
        """
        # Unpack the state (membrane, adaptation) and compute firing rate
        mem_adap_split = self.num_populations
        membrane_potential, adaptation = state[:, :mem_adap_split], state[:, mem_adap_split:]

        firing_rate = compute_firing_rate(membrane_potential - adaptation)

        # Compute current
        fr_per_area = {area_name: firing_rate[:, area_slice]
                       for area_name, area_slice in self.area_slices.items()}
        total_current = self.compute_currents(fr_per_area)

        # Compute derivative membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
            total_current * self.resistance) / self.membrane_time_constant
        delta_adaptation = (-adaptation + torch.tile(self.adaptation_strength, (self.num_columns,)) *
                            firing_rate) / self.adapt_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation), dim=1)
        return state


# TODO: NetworkWrapperODE()
