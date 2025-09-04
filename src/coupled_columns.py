import numpy as np
from scipy.linalg import block_diag
import torch.nn as nn

from src.utils import *


class ColumnArea(torch.nn.Module):

    def __init__(self, column_parameters, area, num_columns, small_network=False):
        super().__init__()

        self.num_columns = num_columns
        self.area = area.lower()

        self._intialize_basic_parameters(column_parameters)
        self._initilize_population_parameters(column_parameters, small_network)
        self._initialize_connection_probabilities(column_parameters)
        self._initialize_synapses(column_parameters, small_network)

        self._build_all_weights()

    def _intialize_basic_parameters(self, column_parameters):
        """
        Initialize basic parameters for the columns.
        """
        # Basic parameters
        self.register_buffer("background_drive", torch.tensor(column_parameters['background_drive'], dtype=torch.float32)) # device
        self.register_buffer("adaptation_strength", torch.tensor(column_parameters['adaptation_strength'], dtype=torch.float32))  # device

        # Time constants and membrane resistance
        self.time_constants = column_parameters['time_constants']
        self.register_buffer("synapse_time_constant", torch.tensor(self.time_constants['synapse'], dtype=torch.float32)) # device
        self.register_buffer("membrane_time_constant", torch.tensor(self.time_constants['membrane'], dtype=torch.float32))  # device
        self.register_buffer("adapt_time_constant", torch.tensor(self.time_constants['adaptation'], dtype=torch.float32))  # device
        resistance = self.time_constants['membrane'] / column_parameters['capacitance']
        self.register_buffer("resistance", torch.tensor(resistance, dtype=torch.float32))  # device

    def _initilize_population_parameters(self, column_parameters, small_network):
        """
        Initialize the population sizes for the columns.
        """
        self.population_sizes = np.array(
            column_parameters['population_size'][self.area])
        self.population_sizes = np.tile(self.population_sizes, self.num_columns)
        if small_network:  # for XOR and WTA
            self.population_sizes = self.population_sizes / self.num_columns

        self.num_populations = len(self.population_sizes)
        self.adaptation_strength = torch.tile(self.adaptation_strength, (self.num_columns,))

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

    def _initialize_synapses(self, column_parameters, small_network):
        """
        Initialize the synapse counts and synaptic strengths for the columns.
        """
        if small_network:  # for training XOR and WTA
            self.background_synapse_counts = torch.tensor([2510, 2510, 2510, 2510, 2510, 2510, 2510, 2510])
        else:  # for training larger networks
            self.background_synapse_counts = torch.tensor(
                column_parameters['synapse_counts']['background'])

        self.feedforward_synapse_counts = torch.tensor(
            column_parameters['synapse_counts']['feedforward'])

        self.background_synapse_counts = torch.tile(
            self.background_synapse_counts, (self.num_columns,))
        self.feedforward_synapse_counts = torch.tile(
            self.feedforward_synapse_counts, (self.num_columns,))

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
        Build recurrent, background, external, and feedforward weights from synapse counts and synaptic strengths.
        """
        self.recurrent_weights = self.recurrent_synapse_counts * self.recurrent_synaptic_strength
        background_weights = self.background_synapse_counts * self.baseline_synaptic_strength # device
        self.register_buffer("background_weights", background_weights) # device
        self.feedforward_weights = self.feedforward_synapse_counts * self.baseline_synaptic_strength

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

        self.internal_mask = mask
        self.external_mask = 1 - mask


class ColumnAreaWTA(ColumnArea):

    '''
    Two columns between which lateral connectivity can be learned to
    exhibit winner-take-all dynamics in perceptual decision-making.
    Connections from L2/3e in column A to L2/3i in column B and L2/3e
    self-excitation connections.
    '''

    def __init__(self, column_parameters, area):
        super().__init__(column_parameters, area, 2, small_network=True)

        self.noise_type = "scalar"  # sde params
        self.sde_type = "ito"

        self._make_lat_in_mask()

        self._initialize_lat_in_weights()
        self._initialize_output_weights()

    def _make_lat_in_mask(self):
        '''
        Mask to select lateral inhibition connections between 2/3 layers.
        '''
        lat_in_mask = torch.zeros((self.num_populations, self.num_populations))
        lat_in_mask[1, 8], lat_in_mask[9, 0] = 1.0, 1.0  # lateral inhibition
        lat_in_mask[0, 0], lat_in_mask[8, 8] = 1.0, 1.0  # self excitation
        self.lat_in_mask = lat_in_mask

    def _initialize_lat_in_weights(self):
        '''
        Weights consist of inner connections (8x8) for both columns and external
        connections between columns, i.e. lateral connections. Only the mask-selected
        connections are learnable.
        '''

        original_weights = self.recurrent_weights.clone().detach()
        std_W = 0.0001
        rand_weights = abs(torch.normal(mean=original_weights, std=std_W))
        lat_in_weights = (rand_weights * (self.lat_in_mask * self.external_mask)) + original_weights
        self.recurrent_weights = nn.Parameter(lat_in_weights, requires_grad=True)

    def _initialize_output_weights(self):
        output_weights = torch.tensor([1.0000, 0.0000, 0.0000, 0.0000,
                                       0.0000, 0.0000, 0.0000, 0.0000])
        self.output_weights = output_weights

    def set_time_vec(self, time_vec):
        '''
        Set the time_vec as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.time_vec = time_vec

    def set_stim(self, stim):
        '''
        Set the stimulus as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.stim = stim

    def forward(self, t, state):
        '''
        State dynamics the ODE uses
        '''
        # Prepare the state (membrane, adaptation, firing rate)
        state = state.squeeze(0)  # lose extra dim
        mem_adap_split = len(state) // 3
        adap_rate_split = len(state) // 3 * 2
        membrane_potential, adaptation = state[:mem_adap_split], state[mem_adap_split:adap_rate_split]

        # Compute new firing rate from membrane and adaptation
        firing_rate = compute_firing_rate(membrane_potential - adaptation)

        # Get current stimulus (ff rate) based on current time t and the time vector time_vec
        feedforward_rate = torch_interp(t, self.time_vec, self.stim)

        # Compute current current
        feedforward_current = self.feedforward_weights * feedforward_rate       # stimulus feedforward input
        background_current = self.background_weights * self.background_drive    # background input
        recurrent_current = torch.matmul(self.recurrent_weights, firing_rate)   # recurrent input

        total_current = (feedforward_current + background_current + recurrent_current) * self.synapse_time_constant

        # State derivatives
        delta_membrane_potential = (-membrane_potential +
            total_current * self.resistance) / self.membrane_time_constant
        delta_adaptation = (-adaptation + self.adaptation_strength *
                            firing_rate) / self.adapt_time_constant
        prev_firing_rate = state[adap_rate_split:]
        delta_firing_rate = (-prev_firing_rate + firing_rate) / self.synapse_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation, delta_firing_rate))

        return state.unsqueeze(0)

    def diffusion(self, t, y):
        '''
        Diffusion function used by SDE. Noise is added to the
        membrane potential only
        '''
        noise_std = 100.0
        g = torch.zeros_like(y)
        split_mem = (len(y[0]) // 3)
        g[:split_mem] = noise_std
        g = g.unsqueeze(dim=-1)
        return g




class ColumnNetworkXOR(torch.nn.Module):

    '''
    Concatenates a number of areas (each consisting of a number
    of columns) to form a larger network, specifically to train XOR.
    Ideally, XOR could be trained on ColumnNetwork (below), but
    this class is currently used for parity training.
    '''

    def __init__(self, column_parameters, network_dict):
        super().__init__()

        self.noise_type = "scalar"  # sde params
        self.sde_type = "ito"

        self._initialize_areas(column_parameters, network_dict)

        self.network_as_area = ColumnArea(column_parameters, 'mt', sum(network_dict['nr_columns_per_area']))
        self.nr_input_units = network_dict['nr_input_units']
        self.nr_columns_per_area = network_dict['nr_columns_per_area']

        self._initialize_lateral_weights()
        self._initialize_ff_masks()
        self._initialize_feedforward_weights()

    def _initialize_areas(self, column_parameters, network_dict):
        '''
        Initialize each area as a ColumnArea object.
        '''
        self.areas = nn.ModuleDict({})
        for area_idx in range(network_dict['nr_areas']):

            area_name = network_dict['areas'][area_idx]
            num_columns = network_dict['nr_columns_per_area'][area_idx]

            area = ColumnArea(column_parameters, area_name, num_columns, small_network=True)
            self.areas[str(area_idx)] = area

    def _initialize_lateral_weights(self):
        '''
        Sets external recurrent weights of all areas to zero, to make sure
        all lateral connectivity is removed.
        '''
        for idx, area in self.areas.items():
            recurr_weights = area.recurrent_weights
            area.recurrent_weights = recurr_weights * area.internal_mask  # set any existing external connectivity to zero

    def _initialize_ff_masks(self):
        '''
        Specify from which population the feedforward flow comes
        (source) and which population it targets (target).
        '''
        # Source of ff is L2/3e
        ff_source_mask = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        self.ff_source_mask = ff_source_mask

        # Target of ff is L4e and L4i
        ff_target_mask = torch.tensor([0., 0., 1., 1., 0., 0., 0., 0.])
        self.ff_target_mask = ff_target_mask

    def _initialize_feedforward_weights(self):
        '''
        Initialize the feedforward weights as learnable weights.
        '''
        feedforward_target_weights = nn.ModuleDict({})

        for area_idx, area in self.areas.items():

            feedforward_target_weights[area_idx] = nn.ParameterList()

            if area_idx == '0':   # if first area, check how many external inputs it receives
                nr_ff_weights = self.nr_input_units
            else:               # for subsequent areas, check how many inputs from previous area
                key_prev_area = str(int(area_idx)-1)
                nr_ff_weights = self.areas[key_prev_area].num_columns

            # Initialize random feedforward weights
            original_target_weights = area.feedforward_weights.clone().detach()
            std_W = 0.0001

            for i in range(nr_ff_weights):

                rand_weights_target = abs(torch.normal(mean=original_target_weights, std=std_W))
                rand_weights_target = rand_weights_target * torch.tile(self.ff_target_mask, (area.num_columns,))
                ff_weights_target = nn.Parameter(rand_weights_target, requires_grad=True)
                feedforward_target_weights[area_idx].append(ff_weights_target)

        self.feedforward_target_weights = feedforward_target_weights

    def set_time_vec(self, time_vec):
        '''
        Set the time_vec as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.time_vec = time_vec

    def set_stim(self, stim):
        '''
        Set the stimulus as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.stim = stim

    def partition_firing_rates(self, firing_rate):
        '''
        Organizes the firing rates into a dict of separate areas.
        This allows easy access to previous area's firing rates.
        '''
        fr_per_area = {}
        idx = 0
        for area_idx, area in self.areas.items():
            fr_area = firing_rate[idx : idx + area.num_populations]
            fr_area_reshape = fr_area.reshape(area.num_columns, 8)
            fr_per_area[area_idx] = fr_area_reshape
            idx = idx + area.num_populations
        return fr_per_area

    def compute_currents(self, ext_ff_rate, fr_per_area, t):
        '''
        Compute the current for each area separately. The total current
        consists of feedforward current (stimulus-driven and/or from other
        brain areas), background current and recurrent current.
        '''
        total_current = torch.Tensor()

        for area_idx, area in self.areas.items():

            # Compute feedforward current of each area, based on
            # area=0: external input or area>0: the previous area's firing rate
            feedforward_current = torch.zeros(area.num_populations)

            for ff_idx, ff_target_weight in enumerate(self.feedforward_target_weights[area_idx]):
            # Multiply each input with each corresponding ff weights
                if area_idx == '0':  # first area gets external input
                    feedforward_current += ext_ff_rate[ff_idx] * ff_target_weight

                elif area_idx > '0':  # subsequent areas receive previous area's firing rate
                    key_prev_area = str(int(area_idx) - 1)
                    prev_area_fr = fr_per_area[key_prev_area][ff_idx] * self.ff_source_mask
                    prev_area_fr_sum = torch.sum(prev_area_fr)
                    prev_area_fr_sum *= 10.  # pump up firing rates
                    feedforward_current += prev_area_fr_sum * ff_target_weight

            # Background and recurrent current
            background_current = area.background_weights * area.background_drive
            recurrent_current = torch.matmul(area.recurrent_weights, fr_per_area[area_idx].flatten())

            # Total current of this area
            total_current_area = (feedforward_current + background_current + recurrent_current) * area.synapse_time_constant

            total_current = torch.cat((total_current, total_current_area), dim=0)
        return total_current

    def forward(self, t, state):
        '''
        State dynamics updating the membrane potential and adaptation;
        ODE should learn these dynamics and update the weights accordingly.
        '''

        # Prepare the state (membrane, adaptation, firing rate)
        state = state.squeeze(0)  # lose extra dim
        mem_adap_split = len(state) // 3
        adap_rate_split = len(state) // 3 * 2
        membrane_potential, adaptation = state[:mem_adap_split], state[mem_adap_split:adap_rate_split]

        firing_rate = compute_firing_rate(membrane_potential - adaptation)

        # Partition firing rate per area
        fr_per_area = self.partition_firing_rates(firing_rate)

        # Get current stimulus (external ff rate) based on current time t and the time vector time_vec
        ext_ff_rate = torch_interp(t, self.time_vec, self.stim)

        # Compute input current
        total_current = self.compute_currents(ext_ff_rate, fr_per_area, t)

        # Compute derivative membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
            total_current * self.network_as_area.resistance) / self.network_as_area.membrane_time_constant
        delta_adaptation = (-adaptation + self.network_as_area.adaptation_strength *
                            firing_rate) / self.network_as_area.adapt_time_constant

        # Compute derivative firing rate
        prev_firing_rate = state[adap_rate_split:]
        delta_firing_rate = (-prev_firing_rate + firing_rate) / self.network_as_area.synapse_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation, delta_firing_rate))

        return state.unsqueeze(0)

    def diffusion(self, t, y):
        '''
        Diffusion function used by SDE. Noise is added to the membrane
        potential only.
        '''
        noise_std = 10.0
        g = torch.zeros_like(y)
        split = (len(y[0]) // 3)
        g[:, :split] = noise_std  # membrane gets noise
        g = g.unsqueeze(dim=-1)
        return g



class ColumnNetwork(torch.nn.Module):

    '''
    Concatenates a number of areas (each consisting of a number
    of columns) to form a larger network. Within an area, only
    lateral connections between columns are allowed. Across areas
    only feedforward connections are allowed.
    '''

    def __init__(self, model_parameters, network_dict, device):
        super().__init__()

        self.noise_type = "scalar"  # sde params
        self.sde_type = "ito"

        self.device = device
        self._initialize_areas(model_parameters, network_dict)

        self.network_as_area = ColumnArea(model_parameters, 'mt', sum(network_dict['nr_columns_per_area']))
        self.nr_input_units = network_dict['nr_input_units']
        self.nr_columns_per_area = network_dict['nr_columns_per_area']
        self.nr_areas = network_dict['nr_areas']

        self._initialize_masks(model_parameters)
        self._initialize_lateral_weights(model_parameters)
        self._initialize_feedforward_weights(model_parameters)
        self._initialize_intput_weights(model_parameters)
        self._initialize_output_weights(model_parameters)

    def _initialize_areas(self, model_parameters, network_dict):
        '''
        Initialize the areas as ColumnArea objects.
        '''
        self.areas = nn.ModuleDict({})
        for area_idx in range(network_dict['nr_areas']):

            area_name = network_dict['areas'][area_idx]
            num_columns = network_dict['nr_columns_per_area'][area_idx]

            area = ColumnArea(model_parameters, area_name, num_columns)
            area = area.to(self.device)
            self.areas[str(area_idx)] = area

    def _initialize_masks(self, model_parameters):
        '''
        Binary masks to select only legal connections between populations,
        based on the nature of the connection.
        '''
        masks = model_parameters['connection_masks']

        self.input_mask = torch.tensor(masks['input'])
        self.output_mask = torch.tensor(masks['output'])
        self.feedforward_mask = torch.tensor(masks['feedforward'])
        self.feedback_mask = torch.tensor(masks['feedback'])
        self.lateral_mask = torch.tensor(masks['lateral'])

    def make_mask_fan_in(self, mask, num_target_blocks, num_source_blocks):
        '''
        Alter the connectivity mask to ensure fan-in connectivity
        instead of fully connected.
        '''
        size_target, size_source = mask.shape
        fan_connectivity = torch.zeros_like(mask)

        fan_target = size_target // num_target_blocks
        fan_source = size_source // num_source_blocks

        for i, j in zip(range(0, size_target, fan_target), range(0, size_source, fan_source)):
            fan_connectivity[i:i + fan_target, j:j + fan_source] = 1.0

        return mask * fan_connectivity

    def make_mask_fan_in_random(self, mask, source_is_input=False):
        '''
        Alter the connectivity mask to ensure fan-in connectivity
        instead of fully connected. Connections between source and
        target columns are randomly initiated.
        '''
        size_target, size_source = mask.shape

        if source_is_input:
            n_pops_per_course_col = 1
            nr_sources_target_receives = 2
        else:
            n_pops_per_course_col = 8
            nr_sources_target_receives = 3

        n_target_cols = size_target // 8
        n_source_cols = size_source // n_pops_per_course_col

        fan_connectivity = torch.zeros_like(mask)

        # Step 1: assign each source col to a random target col
        assignments = torch.randint(0, n_target_cols, (n_source_cols,), generator=None)
        for scol, tcol in enumerate(assignments):
            t_idx = slice(tcol * 8, (tcol + 1) * 8)
            s_idx = slice(scol * n_pops_per_course_col, (scol + 1) * n_pops_per_course_col)
            fan_connectivity[t_idx, s_idx] = 1.0

        # Step 2: fill the remaining fan-in slots per target
        for tcol in range(n_target_cols):
            already = (fan_connectivity[
                       tcol * 8:(tcol + 1) * 8
                       ].sum(0).view(n_source_cols, n_pops_per_course_col).sum(1) > 0).nonzero().flatten()

            # Each target column receives 2 source columns
            need = nr_sources_target_receives - len(already)
            if need > 0:
                choices = torch.tensor(
                    [c for c in range(n_source_cols) if c not in already],
                    dtype=torch.long
                )
                chosen = choices[torch.randperm(len(choices), generator=None)[:need]]
                for scol in chosen:
                    t_idx = slice(tcol * 8, (tcol + 1) * 8)
                    s_idx = slice(scol * n_pops_per_course_col, (scol + 1) * n_pops_per_course_col)
                    fan_connectivity[t_idx, s_idx] = 1.0

        return mask * fan_connectivity

    def _initialize_intput_weights(self, model_parameters):
        '''
        Initialize learnable input weights to weight the input going into the first area.
        '''
        first_area = self.areas['0']

        size_source = self.nr_input_units
        size_target = first_area.num_columns

        input_init = torch.tensor(model_parameters['connection_inits']['input'])
        input_init = torch.tile(input_init, (size_target, size_source))

        std_W = 3.0
        rand_input_weights = abs(torch.normal(mean=input_init, std=std_W)) * self.feedforward_scale
        rand_input_weights *= 0.8

        input_mask = torch.tile(self.input_mask, (size_target, size_source))
        input_mask = self.make_mask_fan_in(input_mask, 2, 2)
        input_mask[0:16, :] = input_mask[32:48, :]
        input_mask[32:48, :] = input_mask[16:32, :]
        # input_mask = self.make_mask_fan_in_random(input_mask, source_is_input=True)  # use this fan-in function to initialize random connections
        first_area.input_mask = input_mask

        rand_input_weights = rand_input_weights * input_mask
        first_area.input_weights = nn.Parameter(rand_input_weights, requires_grad=True)

    def _initialize_feedforward_weights(self, model_parameters):
        '''
        Initialize the feedforward weights between each set of areas as learnable weights.
        Attach the weights to the target area.
        '''

        self.feedforward_scale = 1.0

        for area_idx, area in self.areas.items():
            if area_idx != '0':  # first area gets no ff input

                size_source = self.nr_columns_per_area[int(area_idx) - 1]
                size_target = self.nr_columns_per_area[int(area_idx)]

                ff_init = torch.tensor(model_parameters['connection_inits']['feedforward'])
                ff_init = torch.tile(ff_init, (size_target, size_source))

                std_W = 1.0
                rand_ff_weights = abs(torch.normal(mean=ff_init, std=std_W)) * self.feedforward_scale
                rand_ff_weights *= 4.0

                ff_mask = torch.tile(self.feedforward_mask, (size_target, size_source))
                if size_target > 1:  # no fan-in connectivity for area with only one column
                    ff_mask = self.make_mask_fan_in(ff_mask, 2, 2)
                    # ff_mask = self.make_mask_fan_in(ff_mask)  # use this fan-in function to initialize random connections
                area.feedforward_mask = ff_mask

                rand_ff_weights = rand_ff_weights * ff_mask
                area.feedforward_weights = nn.Parameter(rand_ff_weights, requires_grad=True)

    def _initialize_lateral_weights(self, model_parameters):
        '''
        Random initialization of lateral weights between columns,
        for each area separately.
        '''

        self.lateral_scale = 1.0

        for area_idx, area in self.areas.items():
            area.inner_weights = area.recurrent_weights * area.internal_mask  # set any existing external connectivity to zero
            area.inner_weights = area.inner_weights.to(self.device)

            area.inner_values = area.inner_weights[area.internal_mask.bool()].clone()
            area.inner_indices = area.internal_mask.nonzero(as_tuple=False).T  # shape: [2, num_connections]
            area.inner_size = tuple(area.internal_mask.shape)  # needed for sparse shape

            # Reshape weight initialization
            lateral_init = torch.tensor(model_parameters['connection_inits']['lateral'])
            lateral_init = torch.tile(lateral_init, (area.num_columns, area.num_columns))

            # Reshape mask and store in area
            lateral_mask = torch.tile(self.lateral_mask, (area.num_columns, area.num_columns)) * area.external_mask
            area.lateral_mask = lateral_mask

            # Randomly initialize lateral weights and store in area as learnable param
            std_W = 0.01
            rand_weights = torch.normal(mean=lateral_init, std=std_W) * self.lateral_scale
            rand_weights *= 0.01  # initialize small lateral weights - let them be learned from scratch
            rand_weights *= area.lateral_mask
            rand_weights *= area.external_mask
            rand_weights = rand_weights.to(self.device)

            if area.num_columns > 1:
                area.lateral_weights = nn.Parameter(rand_weights, requires_grad=True)
            else:  # area with only one column should have no trainable lateral connections
                area.lateral_weights = nn.Parameter(rand_weights, requires_grad=False)

    def _initialize_output_weights(self, model_parameters):
        '''
        Initialize learnable output weights that can be used to read out
        the firing rates of the final column as a means of classification.
        '''
        self.output_scale = 1.0

        key_last_area = str(len(self.areas)-1)
        size_source = self.areas[key_last_area].num_columns

        output_init = torch.tensor(model_parameters['connection_inits']['output'])
        output_init = torch.tile(output_init, (size_source,))

        std_W = 0.001
        rand_output_weights = abs(torch.normal(mean=output_init, std=std_W))
        rand_output_weights *= rand_output_weights * torch.tile(self.output_mask, (size_source,))
        rand_output_weights *= self.output_scale

        self.output_weights = nn.Parameter(rand_output_weights, requires_grad=True)

    def set_time_vec(self, time_vec):
        '''
        Set the time_vec as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.time_vec = time_vec

    def set_stim(self, stim):
        '''
        Set the stimulus as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.stim = stim

    def partition_firing_rates(self, firing_rate):
        '''
        Organizes the firing rates into a dict of separate areas.
        This allows easy access to previous area's firing rates.
        '''
        fr_per_area = {}
        idx = 0
        for area_idx, area in self.areas.items():
            fr_area = firing_rate[idx : idx + area.num_populations]
            fr_per_area[area_idx] = fr_area
            idx = idx + area.num_populations
        return fr_per_area

    def compute_currents(self, ext_ff_rate, fr_per_area, t):
        '''
        Compute the current for each area separately. The total current
        consists of feedforward current (stimulus-driven and/or from other
        brain areas), background current and recurrent current.
        '''
        total_current = torch.Tensor().to(self.device)

        for area_idx, area in self.areas.items():

            # Compute feedforward current of each area, based on
            # area=0: external input or area>0: the previous area's firing rate
            feedforward_current = torch.zeros(area.num_populations).to(self.device)
            if area_idx == '0':
                feedforward_current = torch.matmul(area.input_weights, ext_ff_rate)
            elif area_idx > '0':  # subsequent areas receive previous area's firing rate
                idx_prev_area = str(int(area_idx) - 1)
                prev_area_fr = fr_per_area[idx_prev_area]

                feedforward_current = torch.matmul(area.feedforward_weights, prev_area_fr)

            # Compute recurrent current
            recurrent_current = torch.matmul(area.inner_weights, fr_per_area[area_idx])
            lateral_current = torch.matmul(area.lateral_weights, fr_per_area[area_idx])

            # Background current
            background_current = area.background_weights * area.background_drive

            # Total current of this area
            total_current_area = ((feedforward_current / self.feedforward_scale) +
                                  (lateral_current / self.lateral_scale) +
                                  recurrent_current +
                                  background_current) * area.synapse_time_constant
            total_current = torch.cat((total_current, total_current_area), dim=0)
        return total_current

    def forward(self, t, state):
        '''
        State dynamics updating the membrane potential and adaptation;
        ODE should learn these dynamics and update the weights accordingly.
        '''

        # Prepare the state (membrane, adaptation, firing rate)
        state = state.squeeze(0)  # lose extra dim
        mem_adap_split = len(state) // 3
        adap_rate_split = len(state) // 3 * 2
        membrane_potential, adaptation = state[:mem_adap_split], state[mem_adap_split:adap_rate_split]

        firing_rate = compute_firing_rate(membrane_potential - adaptation)

        # Partition firing rate per area
        fr_per_area = self.partition_firing_rates(firing_rate)

        # Get current stimulus (external ff rate) based on current time t and the time vector time_vec
        ext_ff_rate = torch_interp(t, self.time_vec, self.stim)

        # Compute input current
        total_current = self.compute_currents(ext_ff_rate, fr_per_area, t)

        # Compute derivative membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
            total_current * self.network_as_area.resistance) / self.network_as_area.membrane_time_constant
        delta_adaptation = (-adaptation + self.network_as_area.adaptation_strength *
                            firing_rate) / self.network_as_area.adapt_time_constant

        # Compute derivative firing rate
        prev_firing_rate = state[adap_rate_split:]
        delta_firing_rate = (-prev_firing_rate + firing_rate) / self.network_as_area.synapse_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation, delta_firing_rate))

        return state.unsqueeze(0)

    def diffusion(self, t, y):
        '''
        Diffusion function used by SDE, noise is only applied
        to membrane potential.
        '''
        noise_std = 10.0
        g = torch.zeros_like(y)
        split = (len(y[0]) // 3)
        g[:split, :] = noise_std
        g = g.unsqueeze(dim=-1)
        return g

