import numpy as np
from torchdiffeq import odeint
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.utils import *


class ColumnArea:

    def __init__(self, column_parameters, area, num_columns):

        self.num_columns = num_columns
        self.area = area.lower()

        self._intialize_basic_parameters(column_parameters)
        self._initilize_population_parameters(column_parameters)
        self._initialize_connection_probabilities(column_parameters)
        self._initialize_synapses(column_parameters)

        self._build_all_weights()

    def _intialize_basic_parameters(self, column_parameters):
        """
        Initialize basic parameters for the columns.
        """
        # Basic parameters
        self.background_drive = torch.tensor(column_parameters['background_drive'])
        self.adaptation_strength = torch.tensor(column_parameters['adaptation_strength'])

        # Time constants and membrane resistance
        self.time_constants = column_parameters['time_constants']
        self.synapse_time_constant  = torch.tensor(self.time_constants['synapse'])
        self.membrane_time_constant = torch.tensor(self.time_constants['membrane'])
        self.adapt_time_constant    = torch.tensor(self.time_constants['adaptation'])

        self.resistance = torch.tensor(self.time_constants['membrane']
                                       / column_parameters['capacitance'])

    def _initilize_population_parameters(self, column_parameters):
        """
        Initialize the population sizes for the columns.
        """
        self.population_sizes = np.array(
            column_parameters['population_size'][self.area])
        self.population_sizes = np.tile(self.population_sizes, self.num_columns) / 2 ################# self.num_columns ##################

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

        self.lateral_connection_probability = column_parameters[
            'connection_probabilities']['lateral']

        if self.num_populations > 8:
            self.connection_probabilities[1,
                                          8] = self.lateral_connection_probability
            self.connection_probabilities[9,
                                          0] = self.lateral_connection_probability

    def _initialize_synapses(self, column_parameters):
        """
        Initialize the synapse counts and synaptic strengths for the columns.
        """

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
        self.internal_synaptic_strength = column_parameters[
            'synaptic_strength']['internal']
        self.lateral_synaptic_strength = column_parameters[
            'synaptic_strength']['lateral']

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

        if self.num_populations > 8:
            self.recurrent_synaptic_strength[0,
                                             0] = self.internal_synaptic_strength
            self.recurrent_synaptic_strength[8,
                                             8] = self.internal_synaptic_strength
            self.recurrent_synaptic_strength[1, 8] = self.lateral_synaptic_strength
            self.recurrent_synaptic_strength[9, 0] = self.lateral_synaptic_strength

    def _build_all_weights(self):
        """
        Build recurrent, background, external, and feedforward weights from synapse counts and synaptic strengths.
        """
        self.recurrent_weights = self.recurrent_synapse_counts * self.recurrent_synaptic_strength
        self.background_weights = self.background_synapse_counts * self.baseline_synaptic_strength
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

    def dynamics_ode(self, t, state, stim, time_vec):

        # Get current stimulus (ff rate) based on current time t and the time vector time_vec
        feedforward_rate = torch_interp(t, time_vec, stim)

        membrane_potential, adaptation = state[0], state[1]

        firing_rate = compute_firing_rate_torch(membrane_potential - adaptation)

        feedforward_current = self.feedforward_weights * feedforward_rate       # stimulus feedforward input
        background_current = self.background_weights * self.background_drive    # background input
        recurrent_current = torch.matmul(self.recurrent_weights, firing_rate)   # recurrent input

        total_current = (feedforward_current + background_current + recurrent_current) * self.synapse_time_constant

        delta_membrane_potential = (-membrane_potential +
            total_current * self.resistance) / self.membrane_time_constant

        delta_adaptation = (-adaptation + self.adaptation_strength *
                            firing_rate) / self.adapt_time_constant

        return torch.stack([delta_membrane_potential, delta_adaptation])

    def run_ode(self, initial_state, stim, time_vec):
        '''
        Runs a single sample.
        '''
        return odeint(lambda t, y: self.dynamics_ode(t, y, stim, time_vec), initial_state, time_vec)


class ColumnAreaWTA(ColumnArea):

    '''
    Two columns between which lateral connectivity can be learned to
    exhibit winner-take-all dynamics in perceptual decision-making.
    Connections from L2/3e in column A to L2/3i in column B and L2/3e
    self-excitation connections.
    '''

    def __init__(self, column_parameters, area):
        super().__init__(column_parameters, area, 2)

        self._make_ext_mask()
        self._make_lat_in_mask()

        self.scaling_factor = self.synapse_time_constant
        self._initialize_lat_in_weights()

    def _make_ext_mask(self):
        '''
        Mask to select all external connections
        '''
        ext_mask = torch.zeros(size=(self.num_populations, self.num_populations))
        ext_mask[:8, 8:] = 1.0
        ext_mask[8:, :8] = 1.0
        self.ext_mask = ext_mask

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
        original_weights = self.recurrent_weights * self.scaling_factor
        mean_W = original_weights.mean() / 100.
        std_W = original_weights.std() / 100.
        lateral_weights = torch.normal(mean=mean_W.item(), std=std_W.item(), size=self.ext_mask.shape)
        lateral_weights *= self.ext_mask  # set inner connectivity to zero
        lateral_weights *= self.lat_in_mask  # only layer 2/3 connections

        inner_weights = original_weights
        inner_weights *= 1 - self.ext_mask
        self.recurrent_weights = nn.Parameter(inner_weights + lateral_weights, requires_grad=True)
        # self.recurrent_weights = nn.Parameter(original_weights, requires_grad=True)

    def dynamics_ode(self, t, state, stim, time_vec):

        # Get current stimulus (ff rate) based on current time t and the time vector time_vec
        feedforward_rate = torch_interp(t, time_vec, stim)

        membrane_potential, adaptation = state[0], state[1]

        firing_rate = self.compute_firing_rate_torch(membrane_potential - adaptation)

        feedforward_current = self.feedforward_weights * feedforward_rate       # stimulus feedforward input
        background_current = self.background_weights * self.background_drive    # background input
        recurrent_current = torch.matmul(self.recurrent_weights, firing_rate)   # recurrent input

        total_current = (feedforward_current + background_current + (recurrent_current / self.scaling_factor)) * self.synapse_time_constant

        delta_membrane_potential = (-membrane_potential +
            total_current * self.resistance) / self.membrane_time_constant

        delta_adaptation = (-adaptation + self.adaptation_strength *
                            firing_rate) / self.adapt_time_constant

        return torch.stack([delta_membrane_potential, delta_adaptation])


class ColumnNetwork():

    '''
    Concatenates a number of areas (each consisting of a number
    of columns) to form a larger network. Within an area, only
    lateral connections between columns are allowed. Across areas
    only feedforward- and feedback connections are allowed.
    '''

    def __init__(self, column_parameters, network_dict):
        self.areas = {}

        for area_idx in range(network_dict['nr_areas']):

            area_name = network_dict['areas'][area_idx]
            num_columns = network_dict['nr_columns_per_area'][area_idx]

            area = ColumnArea(column_parameters, area_name, num_columns)
            self.areas[area_idx] = area

        self.nr_input_units = network_dict['nr_input_units']
        self.nr_columns_per_area = network_dict['nr_columns_per_area']

        self._set_external_connections_to_zero()
        self._initialize_feedforward_masks()
        self._initialize_feedforward_weights()

    def _set_external_connections_to_zero(self):
        '''
        Sets external recurrent weights of all areas to zero, to make sure
        all lateral connectivity is removed.
        '''
        for idx, area in self.areas.items():
            recurr_weights = area.recurrent_weights
            area.recurrent_weights = recurr_weights * area.internal_mask

    def _initialize_feedforward_masks(self):
        '''
        Specify from which population the feedforward flow comes
        from (source) and which population it targets (target).
        '''
        # Source of ff is L2/3e
        ff_source_mask = torch.zeros(8)
        ff_source_mask[0] = 1.0
        self.ff_source_mask = ff_source_mask

        # Target of ff is L4e and L4i
        ff_target_mask = torch.zeros(8)
        ff_target_mask[2], ff_target_mask[3] = 1.0, 1.0
        self.ff_target_mask = ff_target_mask

    def _initialize_feedforward_weights(self):
        '''
        Initialize the feedforward weights as learnable weights.
        '''
        feedforward_weights = {}

        for area_idx, area in self.areas.items():

            if area_idx not in feedforward_weights:
                feedforward_weights[area_idx] = []

            if area_idx == 0:   # if first area, check how many external inputs it receives
                nr_ff_weights = self.nr_input_units
            else:               # for subsequent areas, check how many inputs from previous area
                nr_ff_weights = self.areas[area_idx-1].num_columns

            # Weight initialization should not be too small, otherwise no ff flow and no training possible
            original_weights = area.feedforward_weights.clone().detach() * area.synapse_time_constant
            std_W = 3. * area.synapse_time_constant

            for i in range(nr_ff_weights):
                rand_weights = abs(torch.normal(mean=original_weights, std=std_W))
                ff_weights = nn.Parameter(rand_weights, requires_grad=True)
                feedforward_weights[area_idx].append(ff_weights)

        self.feedforward_weights = feedforward_weights

    def dynamics_ode(self, t, state, stim, time_vec):

        # stim should be shaped like [first_area.num_populations, num_inputs]

        # Get current stimulus (external ff rate) based on current time t and the time vector time_vec
        ext_ff_rate = torch_interp(t, time_vec, stim)
        ext_ff_rate = ext_ff_rate * 20.  # input in 1Hz range, so scale up

        membrane_potential, adaptation = state[0], state[1]

        firing_rate = compute_firing_rate_torch(membrane_potential - adaptation)

        # -------
        # separately for each area:
            # if 0: take the ext ff rate
            # if not 0: partition the firing rate to get previous area's fr
            # take the relevant ff weights
            # multiply input/fr with ff_weights
            # concatenate the feedforward current and relu it to make sure it is positive

        for area_idx, area in self.areas.items():

            if area_idx == 0:   # first area
                # assuming stim is shaped like above
                for ext_ff_idx in range(len(ext_ff_rate)):
                    # multiply each input with each ff weights
                    result = ext_ff_rate[ext_ff_idx] * self.feedforward_weights[area_idx][ext_ff_idx]
            else:
                prev_area_size = self.areas[area_idx-1].num_columns
                # partition the correct firing rates
                # result = partitioned_firing_rates.reshaped_to_8_by_num_columns * self.feedforward_weights[area_idx]


        firing_rate_C = firing_rate * 10.  # amp up input from A, B to C

        # Compute feedforward current for columns A and B, receiving a weighted sum of both inputs
        ff_current_AB = (ext_ff_rate[0] * self.ff_weights_1) + (ext_ff_rate[1] * self.ff_weights_2)
        # Input to column C are L2/3 firing rates of columns A, B
        ff_current_C = (firing_rate_C[0] * self.ff_weights_AC) + (firing_rate_C[8] * self.ff_weights_BC)
        ff_current_ABC = torch.cat((ff_current_AB, ff_current_C), dim=0)
        feedforward_current = torch.relu(ff_current_ABC)  # make sure ff_currents are never negative

        # -------
        background_current = self.background_weights * self.background_drive    # background input
        recurrent_current = torch.matmul(self.recurrent_weights, firing_rate)   # recurrent input

        total_current = (feedforward_current + background_current + recurrent_current) * self.synapse_time_constant

        delta_membrane_potential = (-membrane_potential +
            total_current * self.resistance) / self.membrane_time_constant

        delta_adaptation = (-adaptation + self.adaptation_strength *
                            firing_rate) / self.adapt_time_constant

        return torch.stack([delta_membrane_potential, delta_adaptation])



# Testing network

col_params = load_config('../config/model.toml')
network_input = {'nr_areas': 2, 'areas': ['mt', 'mt'], 'nr_columns_per_area': [2,1], 'nr_input_units': 2}

print(network_input)

network = ColumnNetwork(col_params, network_input)


