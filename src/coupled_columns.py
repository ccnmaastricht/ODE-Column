import numpy as np
from torchdiffeq import odeint
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.utils import *
from src.xor_columns import ColumnsXOR  # to compare to


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

        self.network_as_area = ColumnArea(column_parameters, 'mt', sum(network_dict['nr_columns_per_area']))
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
                rand_weights_masked = rand_weights * torch.tile(self.ff_target_mask, (area.num_columns,))
                ff_weights = nn.Parameter(rand_weights_masked, requires_grad=True)
                feedforward_weights[area_idx].append(ff_weights)

        self.feedforward_weights = feedforward_weights

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

    def dynamics_ode(self, t, state, stim, time_vec):
        '''
        State dynamics updating the membrane potential and adaptation;
        ODE should learn these dynamics and update the weights accordingly.
        '''
        # Get current stimulus (external ff rate) based on current time t and the time vector time_vec
        ext_ff_rate = torch_interp(t, time_vec, stim)
        ext_ff_rate = ext_ff_rate * 20.  # input in 1Hz range, so scale up

        # Compute firing rate from membrane potential and adaptation
        membrane_potential, adaptation = state[0], state[1]
        firing_rate = compute_firing_rate_torch(membrane_potential - adaptation)

        # Partition firing rate per area
        fr_per_area = self.partition_firing_rates(firing_rate)

        # Compute the next state separately for each area
        total_current = torch.Tensor()

        for area_idx, area in self.areas.items():

            # Compute feedforward current of each area, based on
            # area=0: external input or area>0: the previous area's firing rate
            ff_current_area = torch.zeros(area.num_populations)

            for ff_idx in range(len(self.feedforward_weights[area_idx])):
                if area_idx == 0:   # first area gets external input
                    # multiply each input with each ff weights
                    ff_current_area += ext_ff_rate[ff_idx] * self.feedforward_weights[area_idx][ff_idx]

                elif area_idx > 0:   # subsequent areas receive previous area's firing rate
                    prev_area_fr = fr_per_area[area_idx-1][ff_idx] * self.ff_source_mask
                    prev_area_fr_sum = torch.sum(prev_area_fr)  # if more source layers, sum output
                    prev_area_fr_sum = prev_area_fr_sum * 10.  # amp up input
                    ff_current_area += prev_area_fr_sum * self.feedforward_weights[area_idx][ff_idx]

            feedforward_current = torch.relu(ff_current_area)  # make sure ff_currents are never negative

            # Background and recurrent current
            background_current = area.background_weights * area.background_drive    # background input
            recurrent_current = torch.matmul(area.recurrent_weights, fr_per_area[area_idx].flatten())   # recurrent input

            # Total current of this area
            # Notice that ff is not scaled down by synapse time constant bc ff weights are already scaled down for training
            total_current_area = feedforward_current + (background_current + recurrent_current) * area.synapse_time_constant
            total_current = torch.cat((total_current, total_current_area), dim=0)

        # Compute new membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
            total_current * self.network_as_area.resistance) / self.network_as_area.membrane_time_constant
        delta_adaptation = (-adaptation + self.network_as_area.adaptation_strength *
                            firing_rate) / self.network_as_area.adapt_time_constant

        return torch.stack([delta_membrane_potential, delta_adaptation])

    def run_ode_network(self, state, time_vec, stim):
        return odeint(lambda t, y: self.dynamics_ode(t, y, stim, time_vec), state, time_vec)



### Testing network

if __name__ == '__main__':

    # Init network
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')

    network_input = {'nr_areas': 2, 'areas': ['mt', 'mt'], 'nr_columns_per_area': [2,1], 'nr_input_units': 2}
    num_columns = sum(network_input['nr_columns_per_area'])
    print(network_input)

    network = ColumnNetwork(col_params, network_input)

    # Initial state
    membrane = torch.tensor(sim_params['initial_conditions']['membrane_potential'])
    adaptation = torch.tensor(sim_params['initial_conditions']['adaptation'])
    mem_adap = torch.stack([membrane, adaptation])
    initial_state = torch.tile(mem_adap, (num_columns,))

    # Time vector
    time_steps = int(sim_params['protocol']['stimulus_duration'] * 2 / sim_params['time_step'])
    time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)

    # Stimulus
    stim = create_feedforward_input(network_input['nr_input_units']*8, 0., 1.)
    empty_stim = torch.zeros(1, network_input['nr_input_units']*8)

    phase_length = int(len(time_vec) / 2)
    empty_stim_phase = empty_stim.expand(phase_length, -1)
    stim_phase = stim.expand(phase_length, -1)

    whole_stim_phase = torch.cat((stim_phase, stim_phase), dim=0)

    # Double the stim to input it to both columns
    mirror_stim_phase = torch.cat((whole_stim_phase[:, 8:], whole_stim_phase[:, :8]), dim=1)
    double_stim = torch.stack((whole_stim_phase, mirror_stim_phase), dim=1)  # (time steps, 2, num populations)


    version = 'old'

    xor_network = ColumnsXOR(col_params, 'mt')

    results = torch.Tensor(1000, 2, 24)
    for i, t in enumerate(time_vec):
        if version == 'new':
            next_state = network.dynamics_ode(t, initial_state, double_stim, time_vec)
        else:
            next_state = xor_network.dynamics_xor(t, initial_state, double_stim, time_vec)
        initial_state = next_state
        results[i,:,:] = next_state

    plt.plot(results[:, 0, 0].detach().numpy())
    plt.show()
