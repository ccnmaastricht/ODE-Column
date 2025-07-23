import numpy as np
from torchdiffeq import odeint, odeint_adjoint
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.utils import *
from src.xor_columns import ColumnsXOR  # to compare to


class ColumnArea(torch.nn.Module):

    def __init__(self, column_parameters, area, num_columns):
        super().__init__()

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
        self.population_sizes = np.tile(self.population_sizes, self.num_columns) / self.num_columns ######## keep population sizes as they are

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


class ColumnAreaWTA(ColumnArea):

    '''
    Two columns between which lateral connectivity can be learned to
    exhibit winner-take-all dynamics in perceptual decision-making.
    Connections from L2/3e in column A to L2/3i in column B and L2/3e
    self-excitation connections.
    '''

    def __init__(self, column_parameters, area):
        super().__init__(column_parameters, area, 2)

        self.noise_type = "diagonal"  # sde params
        self.sde_type = "ito"

        self._make_ext_mask()
        self._make_lat_in_mask()

        self.scaling_factor = self.synapse_time_constant
        self.population_sizes = self.population_sizes # / 2  # just like Kris!
        self._initialize_lat_in_weights()
        self._initialize_output_weights() # combination of output layers L23 and L5

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
        # lat_in_mask[0, 0], lat_in_mask[8, 8] = 1.0, 1.0  # self excitation
        self.lat_in_mask = lat_in_mask

        self_ex_mask = torch.zeros((self.num_populations, self.num_populations))
        self_ex_mask[0, 0], self_ex_mask[8, 8] = 1.0, 1.0  # self excitation
        self.self_ex_mask = self_ex_mask

    def _initialize_lat_in_weights(self):
        '''
        Weights consist of inner connections (8x8) for both columns and external
        connections between columns, i.e. lateral connections. Only the mask-selected
        connections are learnable.
        '''
        original_weights = self.recurrent_weights * self.scaling_factor
        self.original_weights = original_weights
        mean_W = original_weights.mean() / 10.
        std_W = original_weights.std() / 10.
        lateral_weights = abs(torch.normal(mean=mean_W.item(), std=std_W.item(), size=self.ext_mask.shape))
        # lateral_weights = torch.full(self.ext_mask.shape, torch.abs(torch.normal(mean_W, std_W)).item())
        lateral_weights *= self.lat_in_mask
        self.lateral_weights = nn.Parameter(lateral_weights, requires_grad=True)

        inner_weights = original_weights
        inner_weights *= 1 - self.ext_mask
        self.inner_weights = nn.Parameter(inner_weights, requires_grad=True)
        self.recurrent_weights = inner_weights + lateral_weights
        # self.recurrent_weights = original_weights

    def _initialize_output_weights(self):
        output_weights = torch.tensor([1.0000, 0.0000, 0.0000, 0.0000,
                                       0.0000, 0.0000, 0.0000, 0.0000])
        self.output_weights = output_weights
        # self.output_weights = nn.Parameter(output_weights, requires_grad=True)

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

        # Prepare the state (membrane, adaptation, firing rate)
        state = state.squeeze(0)  # lose extra dim
        mem_adap_split = len(state) // 3
        adap_rate_split = len(state) // 3 * 2
        membrane_potential, adaptation = state[:mem_adap_split], state[mem_adap_split:adap_rate_split]

        # Compute new firing rate from membrane and adaptation
        firing_rate = compute_firing_rate_torch(membrane_potential - adaptation)

        # Get current stimulus (ff rate) based on current time t and the time vector time_vec
        feedforward_rate = torch_interp(t, self.time_vec, self.stim)

        # Compute current current
        feedforward_current = self.feedforward_weights * feedforward_rate       # stimulus feedforward input
        background_current = self.background_weights * self.background_drive    # background input
        # self.recurrent_weights = self.inner_weights + self.lateral_weights              # on while training, off while testinggggg
        recurrent_current = torch.matmul(self.recurrent_weights, firing_rate)   # recurrent input

        total_current = (feedforward_current + background_current + (recurrent_current / self.scaling_factor)) * self.synapse_time_constant

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
        noise_std = 2.0
        g = torch.zeros_like(y)
        split_mem = (len(y[0]) // 3)
        split_fr = (len(y[0]) // 3) * 2
        g[:split_mem] = noise_std  # membrane and adaptation
        g[split_fr:] = noise_std
        return g


# Example weights matrices
#     def _initialize_ff_fb_weights(self):
#         '''
#         Initialize the feedforward/feedback weights as learnable weights.
#         These connect areas to each other and allow information flow in both
#         feedforward and feedback directions.
#         '''
#
#         ff_weights = nn.ParameterList()
#
#         for area_idx, area in self.areas.items():
#
#             if area_idx == '0':   # first area receives input
#                 size_source = self.nr_input_units  # nr of rows weights matrix
#
#                 ff_area_weights = torch.zeros((1, 8))
#                 ff_area_weights[0, 2], ff_area_weights[0, 3] = 1.0, 1.0
#
#             else:  # for subsequent areas, check how many inputs from previous area
#                 idx_prev_area = str(int(area_idx) - 1)
#                 size_source = self.areas[idx_prev_area].num_columns  # nr of rows weights matrix
#
#                 ff_area_weights = torch.zeros((8, 8))
#                 ff_area_weights[0, 2], ff_area_weights[0, 3], ff_area_weights[4, 2], ff_area_weights[4, 3] = 1.0, 1.0, 1.0, 1.0
#
#             size_target = area.num_columns  # nr of columns weights matrix
#             ff_area_weights = torch.tile(ff_area_weights, (size_source, size_target))
#             ff_weights.append(nn.Parameter(ff_area_weights))
#
#             # # Weight initialization should not be too small, otherwise no ff flow and no training possible
#             # original_weights = area.feedforward_weights.clone().detach() * area.synapse_time_constant
#             # std_W = area.synapse_time_constant
#             #
#             # for i in range(nr_ff_weights):
#             #     rand_weights = abs(torch.normal(mean=original_weights, std=std_W))
#             #     rand_weights_masked = rand_weights * torch.tile(self.input_target_mask, (area.num_columns,))
#             #     blep = nn.Parameter(rand_weights_masked, requires_grad=True)
#             #     ff_weights[area_idx].append(blep)
#
#         self.ff_weights = ff_weights
#
#     def compute_currents(self, ext_ff_rate, fr_per_area, t):
#         '''
#         Compute the current for each area separately. The total current
#         consists of feedforward current (stimulus-driven and/or from other
#         brain areas), background current and recurrent current.
#         '''
#         total_current = torch.Tensor()
#
#         for area_idx, area in self.areas.items():
#
#             # Compute feedforward current of each area, based on
#             # area=0: external input or area>0: the previous area's firing rate
#
#             if area_idx == '0':
#                 feedforward_current = torch.matmul(ext_ff_rate, self.ff_weights[int(area_idx)])
#             elif area_idx > '0':  # subsequent areas receive previous area's firing rate
#                 idx_prev_area = str(int(area_idx) - 1)
#                 prev_area_fr = fr_per_area[idx_prev_area]
#                 feedforward_current = torch.matmul(prev_area_fr, self.ff_weights[int(area_idx)])
#             feedforward_current = torch.relu(feedforward_current)  # make sure ff_currents are never negative
#
#             # Background and recurrent current
#             background_current = area.background_weights * area.background_drive
#             recurrent_current = torch.matmul(area.recurrent_weights, fr_per_area[area_idx].flatten())   # recurrent input
#
#             # Total current of this area
#             # Notice that ff is not scaled down by synapse time constant bc ff weights are already scaled down for training
#             total_current_area = feedforward_current + (background_current + recurrent_current) * area.synapse_time_constant
#
#             total_current = torch.cat((total_current, total_current_area), dim=0)
#         return total_current



class ColumnNetwork(torch.nn.Module):

    '''
    Concatenates a number of areas (each consisting of a number
    of columns) to form a larger network. Within an area, only
    lateral connections between columns are allowed. Across areas
    only feedforward- and feedback connections are allowed.
    '''

    def __init__(self, column_parameters, network_dict):
        super().__init__()

        self.noise_type = "diagonal"  # sde params
        self.sde_type = "ito"

        self._initialize_areas(column_parameters, network_dict)

        self.network_as_area = ColumnArea(column_parameters, 'mt', sum(network_dict['nr_columns_per_area']))
        self.nr_input_units = network_dict['nr_input_units']
        self.nr_columns_per_area = network_dict['nr_columns_per_area']

        self._initialize_lateral_weights()
        self._initialize_ff_fb_masks()
        self._initialize_feedforward_weights()
        self._initialize_feedback_weights()
        self._initialize_output_weights()

    def _initialize_areas(self, column_parameters, network_dict):
        self.areas = nn.ModuleDict({})
        for area_idx in range(network_dict['nr_areas']):

            area_name = network_dict['areas'][area_idx]
            num_columns = network_dict['nr_columns_per_area'][area_idx]

            area = ColumnArea(column_parameters, area_name, num_columns)
            self.areas[str(area_idx)] = area

    def _initialize_lateral_weights(self):
        '''
        Sets external recurrent weights of all areas to zero, to make sure
        all lateral connectivity is removed.
        '''
        for idx, area in self.areas.items():
            recurr_weights = area.recurrent_weights
            area.recurrent_weights = recurr_weights * area.internal_mask  # set any existing external connectivity to zero

            lateral_weights = torch.zeros((16, 16))
            lateral_weights[1, 8], lateral_weights[9, 0] = 0.01, 0.01  # initialize the lateral weights
            lateral_weights = torch.tile(lateral_weights, (area.num_columns, area.num_columns))
            lateral_weights = lateral_weights[:area.num_populations, :area.num_populations]

            std_W = area.synapse_time_constant
            rand_weights = abs(torch.normal(mean=lateral_weights, std=std_W))
            rand_weights *= (lateral_weights * 100.)  # *100 to use it as a 1.0 mask

            area.lateral_weights = nn.Parameter(rand_weights, requires_grad=True)

    def _initialize_ff_fb_masks(self):
        '''
        Specify from which population the feedforward flow comes
        (source) and which population it targets (target).
        '''
        # Source of ff is L2/3e (and L5e)
        ff_source_mask = torch.tensor([1., 0., 0., 0., 1., 0., 0., 0.])
        self.ff_source_mask = ff_source_mask

        # Target of ff is L4e and L4i
        ff_target_mask = torch.tensor([0., 0., 1., 1., 0., 0., 0., 0.])
        self.ff_target_mask = ff_target_mask

        # Source of fb is L5e and L6e
        fb_source_mask = torch.tensor([0., 0., 0., 0., 1., 0., 1., 0.])
        self.fb_source_mask = fb_source_mask

        # Target of fb is L2/3, L5, L6
        fb_target_mask = torch.tensor([1., 1., 0., 0., 1., 1., 1., 1.])
        self.fb_target_mask = fb_target_mask

    def _initialize_feedforward_weights(self):
        '''
        Initialize the feedforward weights as learnable weights. Both target
        and source weights are initialized.
        '''
        feedforward_target_weights = nn.ModuleDict({})
        feedforward_source_weights = nn.ModuleDict({})

        for area_idx, area in self.areas.items():

            feedforward_target_weights[area_idx] = nn.ParameterList()
            feedforward_source_weights[area_idx] = nn.ParameterList()

            if area_idx == '0':   # if first area, check how many external inputs it receives
                nr_ff_weights = self.nr_input_units
            else:               # for subsequent areas, check how many inputs from previous area
                key_prev_area = str(int(area_idx)-1)
                nr_ff_weights = self.areas[key_prev_area].num_columns

            # Weight initialization should not be too small, otherwise no ff flow and no training possible
            original_target_weights = area.feedforward_weights.clone().detach() * area.synapse_time_constant
            init_source_weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0])  # no tiling, cuz will be applied to only one column of prev area
            std_W = area.synapse_time_constant

            for i in range(nr_ff_weights):

                rand_weights_target = abs(torch.normal(mean=original_target_weights, std=std_W))
                rand_weights_target = rand_weights_target * torch.tile(self.ff_target_mask, (area.num_columns,))
                ff_weights_target = nn.Parameter(rand_weights_target, requires_grad=True)
                feedforward_target_weights[area_idx].append(ff_weights_target)

                rand_weights_source = abs(torch.normal(mean=init_source_weights, std=std_W))
                rand_weights_source = rand_weights_source * self.ff_source_mask
                if area_idx == '0':  # the first area needs no source weights; but need to init to keep indexing
                    ff_weights_source = nn.Parameter(rand_weights_source, requires_grad=False)
                else:
                    ff_weights_source = nn.Parameter(rand_weights_source, requires_grad=True)
                feedforward_source_weights[area_idx].append(ff_weights_source)

        self.feedforward_target_weights = feedforward_target_weights
        self.feedforward_source_weights = feedforward_source_weights

    def _initialize_feedback_weights(self):
        '''
        Initialize the feedback weights as learnable weights. Both target
        and source weights are initialized.
        '''
        feedback_target_weights = nn.ModuleDict({})
        feedback_source_weights = nn.ModuleDict({})

        for area_idx, area in self.areas.items():

            if int(area_idx) == (len(self.areas) - 1):  # skip last area
                break

            feedback_target_weights[area_idx] = nn.ParameterList()
            feedback_source_weights[area_idx] = nn.ParameterList()

            key_next_area = str(int(area_idx) + 1)
            nr_fb_weights = self.areas[key_next_area].num_columns

            # Disclaimer: I really don't know what I'm doing here in terms of initialization (which layers, exci/inhi or magnitude)
            # Target weights need to be in range ~25,16 (range original feedforward weights)
            init_target_weights = torch.tile(torch.tensor([5.0, 0.1, 0.0, 0.0, 5.0, 0.1, 5.0, 0.1]), (area.num_columns,)) * area.synapse_time_constant
            init_source_weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0])  # no tiling, cuz will be applied to only one column of prev area
            std_W = area.synapse_time_constant

            for i in range(nr_fb_weights):
                rand_weights_target = abs(torch.normal(mean=init_target_weights, std=std_W))
                rand_weights_target = rand_weights_target * torch.tile(self.fb_target_mask, (area.num_columns,))
                fb_weights_target = nn.Parameter(rand_weights_target, requires_grad=True)
                feedback_target_weights[area_idx].append(fb_weights_target)

                rand_weights_source = abs(torch.normal(mean=init_source_weights, std=std_W))
                rand_weights_source = rand_weights_source * self.fb_source_mask
                fb_weights_source = nn.Parameter(rand_weights_source, requires_grad=True)
                feedback_source_weights[area_idx].append(fb_weights_source)

        self.feedback_target_weights = feedback_target_weights
        self.feedback_source_weights = feedback_source_weights

    def _initialize_output_weights(self):
        '''
        Initialize learnable output weights that can be used to read out
        the firing rates of the final column as a means of classification.
        '''
        key_last_area = str(len(self.areas)-1)
        nr_pops_final_area = self.areas[key_last_area].num_populations
        output_weights = torch.zeros(nr_pops_final_area)
        output_weights[0], output_weights[4] = 1.0, 0.05

        std_W = self.network_as_area.synapse_time_constant
        rand_output_weights = abs(torch.normal(mean=output_weights, std=std_W))
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
                    prev_area_fr = fr_per_area[key_prev_area][ff_idx] * self.feedforward_source_weights[area_idx][ff_idx]
                    prev_area_fr_sum = torch.sum(prev_area_fr)
                    feedforward_current += prev_area_fr_sum * ff_target_weight

            # Compute feedback current
            feedback_current = torch.zeros(area.num_populations)

            if int(area_idx) < (len(self.areas) - 1):  # only last area has no fb weights, so skip that one
                for fb_idx, fb_target_weight in enumerate(self.feedback_target_weights[area_idx]):
                    key_next_area = str(int(area_idx) + 1)
                    next_area_fr = fr_per_area[key_next_area][fb_idx] * self.feedback_source_weights[area_idx][fb_idx]
                    next_area_fr_sum = torch.sum(next_area_fr)
                    feedback_current += next_area_fr_sum * fb_target_weight

            # Background and recurrent current
            background_current = area.background_weights * area.background_drive
            recurrent_current = torch.matmul((area.recurrent_weights + area.lateral_weights), fr_per_area[area_idx].flatten())

            # Total current of this area
            # Notice that ff an fb are not scaled down by synapse time constant bc they are already scaled down for training
            total_current_area = feedforward_current + feedback_current + (background_current + recurrent_current) * area.synapse_time_constant

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

        firing_rate = compute_firing_rate_torch(membrane_potential - adaptation)

        # Partition firing rate per area
        fr_per_area = self.partition_firing_rates(firing_rate)

        # Get current stimulus (external ff rate) based on current time t and the time vector time_vec
        ext_ff_rate = torch_interp(t, self.time_vec, self.stim)
        ext_ff_rate = ext_ff_rate * 20.  # input in 1Hz range, so scale up

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

        # if torch.any(torch.abs(state) > 1e3):  # You can start with 1e3 as a sanity threshold
        #     print(f"[t={t.item():.3f}] 🚨 Large derivative detected! Max: {state.abs().max().item():.2e}")
        #     print(f"[t={t.item():.3f}] max(mem_pot): {delta_membrane_potential.abs().max().item():.2e}")
        #     print(f"[t={t.item():.3f}] max(adaptation): {delta_adaptation.abs().max().item():.2e}")
        #     print(f"[t={t.item():.3f}] max(firing_rate): {delta_firing_rate.abs().max().item():.2e}")

        # state = torch.clamp(state, -1e4, 1e4)

        # if torch.any(membrane_potential > 120) or torch.any(membrane_potential < -120):
        # print(t)
        # pprint(membrane_potential[0])
        # blep = 0

        return state.unsqueeze(0)

    def diffusion(self, t, y):
        # Only membrane gets noise, adaptation is deterministic
        noise_std = 3.0
        g = torch.zeros_like(y)
        split = (len(y[0]) // 3) * 2
        g[:, split:] = noise_std  # only firing rates (after split)
        return g

