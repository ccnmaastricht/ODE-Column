import torch
import torch.nn as nn

from src.utils import *
from src.coupled_columns import ColumnArea


class ColumnNetwork(torch.nn.Module):

    '''
    Concatenates a number of areas (each consisting of a number
    of columns) to form a larger network. Within an area, only
    lateral connections between columns are allowed. Across areas
    only feedforward connections are allowed.
    '''

    def __init__(self, model_parameters, network_dict):
        super().__init__()

        self.noise_type = "diagonal"  # sde params
        self.sde_type = "ito"

        self._initialize_areas(model_parameters, network_dict)

        self.network_as_area = ColumnArea(model_parameters, 'mt', sum(network_dict['nr_columns_per_area']))
        self.nr_input_units = network_dict['nr_input_units']
        self.nr_columns_per_area = network_dict['nr_columns_per_area']
        self.nr_areas = network_dict['nr_areas']

        self._initialize_masks(model_parameters)
        self._initialize_feedforward_weights(model_parameters)
        self._initialize_input_weights(model_parameters)
        self._initialize_lateral_weights(model_parameters)
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
            nr_sources_target_receives = 3
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

    def _initialize_input_weights(self, model_parameters):
        '''
        Initialize learnable input weights to weight the input going into the first area.
        '''
        first_area = self.areas['0']

        size_source = self.nr_input_units
        size_target = first_area.num_columns

        input_init = torch.tensor(model_parameters['connection_inits']['input'])
        input_init = torch.tile(input_init, (size_target, size_source))

        std_W = 1.0 # 0.1 # 1.0 # 3.0
        rand_input_weights = abs(torch.normal(mean=input_init, std=std_W)) * self.feedforward_scale
        rand_input_weights *= 0.8

        input_mask = torch.tile(self.input_mask, (size_target, size_source))
        # input_mask = self.make_mask_fan_in(input_mask, 4, 4)
        input_mask = self.make_mask_fan_in(input_mask, 8, 4)
        input_mask[32:64, :] = input_mask[0:32, :]
        # input_mask[0:32, 1:8] = input_mask[0:32, 0:7].clone()  # SHIFTING RFS
        # input_mask[0:32, 0] = torch.zeros(input_mask[0:32, 0].shape)  # SHIFTING RFS
        # input_mask = self.make_mask_fan_in(input_mask, 2, 2)
        # input_mask[0:16, :] = input_mask[32:48, :]  # ORIGINAL
        # input_mask[32:48, :] = input_mask[16:32, :]  # ORIGINAL
        # input_mask = self.make_mask_fan_in_random(input_mask, source_is_input=True)
        first_area.register_buffer('input_mask', input_mask)

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

                std_W = 0.1 # 0.5 # 1.0
                rand_ff_weights = abs(torch.normal(mean=ff_init, std=std_W)) * self.feedforward_scale
                rand_ff_weights *= 4.0

                ff_mask = torch.tile(self.feedforward_mask, (size_target, size_source))
                if int(area_idx) < (self.nr_areas - 1):  # last area should be fully connected
                    if size_target == 2:
                        ff_mask = self.make_mask_fan_in(ff_mask, 2, 2)
                        rand_ff_weights *= 2.0
                    else:
                        ff_mask = self.make_mask_fan_in(ff_mask, 4, 4)
                        rand_ff_weights *= 2.0
                area.register_buffer('feedforward_mask', ff_mask)

                rand_ff_weights = rand_ff_weights * ff_mask
                area.feedforward_weights = nn.Parameter(rand_ff_weights, requires_grad=True)

    def _initialize_lateral_weights(self, model_parameters):
        '''
        Random initialization of lateral weights between columns,
        for each area separately.
        '''

        for area_idx, area in self.areas.items():
            inner_weights = area.recurrent_weights * area.internal_mask  # set any existing external connectivity to zero
            area.register_buffer('inner_weights', inner_weights)

            # Reshape weight initialization
            lateral_init = torch.tensor(model_parameters['connection_inits']['lateral'])
            lateral_init = torch.tile(lateral_init, (area.num_columns, area.num_columns))

            # Reshape mask and store in area
            lateral_mask = torch.tile(self.lateral_mask, (area.num_columns, area.num_columns)) * area.external_mask
            area.register_buffer('lateral_mask', lateral_mask)

            # Randomly initialize lateral weights and store in area as learnable param
            std_W = 0.01
            rand_weights = torch.normal(mean=lateral_init, std=std_W)
            rand_weights *= 0.1  # initialize small lateral weights - let them be learned from scratch
            rand_weights *= area.lateral_mask
            rand_weights *= area.external_mask
            rand_weights = rand_weights

            if area.num_columns > 1:
                area.lateral_weights = nn.Parameter(rand_weights, requires_grad=True)
            else:  # lateral weights of area with one column should not be trainable
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
        output_mask = torch.tile(self.output_mask, (size_source,))
        self.register_buffer('output_mask_full', output_mask)

        # std_W = 1.0
        # rand_output_weights = abs(torch.normal(mean=output_init, std=std_W))
        # rand_output_weights *= output_mask
        # rand_output_weights *= self.output_scale

        self.output_weights = nn.Parameter(output_init, requires_grad=False)

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

    def get_device(self):
        '''
        Gets the network's current device, based on the first area's
        feedforward weights.
        '''
        return self.areas['0'].input_weights.device

    def constrain(self):
        '''
        Constrain all learnable weights so no illegal updates can be made
        '''
        for area_idx, area in self.areas.items():
            # Input weights
            if area_idx == '0':
                input_weights = area.input_weights * area.input_mask
                area.I = torch.relu(input_weights)  # all input weights should be non-negative

            # Feedforward weights
            elif area_idx > '0':
                ff_weights = area.feedforward_weights * area.feedforward_mask
                area.F = torch.relu(ff_weights)

            # Lateral weights
            lat_weights = area.lateral_weights * area.lateral_mask
            area.L = torch.relu(lat_weights)

    def partition_firing_rates(self, firing_rate):
        '''
        Organizes the firing rates into a dict of separate areas.
        This allows easy access to previous area's firing rates.
        '''
        fr_per_area = {}
        idx = 0
        for area_idx, area in self.areas.items():
            fr_area = firing_rate[:, idx : idx + area.num_populations]
            fr_per_area[area_idx] = fr_area
            idx = idx + area.num_populations
        return fr_per_area

    def compute_currents(self, ext_ff_rate, fr_per_area, t):
        '''
        Compute the current for each area separately. The total current
        consists of feedforward current (stimulus-driven and/or from other
        brain areas), background current and recurrent current.
        '''
        total_current = torch.Tensor().to(self.get_device())

        for area_idx, area in self.areas.items():

            # Compute feedforward current of each area, based on
            # area=0: external input or area>0: the previous area's firing rate
            feedforward_current = torch.zeros(area.num_populations)
            if area_idx == '0':
                feedforward_current = torch.matmul(ext_ff_rate, area.I.T)
            elif area_idx > '0':  # subsequent areas receive previous area's firing rate
                idx_prev_area = str(int(area_idx) - 1)
                prev_area_fr = fr_per_area[idx_prev_area]
                feedforward_current = torch.matmul(prev_area_fr, area.F.T)

            # Compute recurrent current
            recurrent_current = torch.matmul(fr_per_area[area_idx], area.inner_weights.T)
            lateral_current = torch.matmul(fr_per_area[area_idx], area.L.T)

            # Background current
            background_current = torch.tile(area.background_drive, (ext_ff_rate.shape[0], 1)) * area.background_weights

            # Total current of this area
            total_current_area = (feedforward_current +
                                  lateral_current +
                                  recurrent_current +
                                  background_current) * area.synapse_time_constant
            total_current = torch.cat((total_current, total_current_area), dim=1)
        return total_current

    def forward(self, t, state):
        '''
        State dynamics updating the membrane potential and adaptation;
        ODE should learn these dynamics and update the weights accordingly.
        '''
        # Unpack the state (membrane, adaptation) and compute firing rate
        mem_adap_split = state.shape[1] // 2
        membrane_potential, adaptation = state[:, :mem_adap_split], state[:, mem_adap_split:]

        firing_rate = compute_firing_rate(membrane_potential - adaptation)

        # Partition firing rate per area
        fr_per_area = self.partition_firing_rates(firing_rate)

        # If more than half of time has passed, present stim
        ext_ff_rate = torch.zeros_like(self.stim)
        if t > self.time_vec[len(self.time_vec) // 2]:
            ext_ff_rate = self.stim

        # Compute input current
        total_current = self.compute_currents(ext_ff_rate, fr_per_area, t)

        # Compute derivative membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
            total_current * self.network_as_area.resistance) / self.network_as_area.membrane_time_constant
        delta_adaptation = (-adaptation + self.network_as_area.adaptation_strength *
                            firing_rate) / self.network_as_area.adapt_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation), dim=1)

        return state

    def diffusion(self, t, y):
        '''
        Diffusion function used by SDE, noise is only applied
        to membrane potential.
        '''
        g = torch.zeros_like(y)
        n = y.shape[1] // 2
        g[:, :n] = 3.0  # sigma_H
        return g

