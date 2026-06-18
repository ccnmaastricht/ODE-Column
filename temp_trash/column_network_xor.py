import torch.nn as nn

from src.utils import *
from temp_trash.coupled_columns import ColumnArea



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
            std_W = 0.1

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

