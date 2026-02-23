import torch
import torch.nn as nn
from scipy.linalg import block_diag

from src.utils import *
from src.coupled_columns import ColumnArea



class ColumnAreaWTA(ColumnArea):

    '''
    Two columns between which lateral connectivity can be learned to
    exhibit winner-take-all dynamics in perceptual decision-making.
    Connections from L2/3e in column A to L2/3i in column B and L2/3e
    self-excitation connections.
    '''

    def __init__(self, column_parameters, area):
        super().__init__(column_parameters, area, 2, small_network=True)

        self.noise_type = "diagonal"  # sde params
        self.sde_type = "ito"

        self._make_lat_in_mask()

        self._initialize_lat_in_weights()
        self._initialize_output_weights()

    def _make_lat_in_mask(self):
        '''
        Mask to select lateral inhibition connections between L2/3 layers.
        '''
        lat_in_mask = torch.zeros((self.num_populations, self.num_populations))
        lat_in_mask[1, 8], lat_in_mask[9, 0] = 1.0, 1.0  # lateral inhibition
        lat_in_mask[0, 0], lat_in_mask[8, 8] = 1.0, 1.0  # self excitation
        self.register_buffer("lat_in_mask", lat_in_mask)

    def _initialize_lat_in_weights(self):
        '''
        Weights consist of inner connections (8x8) for both columns and external
        connections between columns, i.e. lateral connections. Only the mask-selected
        connections are learnable.
        '''
        zero_weights = torch.zeros_like(self.recurrent_weights)
        std_W = 0.01
        rand_weights = abs(torch.normal(mean=zero_weights, std=std_W))
        lat_in_weights = rand_weights * self.lat_in_mask

        # lat_in_weights[1, 8], lat_in_weights[9, 0] = 1060.0, 1060.0  # lateral inhibition
        # lat_in_weights[0, 0], lat_in_weights[8, 8] = 840.0, 840.0  # self excitation

        self.lat_in_weights = nn.Parameter(lat_in_weights, requires_grad=True)

    def _initialize_output_weights(self):
        output_weights = torch.tensor([1.0000, 0.0000, 0.0000, 0.0000,
                                       0.0000, 0.0000, 0.0000, 0.0000])
        self.register_buffer("output_weights", output_weights)

    def initialize_recurrent_weights(self):
        '''
        Initializes the recurrent (i.e. column-intrinsic) weights are a
        learnable parameter
        '''
        self.recurrent_weights = nn.Parameter(self.recurrent_weights, requires_grad=True)

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
        self.stim = stim.to(self.get_device())

    def get_device(self):
        '''
        Gets the network's current device, based on the recurrent_weights
        '''
        return self.recurrent_weights.device

    def extend_recurr_matrix(self, W):
        W_ext = torch.zeros((self.num_populations, self.num_populations))
        W_ext[:8, :8] = W
        W_ext[8:, 8:] = W
        return W_ext

    def enforce_pos_neg(self, W):
        '''
        Add excitatory/inhibitory constraints to the recurrent matrix
        '''
        cols = torch.arange(W.size(1)).to(self.get_device())

        W_pos = torch.relu(W)  # ≥ 0, all excitatory projections are positive
        W_neg = -torch.relu(-W)  # ≤ 0, all inhibitory projections are negative

        d1 = cols.device
        d2 = W.device

        W = torch.where(cols % 2 == 1, W_neg, W_pos)
        return W

    def constrain_recurr_matrix(self):
        # Add recurrent (inner) weights and lateral inhibition weights and constrain
        W = (self.recurrent_weights * self.internal_mask) + (self.lat_in_weights * self.lat_in_mask)
        self.W = self.enforce_pos_neg(W)

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
        feedforward_current = self.feedforward_weights * feedforward_rate               # stimulus feedforward input
        background_current = self.background_weights * self.background_drive            # background input
        recurrent_current = torch.matmul(self.W, firing_rate)                           # recurrent input
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

        g = torch.zeros_like(y)
        n = y.shape[1] // 3
        sigma_N = 0.5  # original synaptic noise std
        R = self.resistance
        tau_s = self.synapse_time_constant
        tau_m = self.membrane_time_constant
        sigma_H = sigma_N * R / tau_m * (2 * tau_s) ** 0.5
        g[:, :n] = 3.0  # sigma_H
        expected_var = g ** 2 * tau_m / 2

        return g