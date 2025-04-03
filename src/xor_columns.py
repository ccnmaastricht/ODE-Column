import numpy as np

import torch
import torch.nn as nn
from torchdiffeq import odeint as torch_odeint

from src.coupled_columns import ColumnODEFunc


'''
Possible ff weights to solve XOR solution: 
rand_weights_1 = torch.tensor([0.0000, 0.0000, 0.0043, 0.0027, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0130, 0.0082, 0.0000, 0.0000, 0.0000, 0.0000])
rand_weights_2 = torch.tensor([0.0000, 0.0000, 0.0043, 0.0027, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0130, 0.0082, 0.0000, 0.0000, 0.0000, 0.0000])
rand_weights_C = torch.tensor([0.0000, 0.0000, 0.0082, 0.0130, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0130, 0.0082, 0.0000, 0.0000, 0.0000, 0.0000])
Column A receives the same input, but less strong than column B, hence column A represents the AND 
and column B the OR. If both column have high firing rates, column As activity will inhibit column C, 
thus resulting in low firing rates. However, if only column B is activated, column C will not be 
inhibited and show high firing rates as a result. 
'''


class ColumnsXOR(ColumnODEFunc):
    def __init__(self, column_parameters: dict, area: str):
        super().__init__(column_parameters, area, learn_wta=False)

        self._make_ff_masks()
        self._initialize_ff_weights()
        self._initialize_connection_weights()
        self._extend_params_to_three()

    def _make_ff_masks(self):
        '''
        Makes a mask for the feedforward weights that only targets L4
        in two columns
        '''
        # Mask for only layer 4 of both columns
        ff_weights_mask = torch.zeros(16)
        ff_weights_mask[2:4] += 1.0  # col A
        ff_weights_mask[10:12] += 1.0  # col B
        self.ff_weights_mask = ff_weights_mask

    def _initialize_ff_weights(self):
        '''
        Initializes the feedforward weights between the inputs, columns
        A, B and C as learnable parameters
        '''
        # Initialize feedforward weights for inputs 1 and 2, that connect to columns A and B
        original_weights = self.feedforward_weights.clone().detach() * self.synapse_time_constant
        std_W = 3. * self.synapse_time_constant
        rand_weights_1 = abs(torch.normal(mean=original_weights, std=std_W))
        rand_weights_2 = abs(torch.normal(mean=original_weights, std=std_W))

        # Initialize feedforward weights between columns C and A, B
        rand_weights_C = abs(torch.normal(mean=original_weights, std=std_W))

        # Multiply with mask
        ff_weights_1 = rand_weights_1 * self.ff_weights_mask
        ff_weights_2 = rand_weights_2 * self.ff_weights_mask
        weights_AC = rand_weights_C[:8] * self.ff_weights_mask[:8]
        weights_BC = rand_weights_C[8:] * self.ff_weights_mask[:8]

        self.ff_weights_1 = nn.Parameter(ff_weights_1, requires_grad=True)  # learnable parameters
        self.ff_weights_2 = nn.Parameter(ff_weights_2, requires_grad=True)
        self.ff_weights_AC = nn.Parameter(weights_AC, requires_grad=True)
        self.ff_weights_BC = nn.Parameter(weights_BC, requires_grad=True)

    def _extend_params_to_three(self):
        '''
        Extends the background weights and adaptation strength from two
        to three columns.
        '''
        # Background weights to three columns
        bg_weights = self.background_weights.clone().detach()
        bg_weights = torch.cat((bg_weights, bg_weights[:8]), dim=0)
        self.bg_weights = bg_weights

        # Adaptation rates to three columns
        adap_strength = self.adaptation_strength.clone().detach()
        adap_strength = torch.cat((adap_strength, adap_strength[:8]), dim=0)
        self.adap_strength = adap_strength

    def _initialize_connection_weights(self):
        '''
        Extends the connection weights to three columns and initializes the
        lateral connections as learnable parameters.
        '''
        # Set lateral connections to zero for now
        original_weights = torch.tensor(self.recurrent_weights, dtype=torch.float32) * 0.01 # * self.synapse_time_constant
        conn_weights = torch.zeros((24, 24))
        conn_weights[:16, :16] = original_weights
        conn_weights[16:, 16:] = original_weights[:8, :8]
        conn_weights[1, 8], conn_weights[9, 0] = 0., 0.  # set lateral connections to zero
        # self.connection_weights = conn_weights

        # Learnable lateral weights
        mean_W, std_W = 0., 0.
        lateral_weights = torch.normal(mean=mean_W, std=std_W, size=(16, 16))
        lateral_weights *= self.lat_mask  # only layer 2/3 connections
        conn_weights[:16, :16] += lateral_weights
        self.connection_weights = nn.Parameter(conn_weights,  requires_grad=True)


    def dynamics_xor(self, t: torch.tensor, state: torch.tensor, stim: torch.tensor, time_vec: torch.tensor) -> torch.tensor:
        """
        Dynamics from which XOR ODE should learn the feedforward weights
        """

        ff_rate = torch.tensor(np.array(
            [np.stack([np.interp(t.detach().numpy(), time_vec.detach().numpy(), stim[:, i, j].detach().numpy())
                       for j in range(stim.shape[2])]) for i in range(stim.shape[1])]), dtype=torch.float32)
        ff_rate = ff_rate * 10.  # input in range ~10Hz

        membrane_potential, adaptation = state[0], state[1]

        firing_rate = self.compute_firing_rate_torch(membrane_potential - adaptation)
        firing_rate_C = firing_rate * 10.0  # amplify firing rates received by C by factor 10

        # Compute feedforward current for columns A and B, receiving a weighted sum of both inputs
        ff_current_AB = (ff_rate[0] * self.ff_weights_1) + (ff_rate[1] * self.ff_weights_2)
        # Input to column C are L2/3 firing rates of columns A, B
        ff_current_C = (firing_rate_C[0] * self.ff_weights_AC) + (firing_rate_C[8] * self.ff_weights_BC)
        ff_current_ABC = torch.cat((ff_current_AB, ff_current_C), dim=0)
        feedforward_current = torch.relu(ff_current_ABC)  # make sure ff_currents are never negative

        background_current = self.bg_weights * self.background_drive
        recurrent_current = torch.matmul(self.connection_weights, firing_rate) * 100.

        total_current = (background_current + recurrent_current) * self.synapse_time_constant + feedforward_current

        delta_membrane_potential = (-membrane_potential +
            total_current * self.resistance) / self.membrane_time_constant

        delta_adaptation = (-adaptation + self.adap_strength *
                            firing_rate) / self.adapt_time_constant

        return torch.stack([delta_membrane_potential, delta_adaptation])


    def run_ode_xor(self, input_state, stim, time_vec, num_stim_phases=3):
        '''
        Runs a single sample in three phases: a pre-stimulus phase,
        a stimulus phase (with given stim) and a post-stimulus phase.
        '''

        # Prepare stimulus tensor in two or three phases
        phase_length = int(len(time_vec)/num_stim_phases)

        empty_stim = torch.zeros(1, self.num_populations)
        empty_stim_phase = empty_stim.expand(phase_length, -1)

        stim_phase = stim.expand(phase_length, -1)
        if num_stim_phases == 3:
            whole_stim_phase = torch.cat((empty_stim_phase, stim_phase, empty_stim_phase), dim=0)
        elif num_stim_phases == 2:
            whole_stim_phase = torch.cat((empty_stim_phase, stim_phase), dim=0)

        # Double the stim to input it to both columns
        mirror_stim_phase = torch.cat((whole_stim_phase[:, int(self.num_populations / 2):],
                                      whole_stim_phase[:, :int(self.num_populations / 2)]), dim=1)
        double_stim = torch.stack((whole_stim_phase, mirror_stim_phase), dim=1)

        output = torch_odeint(lambda t, y: self.dynamics_xor(t, y, double_stim, time_vec), input_state, time_vec)

        return output


