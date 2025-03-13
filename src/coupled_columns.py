import numpy as np
from scipy.integrate import odeint
from scipy.linalg import block_diag

import torch
import torch.nn as nn

from src.utils import GainFunctionParams, compute_firing_rate, create_feedforward_input


class CoupledColumns:

    def __init__(self, column_parameters: dict, area: str) -> None:

        self.area = area.lower()

        self._intialize_basic_parameters(column_parameters)
        self._initilize_population_parameters(column_parameters)
        self._initialize_connection_probabilities(column_parameters)
        self._initialize_synapses(column_parameters)

        self._build_all_weights()

    def _intialize_basic_parameters(self, column_parameters: dict) -> None:
        """
        Initialize basic parameters for the columns.
        """
        # basic parameters
        self.background_drive = column_parameters['background_drive']
        self.adaptation_strength = np.array(
            column_parameters['adaptation_strength'])

        # time constants and membrane resistance
        self.time_constants = column_parameters['time_constants']
        self.resistance = self.time_constants['membrane'] / column_parameters[
            'capacitance']

        # Gain function parameters
        self.gain_function_parameters = GainFunctionParams(
            **column_parameters['gain_function'])

    def _initilize_population_parameters(self, column_parameters: dict) -> None:
        """
        Initialize the population sizes for the columns.
        """
        self.population_sizes = np.array(
            column_parameters['population_size'][self.area])
        self.num_populations = len(self.population_sizes) * 2
        self.adaptation_strength = np.tile(self.adaptation_strength, 2)
        self.population_sizes = np.tile(self.population_sizes, 2) / 2

    def _initialize_connection_probabilities(self, column_parameters) -> None:
        """
        Initialize the connection probabilities for the columns.
        """
        self.internal_connection_probabilities = np.array(
            column_parameters['connection_probabilities']['internal'])
        self.lateral_connection_probability = column_parameters[
            'connection_probabilities']['lateral']
        self.connection_probabilities = block_diag(
            self.internal_connection_probabilities,
            self.internal_connection_probabilities)

        self.connection_probabilities[1,
                                      8] = self.lateral_connection_probability
        self.connection_probabilities[9,
                                      0] = self.lateral_connection_probability

    def _initialize_synapses(self, column_parameters: dict) -> None:
        """
        Initialize the synapse counts and synaptic strengths for the columns.
        """

        self.background_synapse_counts = np.array(
            column_parameters['synapse_counts']['background'])
        self.feedforward_synapse_counts = np.array(
            column_parameters['synapse_counts']['feedforward'])

        self.background_synapse_counts = np.tile(
            self.background_synapse_counts, 2)
        self.feedforward_synapse_counts = np.tile(
            self.feedforward_synapse_counts, 2)

        self.baseline_synaptic_strength = column_parameters[
            'synaptic_strength']['baseline']
        self.internal_synaptic_strength = column_parameters[
            'synaptic_strength']['internal']
        self.lateral_synaptic_strength = column_parameters[
            'synaptic_strength']['lateral']

        self._compute_recurrent_synapse_counts()
        self._build_recurrent_synaptic_strength_matrix()

    def _compute_recurrent_synapse_counts(self) -> None:
        """
        Compute the number of synapses for recurrent connections based on the
        connection probabilities and population sizes.
        """
        self.recurrent_synapse_counts = np.log(
            1 - self.connection_probabilities) / np.log(
                1 - 1 /
                (np.outer(self.population_sizes, self.population_sizes))
            ) / self.population_sizes[:, None]

    def _build_recurrent_synaptic_strength_matrix(self) -> None:
        """
        Build the synaptic strength matrix.
        """
        inhibitory_scaling_factor = np.array([
            -num_excitatory / num_inhibitory
            for num_excitatory, num_inhibitory in zip(
                self.population_sizes[::2], self.population_sizes[1::2])
        ])
        mask = np.ones((self.num_populations // 2, self.num_populations // 2))
        mask = block_diag(mask, mask).transpose()
        synaptic_strength_column = np.ones(
            self.num_populations) * self.baseline_synaptic_strength
        synaptic_strength_column[
            1::2] = inhibitory_scaling_factor * self.baseline_synaptic_strength

        self.recurrent_synaptic_strength = np.tile(
            synaptic_strength_column, (self.num_populations, 1)) * mask
        self.recurrent_synaptic_strength[0,
                                         0] = self.internal_synaptic_strength
        self.recurrent_synaptic_strength[8,
                                         8] = self.internal_synaptic_strength
        self.recurrent_synaptic_strength[1, 8] = self.lateral_synaptic_strength
        self.recurrent_synaptic_strength[9, 0] = self.lateral_synaptic_strength

    def _build_all_weights(self) -> None:
        """
        Build recurrent, background, external, and feedforward weights from synapse counts and synaptic strengths.
        """
        self.recurrent_weights = self.recurrent_synapse_counts * self.recurrent_synaptic_strength
        self.background_weights = self.background_synapse_counts * self.baseline_synaptic_strength
        self.feedforward_weights = self.feedforward_synapse_counts * self.baseline_synaptic_strength

    def dynamics(self, state: np.ndarray, t: float, *args) -> np.ndarray:
        """
        Compute the dynamics of the coupled columns.
        """

        feedforward_rate = args[0]

        membrane_potential, adaptation = state[:self.num_populations], state[
            self.num_populations:]

        firing_rate = compute_firing_rate(membrane_potential, adaptation,
                                          self.gain_function_parameters)

        feedforward_current = self.feedforward_weights * feedforward_rate
        background_current = self.background_weights * self.background_drive
        recurrent_current = self.recurrent_weights.dot(firing_rate)

        total_current = (feedforward_current + background_current +
                         recurrent_current) * self.time_constants['synapse']

        delta_membrane_potential = (
            -membrane_potential +
            total_current * self.resistance) / self.time_constants['membrane']

        delta_adaptation = (-adaptation + self.adaptation_strength *
                            firing_rate) / self.time_constants['adaptation']

        return np.concatenate([delta_membrane_potential, delta_adaptation])

    def simulate(self, feedforward_rate: np.ndarray,
                 initial_conditions: np.ndarray, simulation_time: float,
                 time_step: float) -> np.ndarray:
        """
        Simulate the dynamics of the coupled columns.
        """

        time = np.arange(0, simulation_time, time_step)

        state = odeint(self.dynamics,
                       initial_conditions,
                       time,
                       args=(feedforward_rate, ))
        return state

    def run_single_sim(self, simulation_parameters: dict,
                       column_parameters: dict,
                       rand_input: bool=True) -> list:
        '''
        Runs a single simulation with either a fixed (in simulation.toml)
        or random stimulus input. Returns a list containing the membrane
        potential, adaptation, firing rate and stimulus input.
        '''
        # Extract parameters
        time_step = simulation_parameters['time_step']
        protocol = simulation_parameters['protocol']

        initial_conditions = np.concatenate(
            (simulation_parameters['initial_conditions']['membrane_potential'],
             simulation_parameters['initial_conditions']['adaptation']))

        layer_4_indices = column_parameters['layer_4_indices']

        states_list = []

        # Pre-stimulus phase
        feedforward_rate_no_stim = np.zeros(self.num_populations)
        state = self.simulate(feedforward_rate_no_stim, initial_conditions,
                                 protocol['pre_stimulus_period'], time_step)
        states_list.append(state)

        # Stimulus phase
        if rand_input is True:
            rand_diff = np.random.uniform(-20.0, 20.0)
            feedforward_rate = create_feedforward_input(
                self.num_populations, layer_4_indices,
                protocol['mean_stimulus_drive'], rand_diff)
        else:
            feedforward_rate = create_feedforward_input(
                self.num_populations, layer_4_indices,
                protocol['mean_stimulus_drive'], protocol['difference_stimulus_drive'])
        state = self.simulate(feedforward_rate, state[-1],
                                 protocol['stimulus_duration'], time_step)
        states_list.append(state)

        # Post-stimulus phase
        feedforward_rate_no_stim = np.zeros(self.num_populations)
        state = self.simulate(feedforward_rate_no_stim, state[-1],
                                 protocol['post_stimulus_period'], time_step)
        states_list.append(state)

        # Convert list to numpy array
        state = np.concatenate(states_list)

        # Compute firing rate
        membrane_potential, adaptation = state[:, :self.num_populations], state[:, self.num_populations:]
        firing_rate = compute_firing_rate(membrane_potential,
                                          adaptation,
                                          self.gain_function_parameters)

        state = np.array([membrane_potential, adaptation, firing_rate]).transpose(1, 0, 2)
        return state, feedforward_rate


class ColumnODEFunc(CoupledColumns):
    def __init__(self, column_parameters: dict, area: str):
        super().__init__(column_parameters, area)

        self.connection_weights = nn.Parameter(torch.tensor(self.recurrent_weights, dtype=torch.float32), requires_grad=True)

        self.feedforward_weights    = torch.tensor(self.feedforward_weights, dtype=torch.float32)
        self.background_weights     = torch.tensor(self.background_weights, dtype=torch.float32)
        self.background_drive       = torch.tensor(self.background_drive, dtype=torch.float32)
        self.synapse_time_constant  = torch.tensor(self.time_constants['synapse'], dtype=torch.float32)
        self.membrane_time_constant = torch.tensor(self.time_constants['membrane'], dtype=torch.float32)
        self.adapt_time_constant    = torch.tensor(self.time_constants['adaptation'], dtype=torch.float32)
        self.resistance             = torch.tensor(self.resistance, dtype=torch.float32)
        self.adaptation_strength    = torch.tensor(self.adaptation_strength, dtype=torch.float32)

    def compute_firing_rate_torch(self, x, params):
        '''
        Compute the firing rates torch-friendly.
        '''
        a, b, d = params.gain, params.threshold, params.noise_factor
        x_nom = a * x - b
        x_activ = x_nom / (1 - torch.exp(-d * x_nom))
        [0.0001 for i in x_activ if i <= 0.]  # ensure positive values (min = 0.0   Hz)
        [500 for i in x_activ if i > 500]  # limit firing rates (max = 500.0 Hz)
        return x_activ

    def dynamics_ode(self, t: float, state: torch.tensor, stim:torch.tensor) -> torch.tensor:
        """
        Compute the dynamics of the coupled columns.
        """

        feedforward_rate = stim

        membrane_potential, adaptation = state[0], state[1]

        firing_rate = self.compute_firing_rate_torch(membrane_potential - adaptation, self.gain_function_parameters)

        feedforward_current = self.feedforward_weights * feedforward_rate
        background_current = self.background_weights * self.background_drive
        recurrent_current = torch.matmul(self.connection_weights, firing_rate)

        total_current = (feedforward_current + background_current +
                         recurrent_current) * self.synapse_time_constant

        delta_membrane_potential = (
            -membrane_potential +
            total_current * self.resistance) / self.membrane_time_constant

        delta_adaptation = (-adaptation + self.adaptation_strength *
                            firing_rate) / self.adapt_time_constant

        return torch.stack([delta_membrane_potential, delta_adaptation])
