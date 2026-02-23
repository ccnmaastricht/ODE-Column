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
            self.population_sizes = self.population_sizes #  / self.num_columns   ######### NO HALVING OF POPULATIONS ###########

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
        recurrent_weights = self.recurrent_synapse_counts * self.recurrent_synaptic_strength
        self.register_buffer("recurrent_weights", recurrent_weights) # device
        background_weights = self.background_synapse_counts * self.baseline_synaptic_strength
        self.register_buffer("background_weights", background_weights) # device
        feedforward_weights = self.feedforward_synapse_counts * self.baseline_synaptic_strength
        self.register_buffer("feedforward_weights", feedforward_weights) # device

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
        self.register_buffer("internal_mask", internal_mask) # device
        external_mask = 1 - mask
        self.register_buffer("external_mask", external_mask) # device

