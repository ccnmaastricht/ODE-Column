import torch
import numpy as np
from scipy.linalg import block_diag


class BrainArea(torch.nn.Module):
    """
    Represents a cortical brain area consisting of a group of laminar columns with
    shared population counts, connection probabilities, and synaptic strengths.

    Args:
        col_params (dict): General configuration dictionary containing population size,
            background synapse count, and synaptic strength specifications.
        area_name (str): Configured area name string matching parameters in config.
        num_columns (int): Number of cortical columns contained within this area.
        unique_id (str): Unique area identifier string within the parent network.

    Attributes:
        num_columns (int): Number of cortical columns in the area.
        area_name (str): Configured name of the area.
        id (str): Unique area identifier.
        population_sizes (np.ndarray): Array of population sizes tiled across columns.
        num_populations (int): Total number of neuronal populations in the area.
        connection_probabilities (np.ndarray): Block-diagonal connection matrix of
            shape `(num_populations, num_populations)`.
        background_synapse_counts (torch.Tensor): Tiled background synapse count vector.
        baseline_synaptic_strength (float): Baseline synaptic strength value.
        recurrent_synapse_counts (torch.Tensor): Recurrent synapse count matrix.
        recurrent_synaptic_strength (torch.Tensor): Recurrent synaptic strength matrix.
        recurrent_weights (torch.Tensor): Registered buffer of internal recurrent weights.
        background_weights (torch.Tensor): Registered buffer of background drive weights.
        internal_mask (torch.Tensor): Registered buffer mask for intra-column connections.
        external_mask (torch.Tensor): Registered buffer mask for inter-column connections.
    """

    def __init__(self, col_params, area_name, num_columns, unique_id):

        super().__init__()

        self.num_columns = num_columns
        self.area_name = area_name
        self.id = unique_id

        self._initialize_population_parameters(col_params)
        self._initialize_connection_probabilities(col_params)
        self._initialize_synapses(col_params)
        self._build_all_weights()

    def _initialize_population_parameters(self, column_parameters):
        """
        Initialize tiled population size arrays across columns and construct internal/external
        column connectivity masks.

        Args:
            column_parameters (dict): Parameter dictionary containing population sizes.
        """
        self.population_sizes = np.array(
            column_parameters['population_size'][self.area_name])
        self.population_sizes = np.tile(self.population_sizes, self.num_columns)
        self.num_populations = len(self.population_sizes)

        self._make_in_ex_masks(self.num_columns)

    def _initialize_connection_probabilities(self, column_parameters):
        """
        Initialize internal connection probability matrix tiled along the block diagonal for
        all columns in the area.

        Args:
            column_parameters (dict): Parameter dictionary containing connection probabilities.
        """
        self.internal_connection_probabilities = torch.tensor(
            column_parameters['connection_probabilities']['internal'])

        # Copy internal connections n times along diagonal for n columns
        blocks = [self.internal_connection_probabilities] * self.num_columns
        self.connection_probabilities = block_diag(*blocks)

    def _initialize_synapses(self, column_parameters):
        """
        Initialize background synapse counts, baseline synaptic strength, and compute recurrent
        synapse count and strength matrices.

        Args:
            column_parameters (dict): Parameter dictionary containing synapse specs.
        """
        background_synapse_counts = torch.tensor(
            column_parameters['background_synapse_counts'][self.area_name])
        self.background_synapse_counts = torch.tile(
            background_synapse_counts, (self.num_columns,))

        self.baseline_synaptic_strength = column_parameters[
            'synaptic_strength']['baseline']

        self._compute_recurrent_synapse_counts()
        self._build_recurrent_synaptic_strength_matrix()

    def _compute_recurrent_synapse_counts(self):
        """
        Compute recurrent synapse count matrix using connection probabilities and population sizes.
        """
        log_numerator = np.log(1 - np.array(self.connection_probabilities))
        log_denominator = np.log(1 - 1 / np.array(np.outer(self.population_sizes, self.population_sizes)))

        recurrent_synapse_counts = log_numerator / log_denominator / self.population_sizes[:, None]
        self.recurrent_synapse_counts = torch.tensor(recurrent_synapse_counts, dtype=torch.float32)

    def _build_recurrent_synaptic_strength_matrix(self):
        """
        Construct recurrent synaptic strength matrix applying baseline strength and inhibitory
        population scaling factors.
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
        Compute recurrent and background weight matrices and register them as PyTorch module buffers.
        """
        recurrent_weights = self.recurrent_synapse_counts * self.recurrent_synaptic_strength
        self.register_buffer("recurrent_weights", recurrent_weights)
        background_weights = self.background_synapse_counts * self.baseline_synaptic_strength
        self.register_buffer("background_weights", background_weights)

    def _make_in_ex_masks(self, num_columns):
        """
        Construct internal intra-column mask and external inter-column mask matrices and register
        them as PyTorch buffers.

        Args:
            num_columns (int): Number of cortical columns in the area.
        """
        column_size = self.num_populations // num_columns

        mask = torch.zeros(self.num_populations, self.num_populations)

        for i in range(0, self.num_populations, column_size):
            idx1 = i
            idx2 = i + column_size
            mask[idx1:idx2, idx1:idx2] = 1.0

        internal_mask = mask
        self.register_buffer("internal_mask", internal_mask)

        external_mask = 1 - mask
        self.register_buffer("external_mask", external_mask)
