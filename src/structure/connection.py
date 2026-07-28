import torch
import math


class Connection(torch.nn.Module):
    """
    Represents structural connections between areas or populations, supporting
    recurrent, background, feedforward, feedback, lateral, input, and output types.

    Args:
        conn_type (str): Connection type classification string (e.g., 'feedforward').
        source (str): Identifier of the source area or input name.
        target (str): Identifier of the target area or output name.
        trainable (bool): Whether connection weights are updated during training.
        source_size (int | None, optional): Total populations in source area. Defaults to None.
        target_size (int | None, optional): Total populations in target area. Defaults to None.
        initialize_weights_and_mask (bool, optional): Whether to allocate empty parameters
            for weights and mask during instantiation. Defaults to False.

    Attributes:
        conn_type (str): Type specification string.
        source_id (str): Source identifier string.
        target_id (str): Target identifier string.
        source_size (int | None): Number of source populations.
        target_size (int | None): Number of target populations.
        trainable (bool): Flag indicating if weights parameter requires grad.
        weights (torch.nn.Parameter): Learnable synaptic weight parameter matrix.
        mask (torch.Tensor): Registered buffer mask enforcing structural connectivity.
        W (torch.Tensor): Constrained weight matrix enforcing sign constraints.
    """

    def __init__(self, conn_type, source, target, trainable, source_size=None, target_size=None, initialize_weights_and_mask=False):

        super().__init__()

        self.conn_type      = conn_type
        self.source_id      = source
        self.target_id      = target
        self.source_size    = source_size
        self.target_size    = target_size
        self.trainable      = trainable

        if initialize_weights_and_mask:

            self.weights = torch.nn.Parameter(
                torch.empty(target_size, source_size),
                requires_grad=self.trainable)

            self.register_buffer(
                "mask",
                torch.empty(target_size, source_size))

    def _get_connection_params(self, params, unique_id=None):
        """
        Extract connection initialization template, connectivity mask template, and baseline
        synaptic strength from configuration dictionary.

        Args:
            params (dict): Configuration parameter dictionary.
            unique_id (str | None, optional): Specific connection lookup key. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, float]: Tuple containing initial template tensor,
                mask template tensor, and baseline synaptic strength float.
        """
        init = torch.tensor(params['model']['connection_inits'][unique_id or self.conn_type])
        mask = torch.tensor(params['model']['connection_masks'][unique_id or self.conn_type])

        baseline_synaptic_strength = params['general']['synaptic_strength']['baseline']
        return init, mask, baseline_synaptic_strength

    def _init_weights(self, init, size_source, size_target, synapse_strength, std, scale):
        """
        Generate randomly sampled normal synaptic weight matrix shaped to `(size_target, size_source)`
        using specified mean, standard deviation, and scale.

        Args:
            init (torch.Tensor): Initial weight template tensor.
            size_source (int): Number of source columns/populations.
            size_target (int): Number of target columns/populations.
            synapse_strength (float): Synaptic strength multiplier.
            std (float): Standard deviation for Gaussian noise sampling.
            scale (float): Scaling factor applied to initial weights.

        Returns:
            torch.Tensor: Randomly initialized weight matrix of shape `(size_target, size_source)`.
        """
        init *= synapse_strength
        init = torch.tile(init, (size_target, size_source))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale
        return rand_weights

    def _make_receptive_field_mask(self, size_source, size_target, receptive_field_size, stride, grid_organization):
        """
        Construct receptive field mask enforcing 1D or 2D spatial connectivity constraints
        between source and target areas.

        Args:
            size_source (int): Number of source columns.
            size_target (int): Number of target columns.
            receptive_field_size (int | None): Size of receptive field window.
            stride (int): Stride step size of receptive field window.
            grid_organization (bool): Whether connectivity assumes a 2D spatial grid layout.

        Returns:
            torch.Tensor: Receptive field binary mask matrix of shape `(size_target, size_source)`.
        """
        receptive_field_mask = torch.zeros(size_target, size_source)

        if receptive_field_size is None:
            receptive_field_size = size_source

        if grid_organization:
            size_source = int(math.sqrt(size_source))  # assumes the source grid shape is perfectly square (x_shape==y_shape)

        nr_receptive_fields = math.ceil((size_source - receptive_field_size + 1) / stride)
        if grid_organization:
            nr_receptive_fields = nr_receptive_fields ** 2
        nr_cols_per_receptive_field = size_target // nr_receptive_fields

        assert size_target % nr_receptive_fields == 0, \
            f"The number of columns in the first area ({size_target}) can not be divided by the number of receptive fields ({nr_receptive_fields})."

        col_idx = 0
        end = size_source - receptive_field_size + 1

        for i in range(0, end, stride):

            if not grid_organization:  # if connectivity organization is one-dimensional
                for k in range(nr_cols_per_receptive_field):  # assign the same receptive field to n columns
                    receptive_field_mask[col_idx,
                    i:i + receptive_field_size] = 1.0  # set all receptive field indices to 1
                    col_idx += 1

            if grid_organization:  # if connectivity organization is two-dimensional (aka grid)
                for j in range(0, end, stride):
                    for k in range(nr_cols_per_receptive_field):  # assign the same receptive field to n columns
                        grid = torch.zeros(size_source, size_source)
                        grid[i:i+receptive_field_size, j:j+receptive_field_size] = 1.0  # set all receptive field indices to 1
                        receptive_field_mask[col_idx, :] = grid.flatten()
                        col_idx += 1

        return receptive_field_mask

    def _initialize_connection_between_areas(self, params, source_area, target_area, receptive_field_size, stride, grid_organization, std, scale):
        """
        Initialize weight and mask matrices for inter-area connections (feedforward, feedback, or lateral).

        Args:
            params (dict): Configuration parameters dictionary.
            source_area (BrainArea): Source brain area instance.
            target_area (BrainArea): Target brain area instance.
            receptive_field_size (int | None): Size of receptive field window.
            stride (int): Stride step size of receptive field window.
            grid_organization (bool): Whether connectivity assumes 2D grid organization.
            std (float): Standard deviation of initial weights.
            scale (float): Scaling factor for initial weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing weight matrix and mask matrix.
        """
        init, mask, synapse_strength = self._get_connection_params(params)

        size_source = source_area.num_columns
        size_target = target_area.num_columns

        init_weights = self._init_weights(init, size_source, size_target, synapse_strength, std, scale)

        receptive_field_mask = self._make_receptive_field_mask(size_source, size_target, receptive_field_size, stride, grid_organization)
        rf_mask = receptive_field_mask.repeat_interleave(8, dim=0).repeat_interleave(8, dim=1)

        mask = torch.tile(mask, (size_target, size_source))
        mask *= rf_mask

        weights = init_weights * mask
        return weights, mask

    def set_weights_and_mask(self, weights, mask):
        """
        Set weight matrix as a PyTorch Module Parameter and mask matrix as a PyTorch registered buffer.

        Args:
            weights (torch.Tensor): Initial weight matrix tensor.
            mask (torch.Tensor): Connectivity mask matrix tensor.
        """
        self.weights = torch.nn.Parameter(weights, requires_grad=self.trainable)
        self.register_buffer("mask", mask)

    def set_sizes(self, source_size, target_size):
        """
        Set source and target population counts.

        Args:
            source_size (int): Number of populations in source.
            target_size (int): Number of populations in target.
        """
        self.source_size = source_size
        self.target_size = target_size

    def get_name(self):
        """
        Generate unique formatted string name for the connection
        (`'{conn_type}_{source_id}_{target_id}'`).

        Returns:
            str: Connection name string.
        """
        return f'{self.conn_type}_{self.source_id}_{self.target_id}'

    def constrain(self, existing_areas):
        """
        Apply structural mask and Dale's law sign constraints to weight matrix, forcing excitatory
        connections to be non-negative and inhibitory connections to be non-positive.

        Args:
            existing_areas (Iterable[str]): Collection of initialized brain area names.
        """
        masked = self.weights * self.mask

        if self.source_id in existing_areas:

            exc = torch.relu(masked[:, 0::2])
            inh = -torch.relu(-masked[:, 1::2])

            ex_in_masked = masked.clone()
            ex_in_masked[:, 0::2] = exc
            ex_in_masked[:, 1::2] = inh

            self.W = ex_in_masked
        else:
            self.W = torch.relu(masked)

    def initialize_recurrent_weights(self, area):
        """
        Extract recurrent weight and mask matrices from a BrainArea instance.

        Args:
            area (BrainArea): Target brain area module.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of recurrent weights and internal mask tensors.
        """
        return area.recurrent_weights, area.internal_mask

    def initialize_background_weights(self, area):
        """
        Extract background drive weight and mask matrices from a BrainArea instance.

        Args:
            area (BrainArea): Target brain area module.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of background weights and mask tensors.
        """
        bg_weights = area.background_weights.unsqueeze(1)  # add extra dim
        mask = torch.ones_like(bg_weights)
        return bg_weights, mask

    def initialize_feedforward_weights(self, params, source_area, target_area, receptive_field_size, stride, grid_organization, std, scale):
        """
        Initialize feedforward connectivity weights and mask between source area and target area.

        Args:
            params (dict): Configuration parameters dictionary.
            source_area (BrainArea): Source brain area instance.
            target_area (BrainArea): Target brain area instance.
            receptive_field_size (int | None): Size of receptive field.
            stride (int): Stride step size of receptive field.
            grid_organization (bool): Whether 2D spatial grid connectivity applies.
            std (float): Standard deviation of initial weights.
            scale (float): Scaling factor for initial weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing initial weights and mask tensors.
        """
        return self._initialize_connection_between_areas(params, source_area, target_area, receptive_field_size, stride, grid_organization, std, scale)

    def initialize_feedback_weights(self, params, source_area, target_area, receptive_field_size, stride, grid_organization, std, scale):
        """
        Initialize feedback connectivity weights and mask between source area and target area.

        Args:
            params (dict): Configuration parameters dictionary.
            source_area (BrainArea): Source brain area instance.
            target_area (BrainArea): Target brain area instance.
            receptive_field_size (int | None): Size of receptive field.
            stride (int): Stride step size of receptive field.
            grid_organization (bool): Whether 2D spatial grid connectivity applies.
            std (float): Standard deviation of initial weights.
            scale (float): Scaling factor for initial weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing initial weights and mask tensors.
        """
        return self._initialize_connection_between_areas(params, source_area, target_area, receptive_field_size, stride, grid_organization, std, scale)

    def initialize_lateral_weights(self, params, area, receptive_field_size, stride, grid_organization, std, scale):
        """
        Initialize lateral connectivity weights and mask within an area, excluding intra-column
        connections.

        Args:
            params (dict): Configuration parameters dictionary.
            area (BrainArea): Target brain area module.
            receptive_field_size (int | None): Size of receptive field.
            stride (int): Stride step size of receptive field.
            grid_organization (bool): Whether 2D spatial grid connectivity applies.
            std (float): Standard deviation of initial weights.
            scale (float): Scaling factor for initial weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing lateral weights and mask tensors.
        """
        weights, mask = self._initialize_connection_between_areas(params, area, area, receptive_field_size, stride, grid_organization, std, scale)

        # Constrain mask and weights such that there are no column_intrinsic connections
        mask *= area.external_mask
        weights *= mask

        return weights, mask

    def initialize_input_weights(self, params, size_input, target_area, receptive_field_size, stride, grid_organization, std, scale):
        """
        Initialize external input weights and mask targeting a brain area.

        Args:
            params (dict): Configuration parameters dictionary.
            size_input (int): Dimension of external input vector.
            target_area (BrainArea): Target brain area module.
            receptive_field_size (int | None): Size of receptive field.
            stride (int): Stride step size of receptive field.
            grid_organization (bool): Whether 2D spatial grid connectivity applies.
            std (float): Standard deviation of initial weights.
            scale (float): Scaling factor for initial weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing input weights and mask tensors.
        """
        init, mask, synapse_strength = self._get_connection_params(params, self.source_id)
        init = torch.transpose(init.unsqueeze(0), 0, 1)
        mask = torch.transpose(mask.unsqueeze(0), 0, 1)
        size_target_area = target_area.num_columns

        rand_weights = self._init_weights(init, size_input, size_target_area, synapse_strength, std, scale)

        receptive_field_mask = self._make_receptive_field_mask(size_input, size_target_area, receptive_field_size, stride, grid_organization)
        rf_mask = receptive_field_mask.repeat_interleave(8, dim=0)  # only interleave for dimension target_area

        mask = torch.tile(mask, (size_target_area, size_input))
        mask *= rf_mask

        weights = rand_weights * mask
        return weights, mask

    def initialize_output_weights(self, params, source_area, std, scale):
        """
        Initialize task output readout weights and mask from a source brain area.

        Args:
            params (dict): Configuration parameters dictionary.
            source_area (BrainArea): Source brain area module.
            std (float): Standard deviation of initial weights.
            scale (float): Scaling factor for initial weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing output readout weights and mask
                tensors.
        """
        init, mask, _ = self._get_connection_params(params, self.target_id)
        size_source_area = source_area.num_columns

        rand_weights = self._init_weights(init, size_source_area, 1, 1.0, std, scale)

        mask = torch.tile(mask, (1, size_source_area))
        weights = rand_weights * mask
        return weights, mask
