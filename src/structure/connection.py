import torch
import math



class Connection(torch.nn.Module):

    """
    Connections allow source activity to influence target areas (between areas) or
    target populations (within areas). Connections include recurrent (intrinsic),
    background, feedforward, feedback, lateral and input.
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

    def _get_connection_params(self, params):
        """
        Obtains the relevant parameters from the params dict to establish
        the connection.
        """
        init = torch.tensor(params['model']['connection_inits'][self.conn_type])
        mask = torch.tensor(params['model']['connection_masks'][self.conn_type])

        baseline_synaptic_strength = params['general']['synaptic_strength']['baseline']
        return init, mask, baseline_synaptic_strength

    def _init_weights(self, init, size_source, size_target, synapse_strength, std, scale):
        """
        Initialize random weights and fit them to shape (target, source).
        """
        init *= synapse_strength
        init = torch.tile(init, (size_target, size_source))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale
        return rand_weights

    def _make_receptive_field_mask(self, size_source, size_target, receptive_field_size, stride, grid_organization):
        """
        Create a receptive field mask to constrain which connections can
        be made between a source and target area.
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
        Initialize connection weights between two BrainArea objects.
        Gets called to initialize feedforward, feedback and lateral connection weights.
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
        Set the connection weights and mask as object attributes.
        """
        self.weights = torch.nn.Parameter(weights, requires_grad=self.trainable)
        self.register_buffer("mask", mask)

    def set_sizes(self, source_size, target_size):
        """
        Set the sizes (number of populations) of the source and target.
        """
        self.source_size = source_size
        self.target_size = target_size

    def get_name(self):
        """
        Returns the unique string specifying the connection.
        """
        return f'{self.conn_type}_{self.source_id}_{self.target_id}'

    def constrain(self, existing_areas):
        """
        Constrain the connection so no illegal connections can be used. Uses the
        connection mask and, if the connection source is a BrainArea, it forces
        excitatory connections to be non-negative and inhibitory ones to be non-positive.
        """
        masked = self.weights * self.mask

        if self.source_id in existing_areas:

            exc = torch.relu(masked[:, 0::2])
            inh = -torch.relu(-masked[:, 1::2])

            ex_in_masked = masked.clone()
            ex_in_masked[:, 0::2] = exc
            ex_in_masked[:, 1::2] = inh

            self.W = masked
        else:
            self.W = torch.relu(masked)

    def initialize_recurrent_weights(self, area):
        """
        Initialize recurrent (i.e. column-intrinsic) weights within the same area.
        """
        return area.recurrent_weights, area.internal_mask

    def initialize_background_weights(self, area):
        """
        Initialize background weights within an area.
        """
        bg_weights = area.background_weights.unsqueeze(1)  # add extra dim
        mask = torch.ones_like(bg_weights)
        return bg_weights, mask

    def initialize_feedforward_weights(self, params, source_area, target_area, receptive_field_size, stride, grid_organization, std, scale):
        """
        Initialize feedforward weights between source area and target area.
        """
        return self._initialize_connection_between_areas(params, source_area, target_area, receptive_field_size, stride, grid_organization, std, scale)

    def initialize_feedback_weights(self, params, source_area, target_area, receptive_field_size, stride, grid_organization, std, scale):
        """
        Initialize feedback weights between source area and target area.
        """
        return self._initialize_connection_between_areas(params, source_area, target_area, receptive_field_size, stride, grid_organization, std, scale)

    def initialize_lateral_weights(self, params, area, receptive_field_size, stride, grid_organization, std, scale):
        """
        Initialize lateral weights within an area.
        """
        weights, mask = self._initialize_connection_between_areas(params, area, area, receptive_field_size, stride, grid_organization, std, scale)

        # Constrain mask and weights such that there are no column_intrinsic connections
        mask *= area.external_mask
        weights *= mask

        return weights, mask

    def initialize_input_weights(self, params, size_input, target_area, receptive_field_size, stride, grid_organization, std, scale):
        """
        Initialize input weights targeting an area.
        """
        init, mask, synapse_strength = self._get_connection_params(params)
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
        Initialize output weights reading out activity from an area.
        """
        init, mask, _ = self._get_connection_params(params)
        size_source_area = source_area.num_columns

        rand_weights = self._init_weights(init, size_source_area, 1, 1.0, std, scale)

        mask = torch.tile(mask, (1, size_source_area))
        weights = rand_weights * mask
        return weights, mask

