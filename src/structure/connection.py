import torch



class Connection(torch.nn.Module):

    """
    Connections allow source activity to influence target areas (between areas) or
    target populations (within areas). Connections include recurrent (intrinsic),
    background, feedforward, feedback, lateral and input.
    """

    def __init__(self, conn_type, source, target, trainable):
        super().__init__()

        self.conn_type  = conn_type
        self.source_id  = source
        self.target_id  = target
        self.trainable  = trainable
        self.size_input = None

    def _get_connection_params(self, params):
        """
        Obtains the relevant parameters from the params dict to establish
        the connection.
        """
        init = torch.tensor(params['model']['connection_inits'][self.conn_type])
        mask = torch.tensor(params['model']['connection_masks'][self.conn_type])

        baseline_synaptic_strength = params['general']['synaptic_strength']['baseline']
        return init, mask, baseline_synaptic_strength

    def _set_weights_and_mask(self, weights, mask):
        """ ... """
        self.weights = torch.nn.Parameter(weights, requires_grad=self.trainable)
        self.register_buffer("mask", mask)

    def get_name(self):
        """ Returns the unique string specifying the connection."""
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
        """ Initialize recurrent (i.e. column-intrinsic) weights within the same area."""
        self._set_weights_and_mask(area.recurrent_weights, area.internal_mask)

    def initialize_background_weights(self, area):
        """ Initialize background weights within an area."""
        bg_weights = area.background_weights.unsqueeze(1)  # add extra dim
        mask = torch.ones_like(bg_weights)
        self._set_weights_and_mask(bg_weights, mask)

    def initialize_feedforward_weights(self, params, source_area, target_area, std, scale):
        """
        Initialize feedforward weights between source area and target area.
        """
        init, mask, synapse_strength = self._get_connection_params(params)

        size_source = source_area.num_columns
        size_target = target_area.num_columns

        init *= synapse_strength
        init = torch.tile(init, (size_target, size_source))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        mask = torch.tile(mask, (size_target, size_source))
        weights = rand_weights * mask
        self._set_weights_and_mask(weights, mask)

    def initialize_feedback_weights(self, params, source_area, target_area, std, scale):
        """
        Initialize feedback weights between source area and target area.
        """
        init, mask, synapse_strength = self._get_connection_params(params)

        size_source = source_area.num_columns
        size_target = target_area.num_columns

        init *= synapse_strength
        init = torch.tile(init, (size_target, size_source))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        mask = torch.tile(mask, (size_target, size_source))
        weights = rand_weights * mask
        self._set_weights_and_mask(weights, mask)

    def initialize_lateral_weights(self, params, area, std, scale):
        """
        Initialize lateral weights within an area.
        """
        init, mask, synapse_strength = self._get_connection_params(params)
        size_area = area.num_columns

        init *= synapse_strength
        init = torch.tile(init, (size_area, size_area))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        mask = torch.tile(mask, (size_area, size_area)) * area.external_mask
        weights = rand_weights * mask
        self._set_weights_and_mask(weights, mask)

    def initialize_input_weights(self, params, size_input, target_area, std, scale):
        """
        Initialize input weights targeting an area.
        """
        self.size_input = size_input

        init, mask, synapse_strength = self._get_connection_params(params)
        init = torch.transpose(init.unsqueeze(0), 0, 1)
        mask = torch.transpose(mask.unsqueeze(0), 0, 1)
        size_target_area = target_area.num_columns

        init *= synapse_strength
        init = torch.tile(init, (size_target_area, size_input))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        mask = torch.tile(mask, (size_target_area, size_input))
        weights = rand_weights * mask
        self._set_weights_and_mask(weights, mask)

    def initialize_output_weights(self, params, source_area, std, scale):
        """
        Initialize output weights reading out activity from an area.
        """
        init, mask, _ = self._get_connection_params(params)
        size_source_area = source_area.num_columns

        init = torch.tile(init, (1, size_source_area))

        rand_weights = abs(torch.normal(mean=init, std=std))
        rand_weights *= scale

        mask = torch.tile(mask, (1, size_source_area))
        weights = rand_weights * mask
        self._set_weights_and_mask(weights, mask)

