import torch


class NetworkOdeWrapper(torch.nn.Module):
    """
    Wraps a BrainNetwork and its dynamic rate functions for interface compatibility
    with numerical ODE/SDE solvers (torchdiffeq and torchsde).

    Args:
    network (BrainNetwork): Target brain network module.
    ext_input (dict[str, torch.Tensor] | None): External drive inputs.
    input_windows (dict[str, tuple[float, float]]): Active time windows for
        external inputs.

    Attributes:
        network (BrainNetwork): Reference to the underlying network model.
        ext_input (dict[str, torch.Tensor] | None): External input drive dictionary.
        input_windows (dict[str, tuple[float, float]]): Active time windows for
            external inputs.
        noise_type (str): SDE noise specification class parameter set to "diagonal".
        sde_type (str): SDE integration scheme class parameter set to "ito".
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, network, ext_input, input_windows):

        super().__init__()

        self.network = network
        self.ext_input = ext_input
        self.input_windows = input_windows

    def forward(self, t, state):
        """
        Evaluate drift derivative step d(state)/dt by delegating to network dynamics.

        Args:
            t (torch.Tensor): Current simulation time scalar.
            state (torch.Tensor): System state tensor of shape
                `(batch_size, 2 * total_populations)`.

        Returns:
            torch.Tensor: Computed state derivative tensor `d(state)/dt` of shape
                `(batch_size, 2 * total_populations)`.
        """
        return self.network.dynamics.forward(
            t,
            state,
            self.ext_input,
            self.input_windows)

    def diffusion(self, t, state):
        """
        Evaluate diffusion noise amplitude matrix by delegating to network dynamics.

        Args:
            t (torch.Tensor): Current simulation time scalar.
            state (torch.Tensor): System state tensor of shape
                `(batch_size, 2 * total_populations)`.

        Returns:
            torch.Tensor: Diffusion noise amplitude tensor of shape
                `(batch_size, 2 * total_populations)`.
        """
        return self.network.dynamics.diffusion(
            t,
            state)
