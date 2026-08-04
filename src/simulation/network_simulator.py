import torch
from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint, sdeint_adjoint

from src.simulation.ode_wrapper import NetworkOdeWrapper


class NetworkSimulator:
    """
    Manages continuous-time neural network simulations using deterministic (ODE)
    or stochastic (SDE) differential equation solvers.

    Args:
        network (BrainNetwork): Target brain network module.

    Attributes:
        network (BrainNetwork): Reference to the target brain network.
        dt (float): Simulation integration time step size in seconds.
        sim_time (float): Total simulation duration in seconds.
        input_window (tuple[float, float]): Default time window `(start, end)` during
            which inputs are active.
        previous_network_state (torch.Tensor | None): Stored final state from previous
            simulation run.
        network_is_ready (bool): Flag indicating whether network device placement
            and state initialization have been completed.
    """

    def __init__(self, network):

        self.network        = network

        time_params         = network.params['model']['time_params']
        self.dt             = time_params['dt']
        self.sim_time       = time_params['sim_time']
        self.input_window   = time_params['input_window']

        self.previous_network_state = None

        self.network_is_ready = False

    def _infer_batch_size(self, ext_input):
        """
        Infer batch size dimension from input tensor shapes.

        Args:
            ext_input (dict[str, torch.Tensor]): Dictionary of external input tensors.

        Returns:
            int: Inferred batch size (size of first tensor dimension).
        """
        first_input = next(iter(ext_input.values()))
        return first_input.shape[0]

    def _validate_input_shapes(self, ext_input):
        """
        Validate that all external input tensors share a consistent batch size in
        their first dimension.

        Args:
            ext_input (dict[str, torch.Tensor]): Dictionary of external input tensors.

        Raises:
            AssertionError: Raised if input tensors have mismatched batch sizes.
        """
        batch_sizes = {tensor.shape[0]
                       for tensor in ext_input.values()}
        assert len(batch_sizes) == 1, f"Input tensors have inconsistent batch sizes: {batch_sizes}"

    def _make_input_dict(self, input_var, input_keys):
        """
        Ensure input variables or window specifications are structured as a dictionary
        mapped by key.

        Args:
            input_var (dict | torch.Tensor | tuple): Single input variable/window or
                pre-formatted dictionary.
            input_keys (list[str] | dict_keys): List of keys to map non-dictionary
                inputs to.

        Returns:
            dict: Dictionary mapping specified keys to input variables.
        """
        if not isinstance(input_var, dict):
            input_var = {input_key: input_var for input_key in input_keys}
        return input_var

    def _convert_to_tensors(self, ext_input, device):
        """
        Convert input arrays/values into float32 PyTorch tensors, move them to the
        target compute device, and unsqueeze single samples into batch dimension.

        Args:
            ext_input (dict[str, Any]): Dictionary containing raw input tensors,
                arrays, or scalars.
            device (torch.device | str): Target compute device (e.g., 'cpu' or 'cuda').

        Returns:
            dict[str, torch.Tensor]: Dictionary containing device-allocated 2D input
                tensors of shape `(batch_size, input_dim)`.
        """
        for name, input_var in ext_input.items():

            if torch.is_tensor(input_var):
                input_on_device = input_var.to(device)
            else:
                input_on_device = torch.tensor(input_var, dtype=torch.float32, device=device)

            if input_on_device.ndim == 1:
                input_on_device = input_on_device.unsqueeze(0)

            ext_input[name] = input_on_device

        return ext_input

    def _validate_network_inputs(self, ext_input):
        """
        Validate that provided input names match network input connection sources and
        feature dimensions align with input weight matrices.

        Args:
            ext_input (dict[str, torch.Tensor]): Processed input tensors mapped by
                source ID.

        Raises:
            AssertionError: Raised if input keys do not match input connections or
                feature dimensions mismatch.
        """
        input_connections = {conn.source_id: conn
                         for conn in self.network.connections.values()
                         if conn.conn_type == "input"}

        assert ext_input.keys() == input_connections.keys(), (f"Mismatch found between specified inputs ({list(ext_input.keys())}) "
                                                              f"and specified input connection sources ({list(input_connections.keys())}).")

        for source_id, connection in input_connections.items():
            assert ext_input[source_id].shape[1] == connection.weights.shape[1], (f"Input tensor '{source_id}' shape (x, {ext_input[source_id].shape[1]}) "
                                                                                  f"does not match with input weights shape (x, {connection.weights.shape[1]}).")

    def _validate_input_windows(self, input_window):
        """
        Validate that active input time window bounds do not exceed total simulation time.

        Args:
            input_window (dict[str, tuple[float, float]]): Dictionary mapping input
                names to `(start, end)` time intervals.

        Raises:
            AssertionError: Raised if input window start or end time exceeds total
                simulation duration.
        """
        for window_i, window_tuple in input_window.items():
            start = window_tuple[0]
            end = window_tuple[1]
            assert start <= self.sim_time and end <= self.sim_time, (f"The input time window ({start}s, {end}s) "
                                                                     f"exceeds total simulation time ({self.sim_time}s)")

    def _prepare_input_for_sim(self, ext_input, input_window, device):
        """
        Format external inputs and time windows as dictionaries on the target device
        and perform validation against network architecture.

        Args:
            ext_input (dict | torch.Tensor | None): External input specification.
            input_window (dict | tuple | None): Active time window specification.
            device (torch.device | str): Target compute device.

        Returns:
            tuple[dict[str, torch.Tensor] | None, dict[str, tuple[float, float]] | None, int]:
                Tuple containing processed input dictionary, time window dictionary,
                and inferred batch size.
        """
        if ext_input is None:
            return None, None, 1

        if input_window is None:
            input_window = self.input_window

        ext_input = self._make_input_dict(ext_input, ['input'])
        input_window = self._make_input_dict(input_window, ext_input.keys())

        ext_input = self._convert_to_tensors(ext_input, device)

        self._validate_input_shapes(ext_input)
        self._validate_network_inputs(ext_input)
        self._validate_input_windows(input_window)

        assert ext_input.keys() == input_window.keys(), (f"Mismatch found between dictionaries of inputs ({list(ext_input.keys())}) "
                                                         f"and their time windows ({list(input_window.keys())}).")

        batch_size = self._infer_batch_size(ext_input)
        return ext_input, input_window, batch_size

    def _extend_init_state(self, batch_size):
        """
        Tile network initial state vector across batch dimension to match simulation
        batch size.

        Args:
            batch_size (int): Target batch size.

        Returns:
            torch.Tensor: Tiled initial state matrix of shape
                `(batch_size, 2 * total_populations)`.
        """
        return torch.tile(self.initial_state, (batch_size, 1))

    def _prepare_network(self, device):
        """
        Finalize network layout, move network parameters to target device, construct
        time vector, and initialize zeroed state.

        Args:
            device (torch.device | str): Target compute device.
        """
        self.network = self.network.to(device)
        self.network.finalize()

        self.time_vec = torch.arange(0, self.sim_time, self.dt, device=device)
        self.initial_state = torch.zeros(1, self.network.num_populations * 2, device=device)

        self.network_is_ready = True

    def _initialize_network_activity(self):
        """
        Simulate network dynamics without input to set initial membrane potentials
        to resting state before primary simulation run.
        """
        with torch.no_grad():
            self.network.constrain_weights()

            sim_wrapper = NetworkOdeWrapper(self.network, None, None)
            resting_state = odeint(sim_wrapper, self.initial_state, self.time_vec)

            membrane_potential_resting_state = resting_state[-1, :, :self.network.num_populations]
            self.initial_state[:, :self.network.num_populations] = membrane_potential_resting_state

    def _store_last_state(self, network_output):
        """
        Store final network state from output tensor for potential initial state reuse
        in subsequent simulations.

        Args:
            network_output (torch.Tensor): Simulation output tensor of shape
                `(time_steps, batch_size, 2 * total_populations)`.
        """
        self.previous_network_state = network_output[-1]

    def run(self, ext_input, input_window, adjoint, stochastic, reset_state, device):
        """
        Execute numerical simulation using standard ODE, adjoint ODE, SDE, or adjoint
        SDE solvers.

        Args:
            ext_input (dict | torch.Tensor | None): External input drive tensors.
            input_window (dict | tuple | None): Active time window interval tuples.
            adjoint (bool): Whether to use adjoint sensitivity method solver variants
                (`odeint_adjoint` / `sdeint_adjoint`).
            stochastic (bool): Whether to simulate stochastic dynamics with noise
                (`sdeint`) or deterministic dynamics (`odeint`).
            reset_state (bool): Whether to reset initial state to resting state or
                carry over `previous_network_state`.
            device (torch.device | str): Target compute device for execution.

        Returns:
            torch.Tensor: Simulation output trajectory tensor of shape
                `(time_steps, batch_size, 2 * total_populations)`.
        """
        # Prepare the external input for simulation and infer the batch size
        ext_input, input_window, batch_size = self._prepare_input_for_sim(ext_input, input_window, device)

        # If this is the first simulation run, finalize the network, put everything to the device
        # and run the network without input to get resting state membrane potential
        if not self.network_is_ready:
            self._prepare_network(device)
            self._initialize_network_activity()

        # Constrain all network weights to ensure no illegal connections can be used
        self.network.constrain_weights()

        # Extend the initial state to match with the batch size
        initial_state = self._extend_init_state(batch_size)

        # If specified, use the last state of the last simulation as the initial state
        if not reset_state and self.previous_network_state is not None:
            initial_state = self.previous_network_state

        # Initialize the wrapper and set the external input
        sim_wrapper = NetworkOdeWrapper(self.network, ext_input, input_window)

        # Run the network simulation with the specified ODE variant
        if not adjoint and not stochastic:
            network_output = odeint(sim_wrapper, initial_state, self.time_vec)

        elif adjoint and not stochastic:
            network_output = odeint_adjoint(sim_wrapper, initial_state, self.time_vec)

        elif not adjoint and stochastic:
            network_output = sdeint(sim_wrapper, initial_state, self.time_vec,
                            names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

        elif adjoint and stochastic:
            network_output = sdeint_adjoint(sim_wrapper, initial_state, self.time_vec,
                                    names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

        self._store_last_state(network_output)

        return network_output
