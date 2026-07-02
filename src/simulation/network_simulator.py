import torch
from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint, sdeint_adjoint

from src.simulation.ode_wrapper import NetworkOdeWrapper



class NetworkSimulator:

    """
    Handles the simulation of the network activity.
    """

    def __init__(self, network):

        self.network        = network

        time_params         = network.params['model']['time_params']
        self.dt             = time_params['dt']
        self.sim_time       = time_params['sim_time']
        self.input_window   = time_params['input_window']

        self.network_is_ready = False

    def _infer_batch_size(self, ext_input):
        """
        Infer the batch size from the shape of the input tensors.
        """
        first_input = next(iter(ext_input.values()))
        return first_input.shape[0]

    def _validate_input_shapes(self, ext_input):
        """
        Check if input tensors have the same shape in the first dimension (i.e. batch size).
        """
        batch_sizes = {tensor.shape[0]
                       for tensor in ext_input.values()}
        assert len(batch_sizes) == 1, f"Input tensors have inconsistent batch sizes: {batch_sizes}"

    def _make_input_dict(self, input_var):
        """
        Convert the input variable to a dictionary.
        """
        if not isinstance(input_var, dict):
            input_var = {"input": input_var}
        return input_var

    def _convert_to_tensors(self, ext_input, device):
        """
        Convert each input tensor to a tensor and put them on the specified device.
        If an input tensor is one-dimensional (aka one sample in a batch), it adds
        an extra dimension.
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
        Check if the inputs have an assigned connection that targets an area and if the
        input tensor has the correct shape in the second dimension (should match with connection.weights)
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
        Check if input window does not exceed the simulation time.
        """
        for window_i, window_tuple in input_window.items():
            start = window_tuple[0]
            end = window_tuple[1]
            assert start <= self.sim_time and end <= self.sim_time, (f"The input time window ({start}s, {end}s) "
                                                                     f"exceeds total simulation time ({self.sim_time}s)")

    def _prepare_input_for_sim(self, ext_input, input_window, device):
        """
        Prepare the input samples and the input time window for simulation:
        they need to be formatted as dictionaries, inputs as tensors and on the device,
        check if compatible with network architecture.
        """
        if ext_input is None:
            return None, None, 1

        if input_window is None:
            input_window = self.input_window

        ext_input = self._make_input_dict(ext_input)
        input_window = self._make_input_dict(input_window)

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
        Extend the initial state to fit with the batch size
        """
        return torch.tile(self.initial_state, (batch_size, 1))

    def _prepare_network(self, device):
        """
        Finalizes the network and brings the network, time vector and initial state
        to the specified device before the first batch is run through the network.
        """
        self.network = self.network.to(device)
        self.network.finalize()

        self.time_vec = torch.arange(0, self.sim_time, self.dt, device=device)
        self.initial_state = torch.zeros(1, self.network.num_populations * 2, device=device)

        self.network_is_ready = True

    def run(self, ext_input, input_window, adjoint, stochastic, device):
        """
        Runs the network simulation.
        """
        if not self.network_is_ready:
            self._prepare_network(device)

        ext_input, input_window, batch_size = self._prepare_input_for_sim(ext_input, input_window, device)
        sim_wrapper = NetworkOdeWrapper(self.network, ext_input, input_window)

        initial_state = self._extend_init_state(batch_size)

        self.network.constrain_weights()

        if not adjoint and not stochastic:
            return odeint(sim_wrapper, initial_state, self.time_vec)

        elif adjoint and not stochastic:
            return odeint_adjoint(sim_wrapper, initial_state, self.time_vec)

        elif not adjoint and stochastic:
            return sdeint(sim_wrapper, initial_state, self.time_vec,
                            names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

        elif adjoint and stochastic:
            return sdeint_adjoint(sim_wrapper, initial_state, self.time_vec,
                                    names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

