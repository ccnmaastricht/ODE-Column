import torch


class NetworkDynamics:
    """
    Handles continuous-time network state dynamics and vector field computation
    by calculating right-hand side derivatives of state variables (membrane
    potential and adaptation) for ODE and SDE integration, as well as intermediate
    quantities such as population firing rates and synaptic currents.

    Args:
        network (BrainNetwork): Parent brain network model containing area parameters,
            registered buffers, connectivity matrices, and geometry slices.

    Attributes:
        network (BrainNetwork): Reference to the underlying network model instance.
    """

    def __init__(self, network):

        self.network = network

    def compute_firing_rate(self, x):
        """
        Compute neural population firing rates from net effective drive by applying
        a smooth sigmoidal activation function:
        \( r = \\frac{g \\cdot x - \\theta}{1 - \\exp(-\\beta (g \\cdot x - \\theta))} \).

        Args:
            x (torch.Tensor): Net effective input (membrane potential minus adaptation),
                shape `(batch_size, total_populations)`.

        Returns:
            torch.Tensor: Population firing rates in Hz,
                shape `(batch_size, total_populations)`.
        """
        # TODO: clamping necessary? -> test

        x_nom = self.network.gain * x - self.network.threshold
        exp_input = -self.network.noise_factor * x_nom
        exp_input = torch.clamp(exp_input, -50, 50)  # CLAMP
        # exp_input = _soft_clamp(exp_input)
        exp_term = torch.exp(exp_input)

        denom = 1 - exp_term + 1e-6  # ADD EPSILON
        x_activ = x_nom / denom
        return x_activ

    def _soft_clamp(self, x, max_val=80):
        """
        Apply soft bounds to input using hyperbolic tangent scaling.

        Args:
            x (torch.Tensor): Input tensor to clamp.
            max_val (float, optional): Maximum boundary value. Defaults to 80.

        Returns:
            torch.Tensor: Softly clamped tensor bounded within `(-max_val, max_val)`.
        """
        # TODO: decide if should keep this function
        return max_val * torch.tanh(x / max_val)

    def _set_activities(self, t, fr_per_area, ext_input, input_windows):
        """
        Aggregate activity sources (background, internal firing rates, external inputs)
        at time t.

        Args:
            t (torch.Tensor): Current simulation time scalar, shape `()` or `(1,)`.
            fr_per_area (dict[int | str, torch.Tensor]): Firing rate tensors partitioned
                per area ID, where each value has shape `(batch_size, area_populations)`.
            ext_input (dict[str, torch.Tensor] | None): Dictionary mapping external
                input names to their input tensors of shape `(batch_size, input_dims)`.
            input_windows (dict[str, tuple[float, float]]): Time window tuples
                `(start_time, end_time)` specifying when each external input is active.

        Returns:
            dict[str, torch.Tensor]: Combined dictionary mapping activity source
                names to current firing rate/drive tensors at time `t`.
        """
        # Add background drive
        activities = {'background': self.network.background_drive}

        # Add firing rates of all network areas
        activities.update(fr_per_area)

        # Add network-external input
        if ext_input is not None:
            for input_name, x in ext_input.items():
                # Present input if t is in input window
                start, end = input_windows[input_name]
                if start <= t.item() < end:
                    activities[input_name] = x
                else:
                    activities[input_name] = torch.zeros_like(x)

        return activities

    def _compute_currents(self, t, firing_rates, ext_input, input_windows):
        """
        Compute total incoming synaptic currents for all network areas at time t by
        multiplying source activities by connection weight matrices and scaling by
        synaptic time constants.

        Args:
            t (torch.Tensor): Current simulation time scalar, shape `()` or `(1,)`.
            firing_rates (torch.Tensor): Firing rate tensor across all network
                populations, shape `(batch_size, total_populations)`.
            ext_input (dict[str, torch.Tensor] | None): External drive input tensors
                mapped by name.
            input_windows (dict[str, tuple[float, float]]): Time intervals specifying
                active windows for external inputs.

        Returns:
            torch.Tensor: Concatenated synaptic currents for all populations ordered
                by area, shape `(batch_size, total_populations)`.
        """
        fr_per_area = {area_id: firing_rates[:, area_slice]
                       for area_id, area_slice in self.network.area_slices.items()}

        activities = self._set_activities(t, fr_per_area, ext_input, input_windows)

        currents = {area_id: torch.zeros(firing_rates.shape[0], area.num_populations, device=firing_rates.device)
                    for area_id, area in self.network.areas.items()}

        for connection in self.network.connections.values():
            source_fr = activities[connection.source_id]
            current = source_fr @ connection.W.T
            currents[connection.target_id] += current * self.network.synapse_time_constant

        total_current = torch.cat([currents[area_id] for area_id in self.network.area_order], dim=1)
        return total_current

    def forward(self, t, state, ext_input, input_windows):
        """
        Compute system state derivatives (d/dt) for ODE/SDE numerical integration by
        evaluating the differential equations for membrane potential \( V \)
        (\(\\tau_m \\frac{dV}{dt} = -V + I_{syn} \\cdot R\)) and spike-frequency
        adaptation \( a \) (\(\\tau_a \\frac{da}{dt} = -a + w_a \\cdot r\)).

        Args:
            t (torch.Tensor): Current time step scalar, shape `()` or `(1,)`.
            state (torch.Tensor): Combined system state containing membrane potentials
                and adaptation, shape `(batch_size, 2 * total_populations)` where
                `state[:, :N]` is membrane potential and `state[:, N:]` is adaptation.
            ext_input (dict[str, torch.Tensor] | None): External drive inputs mapped
                by name.
            input_windows (dict[str, tuple[float, float]]): Time window boundaries
                for external inputs.

        Returns:
            torch.Tensor: Concatenated state derivatives `d(state)/dt`,
                shape `(batch_size, 2 * total_populations)`.
        """
        # Unpack the state (membrane, adaptation) and compute firing rate
        mem_adap_split = self.network.num_populations
        membrane_potential, adaptation = state[:, :mem_adap_split], state[:, mem_adap_split:]

        firing_rate = self.compute_firing_rate(membrane_potential - adaptation)

        # Compute current
        total_current = self._compute_currents(t, firing_rate, ext_input, input_windows)

        # Compute derivative membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
                                    total_current * self.network.resistance) / self.network.membrane_time_constant
        delta_adaptation = (-adaptation + self.network.adaptation_strength_full *
                            firing_rate) / self.network.adapt_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation), dim=1)
        return state

    def diffusion(self, t, state):
        """
        Compute state-dependent diffusion matrix for stochastic differential equations
        (SDEs), applying stochastic Gaussian noise exclusively to membrane potential
        equations.

        Args:
            t (torch.Tensor): Current time step scalar.
            state (torch.Tensor): System state tensor of shape
                `(batch_size, 2 * total_populations)`.

        Returns:
            torch.Tensor: Noise amplitude tensor of shape
                `(batch_size, 2 * total_populations)`, with non-zero noise applied
                to membrane potential components.
        """
        g = torch.zeros_like(state)
        n = self.network.num_populations
        g[:, :n] = 3.0
        return g