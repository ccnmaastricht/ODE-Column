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
        r"""
        Compute neural population firing rates from net effective drive by applying
        a smooth sigmoidal / threshold-linear activation function:
        \[r = \frac{g \cdot x - \theta}{1 - \exp(-d (g \cdot x - \theta))}\]
        where \( x = V - a \) is the net effective drive (membrane potential minus
        rate adaptation), \( g \) is the gain, \( \theta \) is the threshold, and
        \( d \) is the noise factor (sometimes denoted \( \beta \)).
        Numerical Stability:
            The analytic limit as \( g \cdot x - \theta \to 0 \) is \( 1/d \approx 112.36\text{ Hz} \).
            To avoid catastrophic cancellation and division-by-zero, the denominator
            is evaluated using `torch.expm1(-z)`. For small arguments (\( |z| < 10^{-3} \)
            where \( z = d(g \cdot x - \theta) \)), a first-order Taylor expansion
            \( r \approx \frac{1 + 0.5 z}{d} \) is used to guarantee continuity and stable gradients.
        Args:
            x (torch.Tensor): Net effective input (membrane potential minus adaptation),
                shape `(batch_size, total_populations)`.
        Returns:
            torch.Tensor: Population firing rates in Hz,
                shape `(batch_size, total_populations)`.
        """
        x_nom = self.network.gain * x - self.network.threshold
        d = self.network.noise_factor
        z = torch.clamp(d * x_nom, -80.0, 80.0)

        small = z.abs() < 1e-3
        z_safe = torch.where(small, torch.ones_like(z), z)

        rate = x_nom / (-torch.expm1(-z_safe))  # Exact formula: x / (1 - exp(-d*x))
        rate_taylor = (1.0 + 0.5 * z) / d  # Taylor limit at z -> 0: (1/d) * (1 + z/2)

        return torch.where(small, rate_taylor, rate)

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
        activities = {'background': self.network.background_drive}

        activities.update(fr_per_area)

        if ext_input is not None:

            for input_name, x in ext_input.items():
                start, end = input_windows[input_name]

                if start <= t.item() < end:
                    activities[input_name] = x
                else:
                    activities[input_name] = torch.zeros_like(x)

        return activities

    def _compute_currents(self, t, firing_rates, ext_input, input_windows):
        """
        Compute incoming synaptic drive for all network populations.

        Args:
            t: Current simulation time.
            firing_rates: Tensor of shape (batch_size, total_populations).
            ext_input: External drive inputs mapped by name.
            input_windows: Active time windows for external inputs.

        Returns:
            Tensor of shape (batch_size, total_populations).
        """
        fr_per_area = {
            area_id: firing_rates[:, area_slice]
            for area_id, area_slice in self.network.area_slices.items()
        }

        activities = self._set_activities(
            t, fr_per_area, ext_input, input_windows
        )

        drives = {
            area_id: torch.zeros(
                firing_rates.shape[0],
                area.num_populations,
                device=firing_rates.device,
                dtype=firing_rates.dtype,
            )
            for area_id, area in self.network.areas.items()
        }

        for connection in self.network.connections.values():
            if (
                    connection.source_id not in activities
                    or connection.conn_type == "output"
            ):
                continue

            source_fr = activities[connection.source_id]
            drive = source_fr @ connection.W.T

            drives[connection.target_id] += drive

        total_drive = torch.cat(
            [drives[area_id] for area_id in self.network.area_order],
            dim=1,
        )

        return total_drive

    def forward(self, t, state, ext_input, input_windows):
        """
        Compute derivatives of synaptic current, membrane potential, adaptation.

        State shape:
            (batch_size, 3 * total_populations)

        State layout:
            state[:, :N]       -> synaptic current
            state[:, N:2*N]    -> membrane potential
            state[:, 2*N:3*N]  -> adaptation
        """
        N = self.network.num_populations

        # Unpack the state
        synaptic_current = state[:, :N]
        membrane_potential = state[:, N:2 * N]
        adaptation = state[:, 2 * N:3 * N]

        # Compute firing rates from membrane potential and adaptation
        firing_rate = self.compute_firing_rate(
            membrane_potential - adaptation
        )

        # Compute incoming synaptic drive
        total_drive = self._compute_currents(
            t, firing_rate, ext_input, input_windows
        )

        # Synaptic current dynamics
        delta_synaptic_current = (
                -synaptic_current / self.network.synapse_time_constant
                + total_drive
        )

        # Membrane potential dynamics
        delta_membrane_potential = (
                                           -membrane_potential
                                           + synaptic_current * self.network.resistance
                                   ) / self.network.membrane_time_constant

        # Adaptation dynamics
        delta_adaptation = (
                                   -adaptation
                                   + self.network.adaptation_strength_full * firing_rate
                           ) / self.network.adapt_time_constant


        # Concatenate derivatives in the same order as the state
        delta_state = torch.cat(
            (
                delta_synaptic_current,
                delta_membrane_potential,
                delta_adaptation,
            ),
            dim=1,
        )
        return delta_state

    def diffusion(self, t, state, noise_amplitude):
        """
        Compute state-dependent diffusion matrix for stochastic differential equations
        (SDEs), applying stochastic Gaussian noise exclusively to membrane potential
        equations.

        Args:
            t (torch.Tensor): Current time step scalar.
            state (torch.Tensor): System state tensor of shape
                `(batch_size, 2 * total_populations)`.
            noise_amplitude (float): Noise amplitude (standard deviation 𝜎)
                applied to the membrane potential dynamics.

        Returns:
            torch.Tensor: Noise amplitude tensor of shape
                `(batch_size, 2 * total_populations)`, with non-zero noise applied
                to membrane potential components.
        """
        g = torch.zeros_like(state)
        n = self.network.num_populations
        g[:, :n] = noise_amplitude
        return g