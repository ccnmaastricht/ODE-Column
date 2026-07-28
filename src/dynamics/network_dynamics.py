import torch



class NetworkDynamics:

    """
    Handles the dynamics of the given network, gets called by NetworkSimulator > NetworkOdeWrapper.
    """

    def __init__(self, network):

        self.network = network

    def compute_firing_rate(self, x):
        """
        Compute the firing rates from (membrane potential - adaptation).
        """
        # TODO: clamping necessary? -> test

        x_nom = self.network.gain * x - self.network.threshold
        exp_input = -self.network.noise_factor * x_nom
        exp_input = torch.clamp(exp_input, -50, 50)  # CLAMP
        # exp_input = soft_clamp(exp_input)
        exp_term = torch.exp(exp_input)

        denom = 1 - exp_term + 1e-6  # ADD EPSILON
        x_activ = x_nom / denom
        return x_activ

    def soft_clamp(self, x, max_val=80):
        """ maybe this function can go. """
        return max_val * torch.tanh(x / max_val)

    def set_activities(self, t, fr_per_area, ext_input, input_windows):
        """
        Gather all activities in a dict; that includes the firing rates
        of all areas, background rate and external inputs.
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

    def compute_currents(self, t, firing_rates, ext_input, input_windows):
        """
        For each area, compute the current based on all incoming connections.
        """
        fr_per_area = {area_id: firing_rates[:, area_slice]
                       for area_id, area_slice in self.network.area_slices.items()}

        activities = self.set_activities(t, fr_per_area, ext_input, input_windows)

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
        State dynamics computing the derivative of the membrane potential and adaptation at time t.
        """
        # Unpack the state (membrane, adaptation) and compute firing rate
        mem_adap_split = self.network.num_populations
        membrane_potential, adaptation = state[:, :mem_adap_split], state[:, mem_adap_split:]

        firing_rate = self.compute_firing_rate(membrane_potential - adaptation)

        # Compute current
        total_current = self.compute_currents(t, firing_rate, ext_input, input_windows)

        # Compute derivative membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
                                    total_current * self.network.resistance) / self.network.membrane_time_constant
        delta_adaptation = (-adaptation + self.network.adaptation_strength_full *
                            firing_rate) / self.network.adapt_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation), dim=1)
        return state

    def diffusion(self, t, state):
        '''
        Diffusion function used by SDE, noise is only applied to membrane potential.
        '''
        g = torch.zeros_like(state)
        n = self.network.num_populations
        g[:, :n] = 3.0
        return g