# Training on the current
def forward(self, t, current, stim):

    # no dts for ode
    # current = current + (-current / self.tau_s)  # self inhibition
    # current = current + (torch.matmul(self.W, self.state['R'].detach()))  # recurrent input
    # current = current + self.W_bg * self.nu_bg  # background input
    # current = current + stim  # external output
    # self.state['I'] = current
    #
    # # Update the membrane potential and adaptation
    # self.state['H'] = self.state['H'] + ((-self.state['H'] + self.R_ * current) / self.tau_m)
    # self.state['A'] = self.state['A'] + ((-self.state['A'] + self.state['R'] * self.kappa) / self.tau_a)
    # return current
    #
    # # Update firing rate
    # self.state['R'] = self.activation(self.state['H'] - self.state['A'])
    #
    # return current  # return the current

    # Update the current
    current = current + self.dt * (-current / self.tau_s)  # self inhibition
    current = current + self.dt * (torch.matmul(self.W, self.state['R'].detach()))  # recurrent input
    current = current + self.dt * self.W_bg * self.nu_bg  # background input
    current = current + self.dt * stim  # external output
    self.state['I'] = current

    # Update the membrane potential and adaptation
    self.state['H'] = self.state['H'] + self.dt * ((-self.state['H'] + self.R_ * current) / self.tau_m)
    self.state['A'] = self.state['A'] + self.dt * ((-self.state['A'] + self.state['R'] * self.kappa) / self.tau_a)

    # Update firing rate
    self.state['R'] = self.activation(self.state['H'] - self.state['A'])

    return current  # return the current
