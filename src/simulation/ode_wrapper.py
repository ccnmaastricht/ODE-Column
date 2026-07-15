import torch



class NetworkOdeWrapper(torch.nn.Module):

    """
    Sets the external input with initialization and runs the network dynamics
    when called by odeint/sdeint.
    """

    noise_type = "diagonal"  # sde params
    sde_type = "ito"

    def __init__(self, network, ext_input, input_windows):
        super().__init__()

        self.network = network
        self.ext_input = ext_input
        self.input_windows = input_windows

        self.nfe = 0

    def forward(self, t, state):

        self.nfe += 1
        
        return self.network.dynamics.forward(
            t,
            state,
            self.ext_input,
            self.input_windows)

    def diffusion(self, t, state):
        return self.network.dynamics.diffusion(
            t,
            state)

