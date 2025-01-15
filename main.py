from DMF import *
from DMF_single_column import *


# Run DMF single column; DMF double columns
double = False


# Functions to generate stimulus data
def ramp_input(start, end, T=1000):
    return np.linspace(start, end, T)

def sine_input(amplitude, frequency, T=1000, dt=1e-3):
    t = np.arange(0, T * dt, dt)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def square_wave_input(amplitude, frequency, T=1000, dt=1e-3):
    t = np.arange(0, T * dt, dt)
    return amplitude * (np.sign(np.sin(2 * np.pi * frequency * t)) + 1) / 2

def gaussian_pulse(amplitude, center, width, T=1000):
    t = np.linspace(0, T, T)
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))

def random_noise_input(amplitude, T=1000):
    return amplitude * np.random.randn(T)

def step_input(amplitude, step_time, T=1000):
    signal = np.zeros(T)
    signal[step_time:] = amplitude
    return signal


def make_stim(stim, T):
    # stim[2] *= np.array(ramp_input(0, 1, T))
    # stim[3] *= np.array(ramp_input(0, 1, T))

    # stim[2] *= np.array(sine_input(1, 1, T))
    # stim[3] *= np.array(sine_input(1, 1, T))

    # stim[2] *= square_wave_input(1, 1, T)
    # stim[3] *= square_wave_input(1, 1, T)

    # stim[2] *= gaussian_pulse(1, 500, 100, T)
    # stim[3] *= gaussian_pulse(1, 500, 100, T)

    # stim[2] *= random_noise_input(1, T)
    # stim[3] *= random_noise_input(1, T)

    stim[2] *= step_input(1, 50, T)
    stim[3] *= step_input(1, 50, T)

    return stim


if __name__ == '__main__':

    # Initialize the column model
    column = SingleColumnDMF(area='MT')

    # Make a stimulus input
    T = 1000
    stim = np.zeros((column.params['M'], T))
    stim = column.set_stim_ext(stim=stim, nu=20.0, params=column.params)
    stim = make_stim(stim, T)

    # Target other layers
    # stim[4] = np.array(ramp_input(0, 1, T)) * 500
    # stim[5] = np.array(ramp_input(0, 1, T)) * 500

    # Simulate the stimulation
    firing_rates = column.simulate(stim=stim, T=T, state_var='R')

    '''
    State variables cheat sheet
    'I': input current
    'A': adaptation
    'H': membrane potential
    'R': rate
    'N': noise
    '''

    if double:

        # Initialize the parameters
        params = get_params(J_local=0.13, J_lateral=0.172, area='MT')

        # Intialize the starting state (all zeros?)
        state = {}
        M = params['M']  # number of populations (=16)
        state['I'] = np.zeros(M)  # input current
        state['A'] = np.zeros(M)  # adaptation
        state['H'] = np.zeros(M)  # membrane potential
        state['R'] = np.zeros(M)  # rate
        state['N'] = np.zeros(M)  # noise

        # Initialize the stimulation
        stim = np.zeros(M)
        stim = set_vis(stim, column='H', nu=20.0, params=params)  # horizontal column
        stim = set_vis(stim, column='V', nu=20.0, params=params)  # vertical column

        stim = set_stimulation(stim, column='H', layer='L23', nu=20, params=params)
        stim = set_stimulation(stim, column='V', layer='L23', nu=20, params=params)

        stim = set_stimulation(stim, column='H', layer='L4', nu=20, params=params)
        stim = set_stimulation(stim, column='V', layer='L4', nu=20, params=params)

        stim = set_stimulation(stim, column='H', layer='L5', nu=20, params=params)
        stim = set_stimulation(stim, column='V', layer='L5', nu=20, params=params)

        stim = set_stimulation(stim, column='H', layer='L6', nu=20, params=params)
        stim = set_stimulation(stim, column='V', layer='L6', nu=20, params=params)

        # Total time steps
        T = 1000

        # Array for saving firing rate
        R = np.zeros((M, T))

        # Run simulation
        # note: stim does not change for the entirety of the simulation
        for t in range(T):
            state = update(state, params, stim)
            R[:, t] = state['R']

        # Plot the firing rate for each layer
        # column.plot_firing_rates(R[:8])
        # column.plot_firing_rates(R[8:])

        column.heatmap_over_time(R)
