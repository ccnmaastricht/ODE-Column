import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

"""
Dynamic Mean Field

@author: Kris Evers
"""


def get_params(J_local=87.8e-3, J_lateral=87.8e-3, area='MT'):
    """
    Args:
    J_local (float):    local synaptic strength
    J_lateral (float):  lateral synaptic strength
    area (str):         brain area

    Returns:
    params (dict):      model Args
    """

    params = {
        'sigma': 0.5,  # noise amplitude
        'tau_s': 0.5e-3,  # synaptic time constant
        'tau_m': 10e-3,  # membrane time constant
        'C_m': 250e-6,  # membrane capacitance
        'kappa': np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),  # adaptation strength
        'tau_a': 500.,  # adaptation time constant
        'a': 48,  # function gain
        'b': 981,  # function threshold
        'd': 0.0089,  # function noise factor
        'nu_bg': 8.0,  # background input
    }
    params['R'] = params['tau_m'] / params['C_m']

    # POPULATION SIZES (N)
    # N holds the population sizes for 16 populations (two dmf columns)
    if area == 'V1':
        N = np.array([47386, 13366, 70387, 17597, 20740, 4554, 19839, 4063])  # V1
    if area == 'V2':
        N = np.array([50521, 14250, 36685, 9171, 19079, 4189, 19248, 3941])  # V2
    if area == 'V3':
        N = np.array([58475, 16494, 47428, 11857, 12056, 2647, 14529, 2975])  # V3
    if area == 'V3A':
        N = np.array([40887, 11532, 23789, 5947, 12671, 2782, 15218, 3116])  # V3A
    if area == 'MT':
        N = np.array([60606, 17095, 28202, 7050, 14176, 3113, 15837, 3243])  # MT
    if area == 'MSTd':
        N = np.array([44343, 12507, 22524, 5631, 14742, 3237, 17704, 3625])  # MSTd
    if area == 'LIP':
        N = np.array([51983, 14662, 20095, 5024, 11630, 2554, 28115, 5757])  # LIP
    if area == 'FEF':
        N = np.array([44053, 12425, 23143, 5786, 16943, 3720, 16128, 3302])  # FEF
    if area == 'FST':
        N = np.array([36337, 10249, 12503, 3126, 12624, 2772, 15160, 3104])  # FST

    K_bg = np.tile([2510, 2510, 2510, 2510, 2510, 2510, 2510, 2510], 2)

    N = np.tile(N, 2)  # repeat N, so len(N) = 16
    N = N / 2  # divide area in two

    # SYNAPTIC STRENGTH (J)
    # J is a 16x16 matrix containing synaptic strength between each set of populations
    J_E = 87.8e-3
    params['J_E'] = J_E
    k = 0
    g = np.zeros(4)
    for n in range(4):
        g[n] = - (N[k] / N[k + 1])  # relative inhibitory synaptic strength
        k += 2
    J_column = np.array([J_E, J_E * g[0], J_E, J_E * g[1], J_E, J_E * g[2], J_E, J_E * g[3]])  # so each second population is inhibitory
    J = np.zeros((16, 16))
    J_column = np.tile(J_column.T, [8, 1])
    J[:8, :8] = J_column
    J[8:, 8:] = J_column
    J[0, 0] = J_local
    J[8, 8] = J_local
    J[1, 8] = J_lateral
    J[9, 0] = J_lateral

    # CONNECTION PROBABILITIES (P)
    # P is a 16x16 matrix containing connection probability between each set of populations
    # K is a 16x16 matrix containing the number of connections between each set of populations
    P_circuit = np.array(
        [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
         [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
         [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
         [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
         [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
         [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
         [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
         [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
    P_circuit[0, 2] *= 2
    P = np.zeros((16, 16))
    P[:8, :8] = P_circuit
    P[8:, 8:] = P_circuit
    P[1, 8] = 0.1  # lateral connections
    P[9, 0] = 0.1  # lateral connections
    K = np.log(1 - P) / np.log(1 - 1 / (N * N.T)) / N


    params['M'] = 16  # num populations
    params['N'] = N  # population sizes
    params['J'] = J  # synaptic strength
    params['P'] = P  # connection probabilities
    params['K'] = K  # number of connections
    params['W'] = J * K  # recurrent weight

    params['W_bg'] = K_bg * J_E  # background input weight

    params['kappa'] = np.tile([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2)
    params['tau_a'] = 10.
    params['sigma'] = 0.5

    return params


# specify cortical column (horizontal/vertical) AND layer
def set_stimulation(stim, column, layer, nu, params):
    """
    Args:
    stim (array):   external input
    column (str):   column name
    layer (str):    layer name
    nu (float):     input strength
    params (dict):  model Args

    Returns:
    stim (array):   external input
    """

    if column == 'H':
        idx = 0
    if column == 'V':
        idx = 8
    if layer == 'L23':
        idx += 0
    if layer == 'L4':
        idx += 2
    if layer == 'L5':
        idx += 4
    if layer == 'L6':
        idx += 6

    K_ext = np.array([300, 255])
    for i in range(2):
        W_ext = K_ext[i] * params['J_E']
        stim[idx + i] += W_ext * nu

    return stim


# specify cortical column (horizontal/vertical) and NOT layer
# layer = L4, i.e. only L4 receives external stimulus input
def set_vis(stim, column, nu, params):
    """
    Args:
    stim (array):   external input
    column (str):   column name
    nu (float):     input strength
    params (dict):  model Args

    Returns:
    stim (array):   external input
    """

    if column == 'H':
        idx = 0
    if column == 'V':
        idx = 8

    # L4
    P_ext = [0.0983, 0.0619]
    N_ext = 3000
    K_ext = np.array(P_ext) * N_ext
    W_ext = K_ext * params['J_E']  # W_ext: external weight
    # nu: external input, in paper v_i_ext
    stim[idx + 2] = W_ext[0] * nu  # excitatory in layer 4
    stim[idx + 3] = W_ext[1] * nu  # inhibitory in layer 4

    return stim


def update(state, params, stim, dt=1e-4):
    """
    Args:
    state (dict):   model state (I, A, H, R, N)
        state['I']: input current
        state['A']: adaptation
        state['H']: membrane potential
        state['R']: rate
        state['N']: noise

    params (dict):  model parameters
    stim (array):   external input; shape=(M)
    dt (float):     time step

    Returns:
    state (dict):   model state
    """

    # Current (I)
    state['I'] += dt * (-state['I'] / params['tau_s'])  # self inhibition
    state['I'] += dt * np.dot(params['W'], state['R'])  # recurrent input
    state['I'] += dt * params['W_bg'] * params['nu_bg']  # background input
    state['I'] += dt * stim  # external input
    # state['N'] += (- state['N'] / params['tau_s'] + params['sigma'] * np.sqrt(2 / params['tau_s']) * np.random.randn(
    #     params['M'])) * dt
    # state['I'] += state['N']  # noise

    # Adaptation (A/w) and membrane potential (H/h)
    state['A'] += dt * ((-state['A'] + state['R'] * params['kappa']) / params['tau_a'])  # adaptation
    state['H'] += dt * ((-state['H'] + params['R'] * state['I']) / params['tau_m'])  # membrane potential

    # Firing rate (R/v)
    X = np.float64((params['a'] * (state['H'] - state['A']) - params['b']))
    state['R'] = X / (1 - np.exp(-params['d'] * X))

    [0.0001 for i in state['R'] if i <= 0.]  # ensure positive values (min = 0.0   Hz)
    [500 for i in state['R'] if i > 500]  # limit firing rates (max = 500.0 Hz)

    return state


if __name__ == '__main__':

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

    for i in range(16):
        plt.plot(R[i,:])
    plt.show()
