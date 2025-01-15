import numpy as np
import warnings

warnings.filterwarnings("ignore")


class SingleColumnDMF:

    def __init__(self, J_local=87.8e-3, area='MT'):
        self.J_local = J_local  # local synapse strength
        self.area = area  # brain area

        # Get parameters
        self.params = self.get_params(self.J_local, self.area)

        # Intialize the starting state (all zeros?)
        self.state = {}
        M = self.params['M']  # number of populations (=8)
        self.state['I'] = np.zeros(M)  # input current
        self.state['A'] = np.zeros(M)  # adaptation
        self.state['H'] = np.zeros(M)  # membrane potential
        self.state['R'] = np.zeros(M)  # rate
        self.state['N'] = np.zeros(M)  # noise

        self.T = 1000  # number of timesteps; can be changed from default (=1000) when simulating

    def get_params(self, J_local, area):
        """
        Args:
        J_local (float):    local synaptic strength
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
        # N holds the population sizes for 8 populations (two dmf columns
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

        K_bg = np.array([2510, 2510, 2510, 2510, 2510, 2510, 2510, 2510])

        # SYNAPTIC STRENGTH (J)
        # J is a 8x8 matrix containing synaptic strength between each set of populations
        J_E = 87.8e-3
        params['J_E'] = J_E
        k = 0
        g = np.zeros(4)
        for n in range(4):
            g[n] = - (N[k] / N[k + 1])  # relative inhibitory synaptic strength
            k += 2
        J_column = np.array([J_E, J_E * g[0], J_E, J_E * g[1], J_E, J_E * g[2], J_E,
                             J_E * g[3]])  # so each second population is inhibitory
        J = np.tile(J_column.T, [8, 1])  # is J_column
        J[0, 0] = J_local

        # CONNECTION PROBABILITIES (P)
        # P is a 8x8 matrix containing connection probability between each set of populations
        # K is a 8x8 matrix containing the number of connections between each set of populations
        P = np.array(
            [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
             [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
             [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
             [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
             [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
             [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
             [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])  # is P_circuit
        P[0, 2] *= 2  # why??
        K = np.log(1 - P) / np.log(1 - 1 / (N * N.T)) / N

        params['M'] = 8  # num populations
        params['N'] = N  # population sizes
        params['J'] = J  # synaptic strength
        params['P'] = P  # connection probabilities
        params['K'] = K  # number of connections
        params['W'] = J * K  # recurrent weight

        params['W_bg'] = K_bg * J_E  # background input weight

        params['kappa'] = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        params['tau_a'] = 10.
        params['sigma'] = 0.5

        return params

    def update_state(self, stim, dt):
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
        self.state['I'] += dt * (-self.state['I'] / self.params['tau_s'])  # self inhibition
        self.state['I'] += dt * np.dot(self.params['W'], self.state['R'])  # recurrent input
        self.state['I'] += dt * self.params['W_bg'] * self.params['nu_bg']  # background input
        self.state['I'] += dt * stim  # external input
        self.state['N'] += (- self.state['N'] / self.params['tau_s'] + self.params['sigma'] * np.sqrt(
            2 / self.params['tau_s']) * np.random.randn(
            self.params['M'])) * dt
        self.state['I'] += self.state['N']  # noise

        # Adaptation (A/w) and membrane potential (H/h)
        self.state['A'] += dt * ((-self.state['A'] + self.state['R'] * self.params['kappa']) / self.params['tau_a'])  # adaptation
        self.state['H'] += dt * ((-self.state['H'] + self.params['R'] * self.state['I']) / self.params['tau_m'])  # membrane potential

        # Firing rate (R/v)
        X = np.float64((self.params['a'] * (self.state['H'] - self.state['A']) - self.params['b']))
        self.state['R'] = X / (1 - np.exp(-self.params['d'] * X))

        [0.0001 for i in self.state['R'] if i <= 0.]  # ensure positive values (min = 0.0   Hz)
        [500 for i in self.state['R'] if i > 500]  # limit firing rates (max = 500.0 Hz)

        return self.state

    def simulate(self, stim, dt=1e-4, T=1000, state_var='R'):
        '''
        Args:
        stim (2D array):        external input; shape=(T,M)
        dt (float):             time step
        T (int):                total number of time steps

        Returns:
        sim_output (2D array):  simulation output; shape=(M,T)
        '''
        # Array for saving requested simulation output (i.e. I. A. H. R. N.)
        sim_output = np.zeros((self.params['M'], T))

        # Run simulation
        for t in range(T):
            self.state = self.update_state(stim[:,t], dt)
            sim_output[:, t] = self.state[state_var]

        return sim_output

# specify which layer receives input
    def set_stim_bg(self, stim, layer, nu, params):
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

        stim_col = np.zeros((self.params['M'],1))

        if layer == 'L23':
            idx = 0
        if layer == 'L4':
            idx = 2
        if layer == 'L5':
            idx = 4
        if layer == 'L6':
            idx = 6

        K_ext = np.array([300, 255])
        for i in range(2):
            W_ext = K_ext[i] * params['J_E']
            stim_col[idx + i] += W_ext * nu

        # TODO: change this 1000 to self.T
        stim += np.tile(stim_col, 1000)

        return stim

    # layer = L4, i.e. only L4 receives external stimulus input
    def set_stim_ext(self, stim, nu, params):
        """
        Args:
        stim (array):   external input
        column (str):   column name
        nu (float):     input strength
        params (dict):  model Args

        Returns:
        stim (array):   external input
        """

        stim_col = np.zeros((self.params['M'], 1))

        P_ext = [0.0983, 0.0619]
        N_ext = 3000
        K_ext = np.array(P_ext) * N_ext
        W_ext = K_ext * params['J_E']  # W_ext: external weight
        # nu: external input, in paper v_i_ext
        stim_col[2] = W_ext[0] * nu  # excitatory in layer 4
        stim_col[3] = W_ext[1] * nu  # inhibitory in layer 4

        # TODO: change this 1000 to self.T
        stim += np.tile(stim_col, 1000)

        return stim
