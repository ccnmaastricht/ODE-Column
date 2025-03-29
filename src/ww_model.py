# -----------------------------------------------------------------------------
# Contributors: Mario Senden mario.senden@maastrichtuniversity.nl
# Modified by: Vaishnavi Narayanan vaishnavi.narayanan@maastrichtuniversity.nl
# -----------------------------------------------------------------------------

import numpy as np

'''DECISION MAKING (DM)
Wong Wang model of 2006

parameters
----------
gamma           : controls slow rise of NMDA channels (dimensionless)
tau_s           : time constant of fraction of open channels (seconds)
tau_ampa        : time constant of AMPA receptors / noise (seconds)
params          : contains all parameters necessary to set up the model.
  J_within      : within pool connectivity (nA)
  J_between     : between pool connectivity (nA)
  J_ext         : external drive (nA * Hz^-1)
  mu            : unbiased external input
  I_0           : unspecific background input (nA)
  sigma_noise   : standard deviation of unspecific background input (nA)
  dt            : integration time step (seconds)

integration
-----------
prop            : Exponential Euler (EE) propagator
 lin            : linear part of numerical integrator
 nonlin         : nonlinear part of numerical integrator

properties
----------
R               : instantaneous firing rate

functions
---------
f(x)            : (a * x - b) / (1 - exp(-d  * (a * x - b) ))
  parameters
  ----------
  a             : 270 (VnC)^-1
  b             : 108 Hz
  d             : 0.154 seconds
simulate(time)  : simulate model for "time" seconds
update          : perform numerical integration for single time step
reset           : reset the model
phase_plane     : perform phase plane analysis given mu and coherence
set_coherence(x): set coherence to value x
set_mu(x)       : set mu to value x
'''


class DM:

    def __init__(self):
        self.gamma = 0.641
        self.tau_s = 0.1
        self.tau_ampa = 0.002
        self.params = {
            'gamma': 0.641,
            'tau_s': 0.100,
            'tau_ampa': 0.002,
            'J_within': 0.2609,
            'J_between': 0.0497,
            'J_ext': 5.2e-4,
            'I_0':  0.3255,
            'sigma_noise': 0.0,
            'muA': 0.,
            'muB': 0.,
            'dt': 1e-3}
        self.prop = {
            's_lin': -self.params['dt'] / self.tau_s,
            's_nonlin': -np.expm1(-self.params['dt'] / self.tau_s),
            'noise_lin': -self.params['dt'] / self.tau_ampa,
            'noise_rand': np.sqrt(-0.5 * np.expm1(-2. * self.params['dt'] /
                                                  self.tau_ampa))}
        self.coherence = 0.
        self.muA = self.params['muA']
        self.muB = self.params['muB']
        self.s = np.ones(2) * 0.1
        self.x = np.zeros(2)
        self.r = np.zeros(2)
        self.W = np.array([[self.params['J_within'], -self.params['J_between']],
                           [-self.params['J_between'], self.params['J_within']]])
        self.I_noise = np.random.randn(2) * self.params['sigma_noise']
        self.dsig = np.sqrt(self.params['dt'] / self.tau_ampa) *\
                    self.params['sigma_noise']

    def f(self, x):
        return (270. * x - 108) / (1. - np.exp(-0.154 * (270. * x - 108.)))

    def update(self):
        I_ext = np.array([self.params['J_ext'] * self.muA,
                          self.params['J_ext'] * self.muB])

        I_rec = np.dot(self.W, self.s)
        self.I_noise += self.params['dt'] * (self.params['I_0'] - self.I_noise) /\
                        self.tau_ampa + self.dsig * np.random.randn(2)
        self.x = I_rec + I_ext + self.I_noise
        self.r = self.f(self.x)
        self.s += self.params['dt'] * (-self.s / self.tau_s + (1. - self.s) *
                             self.gamma * self.r)

    def simulate(self,time):
        t_steps = int(time / self.params['dt']) + 1
        R = np.zeros((2, t_steps))
        for t in range(t_steps):
            self.update()
            R[:, t] = self.r
        return R

    def run_sim(self, muA, muB):
        # Pre stimulus
        self.set_mu(0., 0, )
        R = self.simulate(5.)

        # Stimulus phase
        self.set_mu(muA, muB)
        R = np.append(R, self.simulate(5.), axis=1)

        # Post stimulus
        self.set_mu(0., 0, )
        R = np.append(R, self.simulate(5.), axis=1)

        self.reset()
        return R

    def set_coherence(self, x):
        self.coherence = x

    def set_mu(self, a, b):
        self.muA = a
        self.muB = b

    def reset(self):
        self.coherence = 0.
        self.muA = self.params['muA']
        self.muB = self.params['muB']
        self.s = np.ones(2) * 0.1
        self.x = np.zeros(2)
        self.r = np.zeros(2)
        self.W = np.array([[self.params['J_within'], -self.params['J_between']],
                           [-self.params['J_between'], self.params['J_within']]])
