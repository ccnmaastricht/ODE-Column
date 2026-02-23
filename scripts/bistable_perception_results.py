import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from pprint import pprint

from sympy.printing.pretty.pretty_symbology import line_width
from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from src.utils import *
from src.column_network_wta import ColumnAreaWTA
from wta_ode import set_stim_three_phases



# Kris' functions for alternation rate and dominance duration
def running_mean(x, N, outliers=False):
    """
    Computes average of last N timepoints and replaces outliers with 0.
    Args:
    x (array):          input
    N (int):            window size
    outliers (bool):    remove outliers
    """
    if outliers==False:
        mean = np.mean(x)
        for i in range(len(x)):
            if x[i] > mean*10:
                x[i] = 0
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def dominance_time(A1, A2, dt=1e-4, cutoff=.1, thresh=0.0001, sliding_window=10000):
    """
    Args:
    A1 (array):         activity of column 1; shape=(num_populations, num_time_steps)
    A2 (array):         activity of column 2; shape=(num_populations, num_time_steps)
    dt (float):         time step
    cutoff (float):     cutoff for dominance interval

    Returns:
    DT (array):         dominance intervals
    """
    # get switching points
    A1_smooth = running_mean(A1, N=sliding_window)
    A2_smooth = running_mean(A2, N=sliding_window)
    A_diff = A1_smooth - A2_smooth

    sign_diff = np.sign(A_diff)
    switch_inds = np.where(np.diff(sign_diff) != 0)[0]
    switch_times = switch_inds * dt

    DT_signed = []
    for i in range(len(switch_times) - 1):
        start = switch_inds[i]
        end = switch_inds[i + 1]
        dur = (end - start) * dt
        if dur >= cutoff:
            dominant = np.sign(np.mean(A_diff[start:end]))
            DT_signed.append(dominant * dur)

    if len(DT_signed) > 0:
        return np.array(DT_signed)

    # No switches or too short
    return np.array([np.sign(np.mean(A_diff)) * len(A1) * dt])

def alternation_rate(A1, A2, dt=1e-4, cutoff=.1, sliding_window=1000):
    """
    Args:
    A1 (array):         activity of column 1; shape=(num_populations, num_time_steps)
    A2 (array):         activity of column 2; shape=(num_populations, num_time_steps)
    dt (float):         time step
    cutoff (float):     cutoff for dominance interval

    Returns:
    AR (float):         alternation rate
    """
    A_diff = running_mean(A1, N=sliding_window) - running_mean(A2, N=sliding_window)
    AL = 0
    k = 0
    for t in range(len(A_diff)):
        if k == 0:
            current = np.sign(A_diff[t])
            k += 1
        else:
            if np.sign(A_diff[t]) != current and k*dt >= cutoff:
                k = 0
                AL += 1
            else:
                k += 1
    AR = (AL / (len(A_diff) * dt))
    return AR, AL

def run_bistable_perception(fn, nr_iterations):

    # Load network
    network = load_pkl_file(fn)
    inner_weights = network.recurrent_weights.detach()
    lat_in_weights = network.lat_in_weights.detach()
    full_weights = inner_weights + lat_in_weights

    self_excitation = [full_weights[0, 0], full_weights[8, 8]]
    lat_inhibition = [full_weights[1, 8], full_weights[9, 0]]

    # lat_in_weights[0, 0], lat_in_weights[8, 8] = 274.0, 274.0
    # lat_in_weights[1, 8], lat_in_weights[9, 0] = 1060.0, 1060.0
    #
    # network.lat_in_weights = torch.nn.Parameter(lat_in_weights, requires_grad=True)

    # Time params
    dt = 1e-4
    phase = 10  # secs
    time_steps = int(phase / dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)

    network.adaptation_strength = torch.tensor([1.5, 0., 0., 0., 0., 0., 0., 0., 1.5, 0., 0., 0., 0., 0., 0., 0., ])

    # Initial state is just zeros - except for the membrane potential
    initial_state = torch.zeros(48).unsqueeze(0)  # 48 = 8*2*3
    initial_state[0, :16] = torch.tensor(
        [-1.7997e-01, 8.3757e+00, 1.1346e+01, 1.1953e+01, -6.5426e+00, 1.0319e+01, -2.9719e+01, 1.2530e+01,
         -1.7997e-01, 8.3757e+00, 1.1346e+01, 1.1953e+01, -6.5426e+00, 1.0319e+01, -2.9719e+01, 1.2530e+01])

    inputs = [10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30.]

    output_dom = torch.zeros((len(inputs), len(inputs)))
    output_alt = torch.zeros((len(inputs), len(inputs)))

    with torch.no_grad():

        for muA in inputs:
            for muB in inputs:
                print(muA, muB)

                # Set stim and time_vec
                stim = torch.zeros(time_steps, 16)
                stim[:, [2, 3]] = muA
                stim[:, [10, 11]] = muB
                network.stim = stim
                network.time_vec = time_vec

                # loop for nr of iterations
                for i in range(nr_iterations):
                    ode_output = sdeint(network, initial_state, time_vec,
                                        names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')
                    comp_fr = compute_firing_rate(ode_output[:, 0, :16] - ode_output[:, 0, 16:32])
                    if i == 0:
                        total = comp_fr

                    else:
                        total = torch.concat([total, comp_fr], dim=0)
                    initial_state = ode_output[-1, :, :]

                #     # Plot results
                #     # m = ode_output[:, 0, :16]
                #     # plt.plot(m[:, 0])
                #     # plt.plot(m[:, 8])
                #     # plt.show()
                #     #
                #     # a = ode_output[:, 0, 16:32]
                #     # plt.plot(a[:, 0])
                #     # plt.plot(a[:, 8])
                #     # plt.show()
                #     #
                #     # plt.plot(comp_fr[:, 0])
                #     # plt.plot(comp_fr[:, 8])
                #     # plt.show()
                #
                # plt.plot(total[:, 0])
                # plt.plot(total[:, 8])
                # plt.show()

                # Dominance duration
                A1 = total[:, 0].detach().numpy()  # func expects np
                A2 = total[:, 8].detach().numpy()
                dom_time = dominance_time(A1, A2, dt=dt, thresh=0.0001, sliding_window=10000)

                # Alternation rate
                alt_rate, alt = alternation_rate(A1, A2, dt=dt, sliding_window=1000)

                output_dom[int(muA-10), int(muB-10)] = torch.tensor(np.round(np.sum(dom_time), 2))
                output_alt[int(muA-10), int(muB-10)] = torch.tensor(np.round(alt_rate, 2))
                pprint(output_dom)
                pprint(output_alt)
    return output_dom, output_alt



def plot_dom_alt(dominance, alternation):

    # small_dom = dominance.reshape(7, 3, 7, 3).mean(axis=(1, 3))
    # small_alt = alternation.reshape(7, 3, 7, 3).mean(axis=(1, 3))

    plt.figure(figsize=(3, 2.5))
    heatmap = plt.imshow(np.rot90(dominance), cmap="RdYlBu", interpolation="nearest", vmin=-50.0, vmax=50.0, extent=[10, 30, 10, 30])
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Dominance duration (s)', fontsize=14)
    plt.xticks([10, 20, 30])
    plt.xlabel('Input column A (Hz)', fontsize=14)
    plt.yticks([10, 20, 30])
    plt.ylabel('Input column B (Hz)', fontsize=14)
    # plt.text(0.95, 0.5, 'Dominance duration (s)', va='center', rotation='vertical', fontsize=14)
    plt.show()

    # heatmap = plt.imshow(np.rot90(small_dom), cmap="RdYlBu", interpolation="nearest", vmin=-50.0, vmax=50.0, extent=[10, 30, 10, 30])
    # plt.colorbar(heatmap)
    # plt.show()
    #
    # heatmap = plt.imshow(np.rot90(alternation), cmap="Reds", interpolation="nearest", extent=[10, 30, 10, 30])
    # plt.colorbar(heatmap)
    # plt.show()
    #
    # heatmap = plt.imshow(np.rot90(small_alt), cmap="Reds", interpolation="nearest", vmin=0.0, extent=[10, 30, 10, 30])
    # plt.colorbar(heatmap)
    # plt.show()



if __name__ == '__main__':
    # fn = '../wta_trained_model.pkl'
    # fn = '../trained_wta_models/wta_full_pops_full.pkl'

    # dom_dur, alt_rate = run_bistable_perception(fn, nr_iterations=5)  # nr_iters * 10 = total seconds
    # save_pkl_file('../dominance_duration.pkl', dom_dur)
    # save_pkl_file('../alternation_rate.pkl', alt_rate)
    dom_dur = load_pkl_file('../1_dominance_duration.pkl')
    alt_rate = load_pkl_file('../1_alternation_rate.pkl')
    plot_dom_alt(dom_dur, alt_rate)

