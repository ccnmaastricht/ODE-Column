import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import time

import torchsde
from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from src.utils import *
from src.coupled_columns import ColumnAreaWTA



def blend_with_white(c, factor=0.6):
    """Blend color c with white by a factor."""
    return (1 - factor) * np.array([1, 1, 1]) + factor * np.array(c)


def coherence_results_ccn(fn):

    # Load network
    with open(fn, 'rb') as f:
        network = pickle.load(f)

    col_params = load_config('../config/model.toml')
    network = ColumnAreaWTA(col_params, area='mt')

    orig_weights = torch.tensor([[ 4.1900e-01, -4.9223e-01,  1.1323e-01, -1.0566e-01,  2.0433e-02,
          0.0000e+00,  5.3040e-03,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 3.8463e-01, -3.9232e-01,  3.9754e-02, -6.5461e-02,  4.8854e-02,
          0.0000e+00,  2.9262e-03,  0.0000e+00,  5.4915e-01,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 2.0566e-02, -1.5744e-02,  6.3114e-02, -1.7955e-01,  4.1836e-03,
         -1.8672e-04,  3.2230e-02,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 1.9051e-01, -7.7270e-03,  1.0242e-01, -2.1542e-01,  2.0571e-03,
          0.0000e+00,  7.7669e-02,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 2.8151e-01, -1.7086e-01,  6.4156e-02, -7.0772e-03,  5.3991e-02,
         -2.9011e-01,  1.4330e-02,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 1.4995e-01, -7.2550e-02,  3.2234e-02, -2.7268e-03,  3.8507e-02,
         -2.3618e-01,  6.0050e-03,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 4.1833e-02, -1.7618e-02,  2.6403e-02, -2.0724e-02,  3.6656e-02,
         -1.2382e-02,  2.8092e-02, -1.7739e-01, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 9.8653e-02, -2.6619e-03,  4.2166e-03, -6.1922e-04,  1.7482e-02,
         -4.9986e-03,  4.7322e-02, -1.0834e-01, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  4.1900e-01, -4.9223e-01,
          1.1323e-01, -1.0566e-01,  2.0433e-02,  0.0000e+00,  5.3040e-03,
          0.0000e+00],
        [ 5.4915e-01,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  3.8463e-01, -3.9232e-01,
          3.9754e-02, -6.5461e-02,  4.8854e-02,  0.0000e+00,  2.9262e-03,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  2.0566e-02, -1.5744e-02,
          6.3114e-02, -1.7955e-01,  4.1836e-03, -1.8672e-04,  3.2230e-02,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  1.9051e-01, -7.7270e-03,
          1.0242e-01, -2.1542e-01,  2.0571e-03,  0.0000e+00,  7.7669e-02,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  2.8151e-01, -1.7086e-01,
          6.4156e-02, -7.0772e-03,  5.3991e-02, -2.9011e-01,  1.4330e-02,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  1.4995e-01, -7.2550e-02,
          3.2234e-02, -2.7268e-03,  3.8507e-02, -2.3618e-01,  6.0050e-03,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  4.1833e-02, -1.7618e-02,
          2.6403e-02, -2.0724e-02,  3.6656e-02, -1.2382e-02,  2.8092e-02,
         -1.7739e-01],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  9.8653e-02, -2.6619e-03,
          4.2166e-03, -6.1922e-04,  1.7482e-02, -4.9986e-03,  4.7322e-02,
         -1.0834e-01]])

    network.recurrent_weights = orig_weights

    # Time params
    dt = 1e-4
    stim_phase = 0.05
    time_steps = int((stim_phase * 3) / dt)  # add pre- and post-stimulus phase
    time_vec = torch.linspace(0., time_steps * dt, time_steps)

    # Initial state is just zeros
    initial_state = torch.zeros(48).unsqueeze(0)  # 48 = 8*2*3

    with torch.no_grad():

        # Run the model for different coherences (diff between input A and B)
        coherences = [0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]

        fr_results = torch.Tensor(4, len(coherences), 600, 2)

        for i, coherence in enumerate(coherences):

            # Determine the stimulus
            muA = 20.
            muB = muA - coherence
            stim = set_stim_three_phases(network.num_populations, time_vec, torch.tensor([muA, muB]))

            # Set stim and time_vec
            network.stim = stim
            network.time_vec = time_vec

            # Run the model
            ode_output = odeint(network, initial_state, time_vec)
            # ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'})

            # Compute firing rates
            fr = ode_output[:, 0, 32:]

            fr_results[0, i, :, :] = fr[400:1000, [0, 8]]  # layer 2/3
            fr_results[1, i, :, :] = fr[400:1000, [2, 10]]  # layer 4
            fr_results[2, i, :, :] = fr[400:1000, [4, 12]]  # layer 5
            fr_results[3, i, :, :] = fr[400:1000, [6, 14]]  # layer 6



        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(7, 4))
        plt.subplots_adjust(hspace=0.5, wspace=0.2)

        colors = plt.get_cmap('rainbow', len(coherences))
        layers = ['Layer 2/3', 'Layer 4', 'Layer 5', 'Layer 6']

        # Normalize for color mapping for the colorbar
        norm = mcolors.Normalize(vmin=min(coherences), vmax=max(coherences))
        sm = cm.ScalarMappable(cmap=colors, norm=norm)
        sm.set_array([])

        # Font
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10

        for l_idx in range(4):

            if l_idx == 0:
                axes_ = axes[0, 0]
            elif l_idx == 1:
                axes_ = axes[1, 0]
            elif l_idx == 2:
                axes_ = axes[0, 1]
            elif l_idx == 3:
                axes_ = axes[1, 1]

            for c_idx in range(len(coherences)):
                color = colors(c_idx)

                axes_.plot(fr_results[l_idx, c_idx, :, 1], linestyle='--', color=color, zorder=1)
                axes_.plot(fr_results[l_idx, c_idx, :, 0], color=color, zorder=2)

            # Add y label indicating layer
            axes_.set_title(layers[l_idx])

            # Add vertical line at x=0 (stimulus)
            axes_.axvline(x=100, color='gray', linestyle='--', linewidth=0.8)

            # Remove the upper and right-most borders
            axes_.spines['top'].set_visible(False)
            axes_.spines['right'].set_visible(False)

            # Remove y-ticks
            axes_.set_yticks([])

            # Set x-ticks
            xticks = np.arange(0, 601, 100)
            if l_idx == 0 or l_idx == 2:  # Remove x-ticks for top subplots
                axes_.set_xticklabels([])
            else:  # For bottom subplots: relabel x-axis from 0–600 to -100–500
                axes_.tick_params(labelsize=8)
                xlabels = xticks - 100  # Shift labels
                # xlabels = xlabels / /10
                axes_.set_xticklabels(xlabels)
            axes_.set_xticks(xticks)
            axes_.set_xlim(0, 600)

        fig.text(0.08, 0.5, 'Firing rates', va='center', rotation='vertical', fontsize=14)

        # Color bar
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', location='top', fraction=0.04, pad=0.1)
        cbar.set_label('Relative evidence in Hz')
        cbar.set_ticks([min(coherences), max(coherences)])
        cbar.set_ticklabels([f'{min(coherences):.2f}', f'{max(coherences):.2f}'])

        plt.show()


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

    # print(switch_times)
    #
    # plt.plot(A1_smooth)
    # plt.plot(A2_smooth)
    # plt.show()

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


def bistable_perception(fn, nr_iterations):

    # Load network
    with open(fn, 'rb') as f:
        network = pickle.load(f)
    weights = network.recurrent_weights

    col_params = load_config('../config/model.toml')
    network = ColumnAreaWTA(col_params, area='mt')

    orig_weights = torch.tensor([[ 4.1900e-01, -4.9223e-01,  1.1323e-01, -1.0566e-01,  2.0433e-02,
          0.0000e+00,  5.3040e-03,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 3.8463e-01, -3.9232e-01,  3.9754e-02, -6.5461e-02,  4.8854e-02,
          0.0000e+00,  2.9262e-03,  0.0000e+00,  5.4915e-01,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 2.0566e-02, -1.5744e-02,  6.3114e-02, -1.7955e-01,  4.1836e-03,
         -1.8672e-04,  3.2230e-02,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 1.9051e-01, -7.7270e-03,  1.0242e-01, -2.1542e-01,  2.0571e-03,
          0.0000e+00,  7.7669e-02,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 2.8151e-01, -1.7086e-01,  6.4156e-02, -7.0772e-03,  5.3991e-02,
         -2.9011e-01,  1.4330e-02,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 1.4995e-01, -7.2550e-02,  3.2234e-02, -2.7268e-03,  3.8507e-02,
         -2.3618e-01,  6.0050e-03,  0.0000e+00, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 4.1833e-02, -1.7618e-02,  2.6403e-02, -2.0724e-02,  3.6656e-02,
         -1.2382e-02,  2.8092e-02, -1.7739e-01, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [ 9.8653e-02, -2.6619e-03,  4.2166e-03, -6.1922e-04,  1.7482e-02,
         -4.9986e-03,  4.7322e-02, -1.0834e-01, -0.0000e+00,  0.0000e+00,
         -0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  4.1900e-01, -4.9223e-01,
          1.1323e-01, -1.0566e-01,  2.0433e-02,  0.0000e+00,  5.3040e-03,
          0.0000e+00],
        [ 5.4915e-01,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  3.8463e-01, -3.9232e-01,
          3.9754e-02, -6.5461e-02,  4.8854e-02,  0.0000e+00,  2.9262e-03,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  2.0566e-02, -1.5744e-02,
          6.3114e-02, -1.7955e-01,  4.1836e-03, -1.8672e-04,  3.2230e-02,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  1.9051e-01, -7.7270e-03,
          1.0242e-01, -2.1542e-01,  2.0571e-03,  0.0000e+00,  7.7669e-02,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  2.8151e-01, -1.7086e-01,
          6.4156e-02, -7.0772e-03,  5.3991e-02, -2.9011e-01,  1.4330e-02,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  1.4995e-01, -7.2550e-02,
          3.2234e-02, -2.7268e-03,  3.8507e-02, -2.3618e-01,  6.0050e-03,
          0.0000e+00],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  4.1833e-02, -1.7618e-02,
          2.6403e-02, -2.0724e-02,  3.6656e-02, -1.2382e-02,  2.8092e-02,
         -1.7739e-01],
        [-0.0000e+00,  0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00,
          0.0000e+00, -0.0000e+00,  0.0000e+00,  9.8653e-02, -2.6619e-03,
          4.2166e-03, -6.1922e-04,  1.7482e-02, -4.9986e-03,  4.7322e-02,
         -1.0834e-01]])

    network.recurrent_weights = weights

    # Time params
    dt = 1e-4
    phase = 10  # secs
    time_steps = int(phase/dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)

    # Initial state is just zeros
    initial_state = torch.zeros(48).unsqueeze(0)  # 48 = 8*2*3

    curr_time = time.time()

    with torch.no_grad():

        for muA in [20.]: # [13., 14., 15., 16., 17., 18., 19., 20.]:
            for muB in [20.]: # [10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]:
                print(muA, muB)

                # Set stim and time_vec
                stim = torch.zeros(time_steps, 16)
                stim[:, [2, 3]] = muA
                stim[:, [10, 11]] = muB
                network.stim = stim
                network.time_vec = time_vec

                # Run the model
                # ode_output = odeint(network, initial_state, time_vec)
                # bm = torchsde.BrownianInterval(
                #     t0=time_vec[0].item(),
                #     t1=time_vec[-1].item(),
                #     size=initial_state.size(),
                #     levy_area_approximation="space-time",  # Required for SRK
                # )
                # bm = torchsde.BrownianTree(
                #     t0=time_vec[0].item(),
                #     t1=time_vec[-1].item(),
                #     w0=torch.zeros_like(initial_state))
                # ode_output = sdeint(network, initial_state, time_vec, bm=bm, names={'drift': 'forward', 'diffusion': 'diffusion'}, method='euler')

                # loop for nr of iterations
                for i in range(nr_iterations):
                    ode_output = sdeint(network, initial_state, time_vec,
                                        names={'drift': 'forward', 'diffusion': 'diffusion'})
                    comp_fr = compute_firing_rate_torch(ode_output[:, 0, :16] - ode_output[:, 0, 16:32])
                    if i == 0:
                        total = comp_fr
                    else:
                        total = torch.concat([total, comp_fr], dim=0)
                    initial_state = ode_output[-1, :, :]

                    # print(time.time() - curr_time)

                    # Plot results
                    m = ode_output[:, 0, :16]
                    plt.plot(m[:, 0])
                    plt.plot(m[:, 8])
                    plt.show()

                    a = ode_output[:, 0, 16:32]
                    plt.plot(a[:, 0])
                    plt.plot(a[:, 8])
                    plt.show()

                    fr = ode_output[:, 0, 32:]
                    plt.plot(fr[:, 0])
                    plt.plot(fr[:, 8])
                    plt.plot(comp_fr[:, 0])
                    plt.plot(comp_fr[:, 8])
                    plt.show()

                # Dominance duration
                A1 = total[:, 0].detach().numpy()  # func expects np
                A2 = total[:, 8].detach().numpy()
                dom_time = dominance_time(A1, A2, dt=dt, thresh=0.0001, sliding_window=10000)
                pprint(dom_time)
                print(np.round(np.sum(dom_time), 2))

                # Alternation rate
                alt_rate, alt = alternation_rate(A1, A2, dt=dt, sliding_window=1000)
                print(np.round(alt_rate, 2))

                # Histogram
                plt.hist(abs(dom_time), bins=100, color='r')
                plt.show()


def plot_dom_alt():

    dominance =    [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 104.8],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-185.74, 48.67, -119.64, 0., 0., 0., 0., 0., 0., 0., 0.],
                    [68.12, -226.35, -67.54, -73.43, -43.45, -36.92, -36.52, 69.51, -274.94, -56.68, -255.72],
                    [104.4, 77.93, -5.95, 65.34, -118.87, -185.14, -43.29, 64.29, -137.95, -297.42, -251.6],
                    [0.2, -23.68, 13.57, -271.09, -439.51, -341.42, -249.26, -161.72, -313.98, -313.5, -139.34]]

    alternation =  [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.26],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0.35, 0.38, 0.48, 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0.41, 0.32, 0.35, 0.29, 0.27, 0.27, 0.32, 0.34, 0.24, 0.26, 0.31],
                    [0.38, 0.41, 0.46, 0.35, 0.28, 0.3, 0.27, 0.25, 0.25, 0.36, 0.31],
                    [0.36, 0.45, 0.3, 0.3, 0.36, 0.37, 0.31, 0.23, 0.24, 0.33, 0.49]]

    heatmap = plt.imshow(dominance, cmap="viridis", interpolation="nearest", extent=[10, 20, 10, 20])
    plt.colorbar(heatmap)
    plt.show()

    heatmap = plt.imshow(alternation, cmap="viridis", interpolation="nearest", extent=[10, 20, 10, 20])
    plt.colorbar(heatmap)
    plt.show()


def wta_timecourse(fn):

    # Load network
    with open(fn, 'rb') as f:
        _network = pickle.load(f)
    weights = _network.recurrent_weights

    col_params = load_config('../config/model.toml')
    network = ColumnAreaWTA(col_params, area='mt')

    network.recurrent_weights = weights

    # Time params
    dt = 1e-4
    phase = 0.5  # secs
    time_steps = int(phase/dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)
    network.time_vec = time_vec

    # Initial state is just zeros
    initial_state = torch.zeros(48).unsqueeze(0)  # 48 = 8*2*3
    initial_state[:, :16] = torch.tile(torch.tensor([-1.5554, 8.9735, 12.0712, 12.5040, -5.2554, 10.4650, -30.8225, 12.6189]), (2,))

    with torch.no_grad():
        i = 0

        for stims in [[0., 0.], [0., 0.], [0., 0.], [10., 30.], [0., 0.], [30., 10.], [0., 0.], [20., 20.], [20., 20.], [20., 20.], [20., 20.], [0., 0.]]:  #
            muA = stims[0]
            muB = stims[1]

            # Set stim and time_vec
            stim = torch.zeros(time_steps, 16)
            stim[:, [2, 3]] = muA
            stim[:, [10, 11]] = muB
            network.stim = stim
            network.time_vec = time_vec

            ode_output = sdeint(network, initial_state, time_vec,
                                names={'drift': 'forward', 'diffusion': 'diffusion'})
            comp_fr = compute_firing_rate_torch(ode_output[:, 0, :16] - ode_output[:, 0, 16:32])
            # comp_fr = ode_output[:, 0, 32:]
            if i == 0:
                time_course = comp_fr
                stim_time_course = stim
            else:
                time_course = torch.concat([time_course, comp_fr], dim=0)
                stim_time_course = torch.concat([stim_time_course, stim], dim=0)
            initial_state = ode_output[-1, :, :]
            i += 1

        with open('../wta_timecourse_plot.pkl', 'wb') as f:
            pickle.dump(time_course, f)

        with open("../wta_timecourse_plot_28-07.pkl", 'rb') as f:
            time_course = pickle.load(f)

        time_course = time_course[time_steps:]
        stim_time_course = stim_time_course[time_steps:]

        time = np.arange(time_course.shape[0]) * dt

        # Set the figure size (wide, not too tall)
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                                 gridspec_kw={'height_ratios': [2.5, 1.0]})

        ax1.plot(time, time_course[:, 0], label='Column A', color='royalblue', linewidth=2)
        ax1.plot(time, time_course[:, 8], label='Column B', color='darkorange', linewidth=2)
        ax1.set_title('L2/3e firing rates in columns A & B', fontsize=14)
        ax1.set_ylabel('Firing Rate', fontsize=12)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax3.plot(time, stim_time_course[:, 2], label='Input 1', color='royalblue', linewidth=5)
        ax3.plot(time, stim_time_course[:, 10], label='Input 2', color='darkorange', linewidth=5, linestyle='--')
        ax3.set_title('Inputs', fontsize=14)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3.set_ylabel('Hz', fontsize=12)
        ax3.set_ylim(-5.0, 40.0)
        ax1.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)

        # Layout adjustment
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    fn = '../ww_trained_model_7.pkl'

    coherence_results_ccn(fn)

    # bistable_perception(fn, nr_iterations=100)  # nr_iters * 10 = total seconds
    # plot_dom_alt()

    # wta_timecourse(fn)
