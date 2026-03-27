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
from wta_ode import set_stim_whole_column



def plot_wta_layers(fr_results, coherences):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
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

            axes_.plot(fr_results[l_idx, :, c_idx, 1], linestyle='--', color=color, zorder=1)
            axes_.plot(fr_results[l_idx, :, c_idx, 0], color=color, zorder=2)

        # Add y label indicating layer
        axes_.set_title(layers[l_idx])

        # Add vertical line at x=0 (stimulus)
        axes_.axvline(x=100, color='gray', linestyle='--', linewidth=0.8)

        # Remove the upper and right-most borders
        axes_.spines['top'].set_visible(False)
        axes_.spines['right'].set_visible(False)

        # set y-ticks for each layer separately
        # if l_idx == 0:
        #     axes_.set_yticks([0, 1])
        # if l_idx == 1:
        #     axes_.set_yticks([6, 10, 14])
        # if l_idx == 2:
        #     axes_.set_yticks([0, 0.1, 0.2])
        # if l_idx == 3:
        #     axes_.set_yticks([0, 1e-6, 2e-6], ['0', '1e-6', '2e-6'])

        # Remove y-ticks
        # axes_.set_yticks([])

        # Set x-ticks
        xticks = np.arange(0, 601, 100)
        if l_idx == 0 or l_idx == 2:  # Remove x-ticks for top subplots
            axes_.set_xticklabels([])
        else:  # For bottom subplots: relabel x-axis from 0–600 to -100–500
            axes_.tick_params(labelsize=8)
            xlabels = xticks - 100  # Shift labels
            xlabels = xlabels // 10
            axes_.set_xticklabels(xlabels)
        axes_.set_xticks(xticks)
        axes_.set_xlim(0, 600)

    fig.text(0.08, 0.5, 'Firing rates (Hz)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.45, 0.03, 'Time (ms)', va='center', fontsize=14)

    # Color bar
    cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', location='top', fraction=0.04, pad=0.1)
    cbar.set_label('Relative evidence in Hz')
    cbar.set_ticks([min(coherences), max(coherences)])
    cbar.set_ticklabels([f'{min(coherences):.2f}', f'{max(coherences):.2f}'])

    plt.show()



def compute_WSI_DivT(firing_rates, dt, divt_threshold=0.1, divt_window=0.02):
    divt_window = int(divt_window / dt)  # convert from seconds to timesteps

    wsi_results = torch.zeros((firing_rates.shape[0], firing_rates.shape[2]))
    divt_results = torch.zeros((firing_rates.shape[0], firing_rates.shape[2]))

    for i_layer in range(firing_rates.shape[0]):
        for i_coh in range(0, firing_rates.shape[2]):
            winner_column = firing_rates[i_layer, :, i_coh, 0]
            loser_column = firing_rates[i_layer, :, i_coh, 1]

            # Compute WSI (winner-selective index)
            wsi = (winner_column - loser_column) / (winner_column + loser_column)
            wsi_mean = torch.mean(wsi)
            wsi_results[i_layer, i_coh] = wsi_mean

            # Compute divergence time
            max_diff = torch.max(torch.abs(firing_rates[i_layer, :, :, 0] - firing_rates[i_layer, :, :, 1]))  # max difference of all coherence levels
            abs_difference = torch.abs(winner_column - loser_column)
            for t_i in range(len(abs_difference) - divt_window):
                fr_window = abs_difference[t_i:t_i+divt_window]
                if torch.all(fr_window > max_diff * divt_threshold):
                    divt_results[i_layer, i_coh] = (t_i - 100) * dt
                    break
    plot_WSI_DivT(wsi_results, divt_results, firing_rates)

def plot_WSI_DivT(wsi_results, divt_results, firing_rates):

    # Plotting WSI
    mean_wsi = torch.mean(wsi_results, dim=1)
    sd_wsi = torch.std(wsi_results, dim=1)
    plt.plot(mean_wsi, color='black', marker='o', linewidth=2)
    plt.errorbar(np.arange(4), mean_wsi, yerr=sd_wsi, ecolor='black', elinewidth=2, capthick=2, capsize=7, barsabove=True)
    # Plotting per coherence level (rainbow)
    colors = plt.get_cmap('rainbow', firing_rates.shape[2])
    for i_coh in range(0, firing_rates.shape[2]):
        color = colors(i_coh)
        plt.plot(wsi_results[:, i_coh], color=color, zorder=1)
    plt.grid(True, linestyle='--', alpha=1.0)
    plt.xticks(np.arange(4), ['L2/3', 'L4', 'L5', 'L6'], fontsize=14)
    plt.yticks([-0.6, 0.0, 0.6], fontsize=10)
    plt.xlim([-0.5, 3.5])
    plt.ylim([-0.8, 1.0])
    plt.ylabel('Winner selectivity index', fontsize=14)
    plt.show()

    # Plotting DivT
    mean_divt = torch.mean(divt_results, dim=1)
    sd_divt = torch.std(divt_results, dim=1)
    plt.plot(mean_divt, color='black', marker='o', linewidth=2)
    plt.errorbar(np.arange(4), mean_divt, yerr=sd_divt, ecolor='black', elinewidth=2, capthick=2, capsize=7, barsabove=True)
    # Plotting per coherence level (rainbow)
    colors = plt.get_cmap('rainbow', firing_rates.shape[2])
    for i_coh in range(0, firing_rates.shape[2]):
        color = colors(i_coh)
        plt.plot(divt_results[:, i_coh], color=color, zorder=1)
    plt.grid(True, linestyle='--', alpha=1.0)
    plt.xticks(np.arange(4), ['L2/3', 'L4', 'L5', 'L6'], fontsize=14)
    plt.yticks([0.0, 0.005, 0.01, 0.015], ['0', '5', '10', '15'], fontsize=10)
    plt.xlim([-0.5, 3.5])
    # plt.ylim([0.0, 0.016])
    plt.ylabel('Divergence timing (ms)', fontsize=14)
    plt.show()

def plot_WSI_DivT_layer_shifted(wsi_results, divt_results, firing_rates):

    population_sizes = np.array([60606, 28202, 14176, 15837])  # L6: 15837

    x_axis = []
    for i in range(len(population_sizes)):
        curr_pop_size = population_sizes[i]
        prev_pop_sizes = sum(population_sizes[:i])
        x_axis.append(prev_pop_sizes + (curr_pop_size // 2))
    x_axis = np.array(x_axis)

    # Plotting WSI
    mean_wsi = torch.mean(wsi_results, dim=1)
    sd_wsi = torch.std(wsi_results, dim=1)
    plt.plot(x_axis, mean_wsi, color='black', marker='o', linewidth=2)
    plt.errorbar(x_axis, mean_wsi, yerr=sd_wsi, ecolor='black', elinewidth=2, capthick=2, capsize=7, barsabove=True)
    # Plotting per coherence level (rainbow)
    colors = plt.get_cmap('rainbow', firing_rates.shape[2])
    for i_coh in range(0, firing_rates.shape[2]):
        color = colors(i_coh)
        plt.plot(x_axis, wsi_results[:, i_coh], color=color, zorder=1)
    plt.xlim(0, population_sizes.sum())
    plt.xticks(x_axis, ['L23', 'L4', 'L5', 'L6'])
    plt.ylabel('Choice selectivity index')
    plt.grid(True, linestyle='--', alpha=1.0)
    plt.show()

    # Plot DivT
    mean_divt = torch.mean(divt_results, dim=1)
    sd_divt = torch.std(divt_results, dim=1)
    plt.plot(x_axis, mean_divt, 'ro-', color='black', label='Layer values')
    plt.errorbar(x_axis, mean_divt, yerr=sd_divt, ecolor='black', elinewidth=2, capthick=2, capsize=7, barsabove=True)
    # Plotting per coherence level (rainbow)
    colors = plt.get_cmap('rainbow', firing_rates.shape[2])
    for i_coh in range(0, firing_rates.shape[2]):
        color = colors(i_coh)
        plt.plot(x_axis, divt_results[:, i_coh], color=color, zorder=1)
    plt.xlim(0, population_sizes.sum())
    plt.xticks(x_axis, ['L23', 'L4', 'L5', 'L6'])
    plt.ylabel('Divergence timing')
    plt.grid(True, linestyle='--', alpha=1.0)
    plt.show()

    interpolation_plot_wsi(abs(mean_wsi), mean_divt)




def wta_rainbow_plots(fn):

    # Load network
    network = load_pkl_file(fn)

    # Time params
    dt = 1e-4
    stim_phase = 0.05
    time_steps = int((stim_phase * 3) / dt)  # add pre- and post-stimulus phase
    time_vec = torch.linspace(0., time_steps * dt, time_steps)

    with torch.no_grad():

        # Run the model for different coherences (diff between input A and B)
        coherences = [2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]

        # Initial state is just zeros, expect for the membrane potential
        initial_state = torch.zeros(len(coherences), 32)
        initial_state[:, :16] = torch.tile(torch.tensor([-1.7997e-01, 8.3757e+00, 1.1346e+01, 1.1953e+01,
                                                         -6.5426e+00, 1.0319e+01, -2.9719e+01, 1.2530e+01]), (len(coherences), 2,))

        # Set the stimuli
        stims = torch.zeros((len(coherences), 2))
        for i, coherence in enumerate(coherences):
            muA = 40.
            muB = muA - coherence
            stims[i, :] = torch.tensor([muA, muB])
        stims = set_stim_whole_column(stims)

        # Set stim and time_vec
        network.set_stim(stims)
        network.set_time_vec(time_vec)

        # Run the model
        ode_output = odeint(network, initial_state, time_vec)
        fr = compute_firing_rate(ode_output[:, :, :16] - ode_output[:, :, 16:32])

        fr_results = torch.Tensor(4, 600, len(coherences), 2)
        fr_results[0, :, :, :] = fr[400:1000, :, [0, 8]]  # layer 2/3
        fr_results[1, :, :, :] = fr[400:1000, :, [2, 10]]  # layer 4
        fr_results[2, :, :, :] = fr[400:1000, :, [4, 12]]  # layer 5
        fr_results[3, :, :, :] = fr[400:1000, :, [6, 14]]  # layer 6

        plot_wta_layers(fr_results, coherences)
        compute_WSI_DivT(fr_results, dt)




def interpolation_plot_wsi(wsi, divt):

    population_sizes = np.array([60606, 28202, 14176, 15837])

    x_axis = []
    for i in range(len(population_sizes)):
        curr_pop_size = population_sizes[i]
        prev_pop_sizes = sum(population_sizes[:i])
        x_axis.append(prev_pop_sizes + (curr_pop_size // 2))
    x_axis = np.array(x_axis)

    # --------- CSI ---------
    # Fit polynomial
    coeffs = np.polyfit(x_axis, wsi, deg=3)
    poly = np.poly1d(coeffs)

    # Smooth x for plotting
    x_smooth = np.linspace(0, population_sizes.sum(), 500)
    y_smooth = poly(x_smooth)
    # y_smooth = np.clip(poly(x_smooth), None, 1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_axis, wsi, 'o', color='black', label='Layer values')
    ax.plot(x_smooth, y_smooth, '-', color='black', linewidth=2, label='Cubic fit')
    ax.set_xlim(0, population_sizes.sum())
    ax.set_xticks(x_axis, ['L2/3', 'L4', 'L5', 'L6'], fontsize=14)
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel('Choice Selectivity Index', fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=1.0)

    # x-axis grid lines
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    boundaries = np.concatenate([[0], np.cumsum(population_sizes)])
    ax_top.set_xticks(boundaries)
    ax_top.set_xticklabels([])
    ax_top.grid(True, axis='x', linestyle='--', alpha=1.0)
    ax_top.spines['top'].set_visible(False)
    ax_top.tick_params(top=False)
    plt.tight_layout()
    plt.show()

    # --------- DivT ---------
    # Fit polynomial
    coeffs = np.polyfit(x_axis, divt, deg=3)
    poly = np.poly1d(coeffs)

    # Smooth x for plotting
    x_smooth = np.linspace(0, population_sizes.sum(), 500)
    y_smooth = poly(x_smooth)
    # y_smooth = np.clip(poly(x_smooth), 0.0, None)

    # Plot DivT
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_axis, divt, 'o', color='black', label='Layer values')
    ax.plot(x_smooth, y_smooth, '-', color='black', linewidth=2, label='Cubic fit')
    ax.set_xlim(0, population_sizes.sum())
    ax.set_xticks(x_axis, ['L2/3', 'L4', 'L5', 'L6'], fontsize=14)
    ax.set_ylim([-0.001, 0.015])
    ax.set_yticks([0.0, 0.005, 0.01, 0.015], ['0', '5', '10', '15'])
    ax.set_ylabel('Divergence timing (ms)', fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=1.0)

    # x-axis grid lines
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    boundaries = np.concatenate([[0], np.cumsum(population_sizes)])
    ax_top.set_xticks(boundaries)
    ax_top.set_xticklabels([])
    ax_top.grid(True, axis='x', linestyle='--', alpha=1.0)
    ax_top.spines['top'].set_visible(False)
    ax_top.tick_params(top=False)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # fn = '../trained_wta_models/wta_10_seeds/wta_seed_1.pkl'
    # fn = '../trained_wta_models/wta_adjust_seeds/wta_adjust_seed_4.pkl'
    # fn = '../trained_wta_models/wta_scrambled_01.pkl'
    fn = '../trained_wta_models/wta_scrambled_1.pkl'

    wta_rainbow_plots(fn)


