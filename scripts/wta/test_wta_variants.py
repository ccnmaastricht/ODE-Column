import torch
import numpy as np
import matplotlib.pyplot as plt

from test_wta import compute_wsi_and_divt
from train_wta_optimized import compute_deviation

from src.brain_network import BrainNetwork
from src.utils.paths import config_path, models_path, results_path
from src.utils.plotting.helpers import *
import src.utils.plotting.plotstyle
from src.utils.plotting.colors import *



def run_and_compute_wsi_and_divt(network_paths):
    wsi_all_models = torch.zeros((4, 2, len(network_paths)))  # shape = (4 populations, 2 columns, N models)
    divt_all_models = torch.zeros((4, 2, len(network_paths)))

    for network_i, network_fn in enumerate(network_paths):

        with torch.no_grad():
            network = BrainNetwork.load(network_fn)

            # Consider both cases (i.e. A wins or B wins)
            for winner_idx in [0, 1]:

                # Run the model for different coherences (diff between input A and B)
                coherences = [2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]

                # Set the stimuli
                stims = []
                for i, coherence in enumerate(coherences):
                    if winner_idx == 0:  # column A wins
                        muA = 40.
                        muB = muA - coherence
                    elif winner_idx == 1:  # column B wins
                        muB = 40.
                        muA = muB - coherence
                    stims.append([muA, muB])

                output = network.run(stims, adjoint=False, stochastic=False, reset_state=False)
                firing_rates = network.get_firing_rates(output, return_as_np_array=False)

                fr_results = torch.Tensor(4, 600, len(coherences), 2)
                fr_results[0, :, :, :] = firing_rates[400:1000, :, [0, 8]]  # layer 2/3e
                fr_results[1, :, :, :] = firing_rates[400:1000, :, [2, 10]]  # layer 4e
                fr_results[2, :, :, :] = firing_rates[400:1000, :, [4, 12]]  # layer 5e
                fr_results[3, :, :, :] = firing_rates[400:1000, :, [6, 14]]  # layer 6e

                # Compute winner-selectivity and divergence timing
                wsi, divt = compute_wsi_and_divt(fr_results, network.simulator.dt, winner_idx)
                wsi_all_models[:, winner_idx, network_i] = torch.mean(wsi, dim=1)
                divt_all_models[:, winner_idx, network_i] = torch.mean(divt, dim=1)

    wsi_all_models = torch.mean(wsi_all_models, dim=1)  # average over winner id
    divt_all_models = torch.mean(divt_all_models, dim=1)
    return wsi_all_models, divt_all_models


def plot_opt_csi_and_divt(paths, sub_fig_csi_opt, sub_fig_divt_opt, marker_size=2.0):
    wsi_raw, divt_raw = run_and_compute_wsi_and_divt(paths)
    csi_list = abs(wsi_raw).T
    divt_list = divt_raw.T

    population_sizes = np.array([60606, 28202, 14176, 15837])

    x_axis = []
    for i in range(len(population_sizes)):
        curr_pop_size = population_sizes[i]
        prev_pop_sizes = sum(population_sizes[:i])
        x_axis.append(prev_pop_sizes + (curr_pop_size // 2))
    x_axis = np.array(x_axis)

    x_smooth = np.linspace(0, population_sizes.sum(), 500)

    # Plot all runs
    for csi in csi_list:
        coeffs = np.polyfit(x_axis, csi, deg=3)
        poly = np.poly1d(coeffs)

        # Thin gray curve
        sub_fig_csi_opt.plot(x_smooth, poly(x_smooth), color='gray', alpha=0.6)

    # Mean values
    csi_mean = csi_list.mean(dim=0).cpu().numpy()
    coeffs = np.polyfit(x_axis, csi_mean, deg=3)
    poly = np.poly1d(coeffs)
    sub_fig_csi_opt.plot(x_smooth, poly(x_smooth), color='black')
    sub_fig_csi_opt.plot(x_axis, csi_mean, 'o', color='black', markersize=marker_size)

    csi_sd = csi_list.std(dim=0).cpu().numpy()
    sub_fig_csi_opt.errorbar(x_axis, csi_mean, yerr=csi_sd, ecolor='black', barsabove=True, fmt='none',
                              elinewidth=marker_size/2, capthick=marker_size/2, capsize=marker_size*1.5)

    sub_fig_csi_opt.set_xlim(0, population_sizes.sum())
    sub_fig_csi_opt.set_xticks(x_axis, ['L2/3', 'L4', 'L5', 'L6'])
    sub_fig_csi_opt.set_ylim([-0.6, 1.1])
    sub_fig_csi_opt.set_yticks([-0.5, 0.0, 0.5, 1.0])
    sub_fig_csi_opt.set_ylabel('Choice selectivity index')
    sub_fig_csi_opt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # x-axis grid lines
    ax_top = sub_fig_csi_opt.twiny()
    ax_top.set_xlim(sub_fig_csi_opt.get_xlim())
    boundaries = np.concatenate([[0], np.cumsum(population_sizes)])
    ax_top.set_xticks(boundaries)
    ax_top.set_xticklabels([])
    ax_top.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.tick_params(top=False)

    # --------- DivT ---------
    # Plot all runs
    for divt in divt_list:
        coeffs = np.polyfit(x_axis, divt, deg=3)
        poly = np.poly1d(coeffs)

        # Thin gray curve
        sub_fig_divt_opt.plot(x_smooth, poly(x_smooth), color='gray', alpha=0.6)

    # Mean values
    divt_mean = divt_list.mean(dim=0).cpu().numpy()
    coeffs = np.polyfit(x_axis, divt_mean, deg=3)
    poly = np.poly1d(coeffs)
    sub_fig_divt_opt.plot(x_smooth, poly(x_smooth), color='black')
    sub_fig_divt_opt.plot(x_axis, divt_mean, 'o', color='black', markersize=marker_size)

    divt_sd = divt_list.std(dim=0).cpu().numpy()
    sub_fig_divt_opt.errorbar(x_axis, divt_mean, yerr=divt_sd, ecolor='black', barsabove=True, fmt='none',
                              elinewidth=marker_size/2, capthick=marker_size/2, capsize=marker_size*1.5)

    sub_fig_divt_opt.set_xlim(0, population_sizes.sum())
    sub_fig_divt_opt.set_xticks(x_axis, ['L2/3', 'L4', 'L5', 'L6'])
    sub_fig_divt_opt.set_ylim([0.0, 0.020])
    sub_fig_divt_opt.set_yticks([0.0, 0.005, 0.01, 0.015], ['0', '5', '10', '15'])
    sub_fig_divt_opt.set_ylabel('Divergence timing (ms)')
    sub_fig_divt_opt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # x-axis grid lines
    ax_top = sub_fig_divt_opt.twiny()
    ax_top.set_xlim(sub_fig_divt_opt.get_xlim())
    boundaries = np.concatenate([[0], np.cumsum(population_sizes)])
    ax_top.set_xticks(boundaries)
    ax_top.set_xticklabels([])
    ax_top.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.tick_params(top=False)

    return csi_list, divt_list


def plot_scr_csi_and_divt(all_paths, sub_fig_csi_scr, sub_fig_divt_scr, linewidth_mean=1.0):

    population_sizes = np.array([60606, 28202, 14176, 15837])

    x_axis = []
    for i in range(len(population_sizes)):
        curr_pop_size = population_sizes[i]
        prev_pop_sizes = sum(population_sizes[:i])
        x_axis.append(prev_pop_sizes + (curr_pop_size // 2))
    x_axis = np.array(x_axis)

    x_smooth = np.linspace(0, population_sizes.sum(), 500)

    # Loop through all the scrambled levels
    complete_csi_list = []
    complete_divt_list = []

    colors = [my_green, my_orange, excitatory_color]

    for color_i, paths in enumerate(all_paths):
        color = colors[color_i]

        wsi_raw, divt_raw = run_and_compute_wsi_and_divt(paths)
        csi_list = abs(wsi_raw).T
        divt_list = divt_raw.T
        complete_csi_list.append(csi_list)
        complete_divt_list.append(divt_list)

        # Plot all runs
        for csi in csi_list:
            coeffs = np.polyfit(x_axis, csi, deg=3)
            poly = np.poly1d(coeffs)

            # Thin gray curve
            sub_fig_csi_scr.plot(x_smooth, poly(x_smooth), color=color, alpha=0.6)

        # --------- DivT ---------
        # Plot all runs
        for divt in divt_list:
            coeffs = np.polyfit(x_axis, divt, deg=3)
            poly = np.poly1d(coeffs)

            # Thin gray curve
            sub_fig_divt_scr.plot(x_smooth, poly(x_smooth), color=color, alpha=0.6)

    sub_fig_csi_scr.set_xlim(0, population_sizes.sum())
    sub_fig_csi_scr.set_xticks(x_axis, ['L2/3', 'L4', 'L5', 'L6'])
    sub_fig_csi_scr.set_ylim([-0.6, 1.1])
    sub_fig_csi_scr.set_yticks([-0.5, 0.0, 0.5, 1.0])
    sub_fig_csi_scr.set_ylabel('Choice selectivity index')
    sub_fig_csi_scr.grid(True, axis='y', linestyle='--', alpha=0.5)

    # x-axis grid lines
    ax_top = sub_fig_csi_scr.twiny()
    ax_top.set_xlim(sub_fig_csi_scr.get_xlim())
    boundaries = np.concatenate([[0], np.cumsum(population_sizes)])
    ax_top.set_xticks(boundaries)
    ax_top.set_xticklabels([])
    ax_top.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.tick_params(top=False)

    sub_fig_divt_scr.set_xlim(0, population_sizes.sum())
    sub_fig_divt_scr.set_xticks(x_axis, ['L2/3', 'L4', 'L5', 'L6'])
    sub_fig_divt_scr.set_ylim([-0.01, 0.02])
    sub_fig_divt_scr.set_yticks([0.0, 0.015], ['0', '15'])
    sub_fig_divt_scr.set_ylabel('Divergence timing (ms)')
    sub_fig_divt_scr.grid(True, axis='y', linestyle='--', alpha=0.5)

    # x-axis grid lines
    ax_top = sub_fig_divt_scr.twiny()
    ax_top.set_xlim(sub_fig_divt_scr.get_xlim())
    boundaries = np.concatenate([[0], np.cumsum(population_sizes)])
    ax_top.set_xticks(boundaries)
    ax_top.set_xticklabels([])
    ax_top.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.tick_params(top=False)

    return complete_csi_list, complete_divt_list


def get_orig_connectivity(area='mt'):
    config = config_path('wta_params.toml')
    general_config = config_path('general_params_wta.toml')
    network = BrainNetwork.from_toml(config, general_config)
    network.add_area(area, 2)

    return network.connections[f'recurrent_{area}_{area}'].weights.clone().detach()


def plot_opt_fit(paths, csi, divt, sub_fig_csi_opt_fit, sub_fig_divt_opt_fit, scattersize=10.0):

    original_connectivity = get_orig_connectivity()

    deviations = []
    csi_fits = [0.76, 0.71, 0.65, 0.75, 0.74, 0.58, 0.61, 0.65, 0.65, 0.71]
    divt_fits = [0.82, 0.81, 0.75, 0.75, 0.74, 0.78, 0.75, 0.85, 0.77, 0.81]

    for network_i, network_fn in enumerate(paths):

        with torch.no_grad():
            network = BrainNetwork.load(network_fn)

            network.constrain_weights()
            connectivity_deviation = compute_deviation(network, original_connectivity)
            deviations.append(connectivity_deviation)

    # Linear fit
    x_fit = np.linspace(min(deviations), max(deviations), 100)
    coeffs = np.polyfit(deviations, csi_fits, deg=1)
    fit_line = np.poly1d(coeffs)
    sub_fig_csi_opt_fit.plot(x_fit, fit_line(x_fit), color='black')

    coeffs = np.polyfit(deviations, divt_fits, deg=1)
    fit_line = np.poly1d(coeffs)
    sub_fig_divt_opt_fit.plot(x_fit, fit_line(x_fit), color='black')

    sub_fig_csi_opt_fit.scatter(deviations, csi_fits, color='gray', s=scattersize, alpha=0.6, zorder=2)
    sub_fig_csi_opt_fit.set_xlabel('Connectivity deviation')
    sub_fig_csi_opt_fit.set_ylabel('Empirical fit')

    sub_fig_divt_opt_fit.scatter(deviations, divt_fits, color='gray', s=scattersize, alpha=0.6, zorder=2)
    sub_fig_divt_opt_fit.set_xlabel('Connectivity deviation')
    sub_fig_divt_opt_fit.set_ylabel('Empirical fit')


def plot_scr_fit(all_paths, scr_csi, scr_divt, sub_fig_csi_scr_fit, sub_fig_divt_scr_fit, scattersize=10.0):

    original_connectivity = get_orig_connectivity()

    csi_fits = [
        [0.7, 0.65, 0.69, 0.4, 0.6, 0.57, 0.55, 0.63, 0.67, 0.51],
        [0.3, 0.45, 0.36, 0.37, 0.27],
        [0.2]
    ]
    divt_fits = [
        [0.78, 0.65, 0.73, 0.55, 0.63, 0.66, 0.61, 0.69, 0.59, 0.68],
        [0.37, 0.52, 0.47, 0.48, 0.3],
        [0.25]
    ]

    # Loop through all the scrambled levels
    colors = [my_green, my_orange, excitatory_color]

    all_deviations = []
    all_csi_values = []
    all_divt_values = []

    for color_i, paths in enumerate(all_paths):
        color = colors[color_i]

        deviations = []

        for network_i, network_fn in enumerate(paths):
            with torch.no_grad():
                network = BrainNetwork.load(network_fn)

                network.constrain_weights()
                connectivity_deviation = compute_deviation(network, original_connectivity)
                deviations.append(connectivity_deviation)

        deviations = np.array(deviations)
        csi_values = np.array(csi_fits[color_i])
        divt_values = np.array(divt_fits[color_i])

        all_deviations.extend(deviations)
        all_csi_values.extend(csi_values)
        all_divt_values.extend(divt_values)

        # Scatter plots
        sub_fig_csi_scr_fit.scatter(deviations, csi_values, color=color, s=scattersize, alpha=0.6, zorder=2)
        sub_fig_divt_scr_fit.scatter(deviations, divt_values, color=color, s=scattersize, alpha=0.6, zorder=2)

    all_deviations = np.array(all_deviations)
    all_csi_values = np.array(all_csi_values)
    all_divt_values = np.array(all_divt_values)

    # Fitted line
    csi_coeffs = np.polyfit(all_deviations, all_csi_values, deg=1)
    csi_fit = np.poly1d(csi_coeffs)
    divt_coeffs = np.polyfit(all_deviations, all_divt_values, deg=1)
    divt_fit = np.poly1d(divt_coeffs)
    x_fit = np.linspace(all_deviations.min(),all_deviations.max(),100)

    sub_fig_csi_scr_fit.plot(x_fit, csi_fit(x_fit), color='black')
    sub_fig_divt_scr_fit.plot(x_fit, divt_fit(x_fit), color='black')

    sub_fig_csi_scr_fit.set_xlabel('Connectivity deviation')
    sub_fig_csi_scr_fit.set_ylabel('Empirical fit')

    sub_fig_divt_scr_fit.set_xlabel('Connectivity deviation')
    sub_fig_divt_scr_fit.set_ylabel('Empirical fit')


def make_wta_variants_figure(opt_paths, scr_paths_low, scr_paths_medium, scr_paths_high):

    fig = plt.figure(figsize=(17 / 2.54, 8 / 2.54))

    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0]
    )

    sub_fig_csi_opt = fig.add_subplot(gs[0, 0])
    sub_fig_csi_scr = fig.add_subplot(gs[1, 0])

    sub_fig_csi_opt_fit = fig.add_subplot(gs[0, 1])
    sub_fig_csi_scr_fit = fig.add_subplot(gs[1, 1])

    sub_fig_divt_opt = fig.add_subplot(gs[0, 2])
    sub_fig_divt_scr = fig.add_subplot(gs[1, 2])

    sub_fig_divt_opt_fit = fig.add_subplot(gs[0, 3])
    sub_fig_divt_scr_fit = fig.add_subplot(gs[1, 3])

    # Plot the CSI and DivT layer profiles
    opt_csi, opt_divt = plot_opt_csi_and_divt(opt_paths, sub_fig_csi_opt, sub_fig_divt_opt)
    scr_csi, scr_divt = plot_scr_csi_and_divt([scr_paths_low, scr_paths_medium, scr_paths_high],
                                              sub_fig_csi_scr, sub_fig_divt_scr)

    # Plot the fit with empirical data against deviation from original connectivity
    plot_opt_fit(opt_paths, opt_csi, opt_divt, sub_fig_csi_opt_fit, sub_fig_divt_opt_fit)
    plot_scr_fit([scr_paths_low, scr_paths_medium, scr_paths_high],
                 scr_csi, scr_divt, sub_fig_csi_scr_fit, sub_fig_divt_scr_fit)

    label_mapping = {
        '(A)': sub_fig_csi_opt,
        '(B)': sub_fig_csi_opt_fit,
        '(C)': sub_fig_divt_opt,
        '(D)': sub_fig_divt_opt_fit,
        '(E)': sub_fig_csi_scr,
        '(F)': sub_fig_csi_scr_fit,
        '(G)': sub_fig_divt_scr,
        '(H)': sub_fig_divt_scr_fit
        }

    add_sub_fig_labels(label_mapping)

    fig.tight_layout(
        pad=1.0,
        w_pad=1.0,
        h_pad=2.0)

    finish_plot(fig, results_path('wta', 'wta_variants_fig.pdf'))
    finish_plot(fig, results_path('wta', 'wta_variants_fig.svg'))



if __name__ == '__main__':

    optimized_paths = [models_path('wta_optimized', f'wta_optimized_{i}.pt') for i in range(1, 11)]

    scrambled_paths_low = [models_path('wta_scrambled', f'wta_scrambled_0.1_{i}.pt')
                           for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    scrambled_paths_medium = [models_path('wta_scrambled', f'wta_scrambled_0.5_{i}.pt')
                              for i in [4, 6, 7, 8, 9]]
    scrambled_paths_high = [models_path('wta_scrambled', f'wta_scrambled_1.0_{i}.pt')
                            for i in [10]]

    # history_paths = [models_path('wta_scrambled', f'wta_scrambled_history_1.0_{i}.pt') for i in range(1, 11)]
    # for path in history_paths:
    #     history = torch.load(path, weights_only=False)
    #     print(history['test_losses'])

    make_wta_variants_figure(optimized_paths, scrambled_paths_low, scrambled_paths_medium, scrambled_paths_high)

