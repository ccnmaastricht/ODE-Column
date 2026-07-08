import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.brain_network import BrainNetwork
from src.utils.paths import config_dir, models_dir, results_dir
from src.utils.set_seed import set_seed
from src.utils.plotting.helpers import *
from src.utils.plotting.plot_history import plot_training_history
import src.utils.plotting.plotstyle
from src.utils.plotting.colors import *
from src.utils.bistable_perception import dominance_time



def run_bistable_perception(network_path):
    config_path = config_dir('wta_bistable_perception_params.toml')
    network = BrainNetwork.load(network_path, model_config_path=config_path)

    inputs = [10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
              20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30.]

    input_grid = []
    for muA in inputs:
        for muB in inputs:
            input_grid.append([muA, muB])

    with torch.no_grad():

        for i in range(5):  # 5 * 10s of simulation

            output = network.run(input_grid, adjoint=False, stochastic=True, reset_state=False)
            firing_rates = network.get_firing_rates(output, population='L23e')

            if i == 0:
                total_firing_rates = firing_rates
            else:
                total_firing_rates = np.concatenate([total_firing_rates, firing_rates], axis=0)

        # Dominance duration
        dom_dur = []
        for sample in range(len(input_grid)):
            dom = dominance_time(
                total_firing_rates[:, sample, 0],
                total_firing_rates[:, sample, 1],
                dt=network.simulator.dt,
                thresh=0.0001,
                sliding_window=1000)

            dom_dur.append(np.round(np.sum(dom), 2))

        dom_dur = np.reshape(dom_dur, (21, 21))

        return dom_dur


def compute_wsi_and_divt(firing_rates, dt, winner_idx, divt_threshold=0.1, divt_window=0.02):
    divt_window = int(divt_window / dt)  # convert from seconds to timesteps

    wsi_results = torch.zeros((firing_rates.shape[0], firing_rates.shape[2]))
    divt_results = torch.zeros((firing_rates.shape[0], firing_rates.shape[2]))

    for i_layer in range(firing_rates.shape[0]):
        for i_coh in range(0, firing_rates.shape[2]):
            winner_column = firing_rates[i_layer, :, i_coh, winner_idx]
            loser_column = firing_rates[i_layer, :, i_coh, (1 - winner_idx)]

            # Compute WSI (winner-selectivity index)
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

    # Ensure non-negative divergence timings (happens when trajectories never intersect and thus never diverge)
    divt_results = torch.clamp(divt_results, min=0.0)
    return wsi_results, divt_results


def plot_time_course(network_paths, sub_fig_top, sub_fig_bottom, network_to_plot=0):
    idx_path = network_paths[network_to_plot]
    config_path = config_dir('wta_test_params.toml')
    network = BrainNetwork.load(idx_path, model_config_path=config_path)

    with torch.no_grad():

        i = 0
        for stim in [[0., 0.], [0., 0.], [30., 10.], [0., 0.], [10., 30.], [0., 0.], [20., 20.], [20., 20.], [20., 20.], [20., 20.], [0., 0.]]:

            output = network.run(stim, adjoint=False, stochastic=True, reset_state=False)
            firing_rates = network.get_firing_rates(output, population='L23e', return_as_np_array=False)

            # Store results_old
            stim_tensor = torch.tensor(stim)
            stim_over_time = torch.tile(stim_tensor, (len(firing_rates), 1))

            if i == 1:
                time_course = firing_rates[:, 0, :]
                stim_time_course = stim_over_time
            elif i > 1:
                time_course = torch.concat([time_course, firing_rates[:, 0, :]], dim=0)
                stim_time_course = torch.concat([stim_time_course, stim_over_time], dim=0)
            i += 1

    dt = network.simulator.dt
    total_time = np.arange(time_course.shape[0]) * dt
    time_ticks = np.arange(0, total_time[-1]+2.0, 2.0)

    # Top plot: firing rates
    sub_fig_top.plot(total_time, time_course[:, 0], color=my_green, label='Column A')
    sub_fig_top.plot(total_time, time_course[:, 1], color=my_orange, label='Column B')
    sub_fig_top.set_ylabel('L2/3e firing rate (Hz)')
    sub_fig_top.set_xticks([])
    sub_fig_top.set_yticks([0.0, 0.5, 1.0])
    sub_fig_top.set_xlim(time_ticks[0], time_ticks[-1])
    sub_fig_top.tick_params(labelbottom=False)
    sub_fig_top.legend()
    sub_fig_top.grid(True, linewidth=0.4, linestyle='--', alpha=0.5)

    # Bottom plot: stimulus-driven input
    sub_fig_bottom.plot(total_time, stim_time_course[:, 0], color=my_green, label='Input A', linewidth=2.0)
    sub_fig_bottom.plot(total_time, stim_time_course[:, 1], color=my_orange, label='Input B', linewidth=2.0, linestyle='--')
    sub_fig_bottom.set_xlabel('Time (s)')
    sub_fig_bottom.set_ylabel('Input rate (Hz)')
    sub_fig_bottom.set_xticks(time_ticks)
    sub_fig_bottom.set_xlim(time_ticks[0], time_ticks[-1])
    sub_fig_bottom.set_ylim(-5.0, 40.0)
    sub_fig_bottom.legend()
    sub_fig_bottom.grid(True, linewidth=0.4, linestyle='--', alpha=0.5)


def plot_bistable_perception(network_paths, sub_fig, network_to_plot=0):
    idx_path = network_paths[network_to_plot]

    # dominance_duration = run_bistable_perception(idx_path)

    dominance_duration = torch.rand((21, 21)) * 50.0  # TODO: note that not running real bistable perception

    colors = [my_green, '#ffffff', my_orange]

    cmap_custom = LinearSegmentedColormap.from_list(
        "custom_diverging",
        colors,
        N=256)

    heatmap = sub_fig.imshow(
        dominance_duration,
        cmap=cmap_custom,
        interpolation="nearest",
        vmin=-50.0,
        vmax=50.0,
        extent=[10, 30, 10, 30],
        origin="lower")

    cbar = sub_fig.figure.colorbar(heatmap, ax=sub_fig, orientation='horizontal', location='top')
    cbar.set_label("Dominance duration (s)")

    sub_fig.set_xticks([10, 20, 30])
    sub_fig.set_yticks([10, 20, 30])

    sub_fig.set_xlabel("Input column A (Hz)")
    sub_fig.set_ylabel("Input column B (Hz)")


def plot_firing_rates_per_layer(firing_rates, coherences, fig, sub_fig, sub_fig_dict):
    colors = plt.get_cmap('rainbow', len(coherences))
    layers = ['L2/3', 'L4', 'L5', 'L6']

    # Normalize for color mapping for the colorbar
    norm = mcolors.Normalize(vmin=min(coherences), vmax=max(coherences))
    sm = cm.ScalarMappable(cmap=colors, norm=norm)
    sm.set_array([])

    for l_idx in range(4):

        if l_idx == 0:
            sub_sub_fig = sub_fig_dict['L2/3']
        elif l_idx == 1:
            sub_sub_fig = sub_fig_dict['L4']
        elif l_idx == 2:
            sub_sub_fig = sub_fig_dict['L5']
        elif l_idx == 3:
            sub_sub_fig = sub_fig_dict['L6']

        for c_idx in range(len(coherences)):
            color = colors(c_idx)

            sub_sub_fig.plot(firing_rates[l_idx, :, c_idx, 1], linestyle='--', color=color, zorder=1)
            sub_sub_fig.plot(firing_rates[l_idx, :, c_idx, 0], color=color, zorder=2)

        # Add y label indicating layer
        sub_sub_fig.set_title(layers[l_idx])

        # Add vertical line at x=0 (stimulus)
        sub_sub_fig.axvline(x=100, color='gray', linestyle='--', linewidth=0.5)

        # Remove the upper and right-most borders
        sub_sub_fig.spines['top'].set_visible(False)
        sub_sub_fig.spines['right'].set_visible(False)

        # Set y-ticks for each layer separately
        if l_idx == 0:
            sub_sub_fig.set_yticks([0, 1])
        if l_idx == 1:
            sub_sub_fig.set_yticks([5, 10, 15])
        if l_idx == 2:
            sub_sub_fig.set_yticks([0, 0.15, 0.3])
        if l_idx == 3:
            sub_sub_fig.set_yticks([0, 1e-6, 2e-6], ['0', '1e-6', '2e-6'])

        # Set x-ticks
        xticks = np.arange(0, 601, 100)
        sub_sub_fig.set_xticks(xticks)
        if l_idx < 3:  # Remove x-ticks for top subplots
            sub_sub_fig.set_xticklabels([])
        else:  # For bottom subplots: relabel x-axis from 0–600 to -100–500
            xlabels = xticks - 100  # Shift labels
            xlabels = xlabels // 10
            sub_sub_fig.set_xticklabels(xlabels)
            sub_sub_fig.set_xlabel('Time (ms)')
        sub_sub_fig.set_xlim(0, 600)

    # fig.text(0.72, 0.60, "Time (ms)", ha="center", va="center", fontsize=6)
    # fig.text(0.47, 0.77, "Firing rate (Hz)", rotation="vertical", ha="center", va="center", fontsize=6)

    # # Color bar
    # cbar = fig.colorbar(sm, ax=list(sub_fig_dict.values()), orientation='horizontal', location='top', fraction=0.04, pad=0.1)
    # cbar.set_label('Relative evidence in Hz')
    # cbar.set_ticks([min(coherences), max(coherences)])
    # cbar.set_ticklabels([f'{min(coherences):.2f}', f'{max(coherences):.2f}'])


def plot_wsi_and_divt(wsi_results, divt_results, sub_fig_wsi, sub_fig_divt, linewidth_mean=1.0):
    x = np.arange(4)

    # ----- WSI -----
    mean_wsi = torch.mean(wsi_results, dim=1)
    sd_wsi = torch.std(wsi_results, dim=1)

    sub_fig_wsi.plot(mean_wsi, color='black', marker='o', markersize=linewidth_mean*2, linewidth=linewidth_mean)
    sub_fig_wsi.errorbar(np.arange(4), mean_wsi, yerr=sd_wsi, ecolor='black', elinewidth=linewidth_mean,
                         capthick=linewidth_mean, capsize=linewidth_mean*3, barsabove=True)

    # Plotting per coherence level (rainbow)
    colors = plt.get_cmap('rainbow', wsi_results.shape[1])
    for i_coh in range(0, wsi_results.shape[1]):
        color = colors(i_coh)
        sub_fig_wsi.plot(wsi_results[:, i_coh], color=color, zorder=1)

    sub_fig_wsi.grid(True, linestyle="--", alpha=0.5)
    sub_fig_wsi.set_xticks(x)
    sub_fig_wsi.set_xticklabels(["L2/3", "L4", "L5", "L6"])
    sub_fig_wsi.set_yticks([-0.6, 0.0, 0.6])
    sub_fig_wsi.set_xlim(-0.5, 3.5)
    sub_fig_wsi.set_ylim(-0.8, 1.0)
    sub_fig_wsi.set_ylabel("Winner selectivity index")

    # ----- DivT -----
    mean_divt = torch.mean(divt_results, dim=1)
    sd_divt = torch.std(divt_results, dim=1)

    sub_fig_divt.plot(mean_divt, color='black', marker='o', markersize=linewidth_mean*2, linewidth=linewidth_mean)
    sub_fig_divt.errorbar(np.arange(4), mean_divt, yerr=sd_divt, ecolor='black', elinewidth=linewidth_mean,
                         capthick=linewidth_mean, capsize=linewidth_mean*3, barsabove=True)

    # Plotting per coherence level (rainbow)
    colors = plt.get_cmap('rainbow', divt_results.shape[1])
    for i_coh in range(0, divt_results.shape[1]):
        color = colors(i_coh)
        sub_fig_divt.plot(divt_results[:, i_coh], color=color, zorder=1)

    sub_fig_divt.grid(True, linestyle="--", alpha=0.5)
    sub_fig_divt.set_xticks(x)
    sub_fig_divt.set_xticklabels(["L2/3", "L4", "L5", "L6"])
    sub_fig_divt.set_yticks([0.0, 0.005, 0.01, 0.015])
    sub_fig_divt.set_yticklabels(["0", "5", "10", "15"])
    sub_fig_divt.set_xlim(-0.5, 3.5)
    sub_fig_divt.set_ylabel("Divergence timing (ms)")


def plot_interp_wsi_and_divt(wsi_list, divt_list, sub_fig_wsi, sub_fig_divt, linewidth_mean=1.0):
    population_sizes = np.array([60606, 28202, 14176, 15837])

    x_axis = []
    for i in range(len(population_sizes)):
        curr_pop_size = population_sizes[i]
        prev_pop_sizes = sum(population_sizes[:i])
        x_axis.append(prev_pop_sizes + (curr_pop_size // 2))
    x_axis = np.array(x_axis)

    # Smooth x for plotting
    x_smooth = np.linspace(0, population_sizes.sum(), 500)

    # Color map (automatically spreads colors)
    colors = plt.cm.tab10(np.linspace(0, 1, len(wsi_list)))

    # ----- WSI -----
    for i, wsi in enumerate(wsi_list):
        # Fit polynomial
        coeffs = np.polyfit(x_axis, wsi, deg=3)
        poly = np.poly1d(coeffs)
        y_smooth = poly(x_smooth)

        # Plot points
        sub_fig_wsi.plot(x_axis, wsi, color='black', marker='o', markersize=linewidth_mean*2, ls='')

        # Plot smooth curve
        sub_fig_wsi.plot(x_smooth, y_smooth, '-', color='black')

    sub_fig_wsi.set_xlim(0, population_sizes.sum())
    sub_fig_wsi.set_xticks(x_axis, ['L2/3', 'L4', 'L5', 'L6'])
    sub_fig_wsi.set_ylim([-0.1, 1.1])
    sub_fig_wsi.set_yticks([0.0, 0.5, 1.0])
    sub_fig_wsi.set_ylabel('Choice Selectivity Index')
    sub_fig_wsi.grid(True, axis='y', linestyle='--', alpha=0.5)

    # x-axis grid lines
    ax_top = sub_fig_wsi.twiny()
    ax_top.set_xlim(sub_fig_wsi.get_xlim())
    boundaries = np.concatenate([[0], np.cumsum(population_sizes)])
    ax_top.set_xticks(boundaries)
    ax_top.set_xticklabels([])
    ax_top.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.tick_params(top=False)

    # --------- DivT ---------
    for i, divt in enumerate(divt_list):
        # Fit polynomial
        coeffs = np.polyfit(x_axis, divt, deg=3)
        poly = np.poly1d(coeffs)
        y_smooth = poly(x_smooth)

        # Plot raw points
        sub_fig_divt.plot(x_axis, divt, color='black', marker='o', markersize=linewidth_mean*2, ls='')

        # Plot smooth curve
        sub_fig_divt.plot(x_smooth, y_smooth, '-', color='black')

    sub_fig_divt.set_xlim(0, population_sizes.sum())
    sub_fig_divt.set_xticks(x_axis, ['L2/3', 'L4', 'L5', 'L6'])
    # sub_fig_divt.set_ylim([-0.001, 0.015])
    sub_fig_divt.set_ylim([-0.005, 0.02])
    sub_fig_divt.set_yticks([0.0, 0.005, 0.01, 0.015], ['0', '5', '10', '15'])
    sub_fig_divt.set_ylabel('Divergence timing (ms)')
    sub_fig_divt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # x-axis grid lines
    ax_top = sub_fig_divt.twiny()
    ax_top.set_xlim(sub_fig_divt.get_xlim())
    boundaries = np.concatenate([[0], np.cumsum(population_sizes)])
    ax_top.set_xticks(boundaries)
    ax_top.set_xticklabels([])
    ax_top.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.tick_params(top=False)


def plot_layer_results(network_paths, fig, sub_fig_fr, sub_fig_fr_dict, sub_fig_wsi,
                       sub_fig_wsi_interp, sub_fig_divt, sub_fig_divt_interp, network_to_plot=0):
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

                # Run the network first without input, then introduce input
                empty_stims = np.zeros_like(stims)
                resting_state = network.run(empty_stims, adjoint=False, stochastic=False)
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

                # Plot layer-wise firing rates and profile; only for the specified network and winner=A
                if network_i == network_to_plot and winner_idx == 0:
                    plot_firing_rates_per_layer(fr_results, coherences, fig, sub_fig_fr, sub_fig_fr_dict)
                    plot_wsi_and_divt(wsi, divt, sub_fig_wsi, sub_fig_divt)

    # Plot selectivity index and divergence timing for all models
    wsi_all_models = torch.mean(wsi_all_models, dim=1)  # average over winner id
    divt_all_models = torch.mean(divt_all_models, dim=1)
    plot_interp_wsi_and_divt(abs(wsi_all_models).T, divt_all_models.T, sub_fig_wsi_interp, sub_fig_divt_interp)


def plot_network_architecture(sub_fig):
    img = plt.imread("wta_network_image.png")
    sub_fig.imshow(img)
    sub_fig.axis("off")


def plot_learned_weights(network_paths, sub_fig_left, sub_fig_right):

    weights = []

    for path in network_paths:
        network = BrainNetwork.load(path)

        self_excitation_weights = network.connections['self_excitation_mt_mt'].weights.detach().numpy()
        intrinsic_weights = network.connections['recurrent_mt_mt'].weights.detach().numpy()
        lateral_weights = network.connections['lateral_mt_mt'].weights.detach().numpy()

        weights.append([
            (self_excitation_weights[0, 0] + intrinsic_weights[0, 0]),
            (self_excitation_weights[8, 8] + intrinsic_weights[8, 8]),
            lateral_weights[9, 0],
            lateral_weights[1, 8]])

    weights = np.asarray(weights)
    n_runs, n_weights = weights.shape

    means = weights.mean(axis=0)
    stds = weights.std(axis=0)

    rng = np.random.default_rng(42)

    for i in range(n_weights):

        x = i + rng.uniform(-0.08, 0.08, size=n_runs)  # Small horizontal jitter

        color = my_green if i % 2 == 0 else my_orange
        sub_fig = sub_fig_left if i < 2 else sub_fig_right

        # Individual learned weights
        sub_fig.scatter(x, weights[:, i], color=color, alpha=0.6, s=30, zorder=2)

        # Mean ± SD
        sub_fig.errorbar(i, means[i], yerr=stds[i], fmt="o", color="black", capsize=5, markersize=2.0, zorder=3)

    sub_fig_right.set_xticks(range(2))
    sub_fig_left.set_xticklabels(['Column A', 'Column B'])
    sub_fig_left.set_ylabel("L2/3e self-excitation weight")
    # sub_fig.set_ylim(650, 950)
    # sub_fig.set_yticks([700, 800, 900])
    sub_fig_left.grid(True, linewidth=0.4, linestyle='--', alpha=0.5)

    sub_fig_right.set_xticks(range(2))
    sub_fig_right.set_xticklabels(['Column A', 'Column B'])
    sub_fig_right.set_ylabel("Lateral inhibition weight")
    # sub_fig_right.set_ylim(650, 950)
    # sub_fig_right.set_yticks([700, 800, 900])
    sub_fig_right.grid(True, linewidth=0.4, linestyle='--', alpha=0.5)


def create_wta_general_results_fig(network_paths, seed=1):

    set_seed(seed)

    fig = plt.figure(figsize=(17 / 2.54, 12 / 2.54))

    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1.2, 0.6, 0.6],
        height_ratios=[0.8, 1.0]
    )

    sub_fig_network = fig.add_subplot(gs[0, 0])

    sub_fig_weights = gs[0, 1:].subgridspec(1, 2)
    sub_fig_weights_left = fig.add_subplot(sub_fig_weights[0])
    sub_fig_weights_right = fig.add_subplot(sub_fig_weights[1], sharex=sub_fig_weights_left)

    sub_fig_fr = gs[1, :2].subgridspec(2, 1, height_ratios=[1.2, 0.8])
    sub_fig_fr_top = fig.add_subplot(sub_fig_fr[0])
    sub_fig_fr_bottom = fig.add_subplot(sub_fig_fr[1], sharex=sub_fig_fr_top)

    sub_fig_bp = fig.add_subplot(gs[1, 2])

    label_mapping = {
        '(A)': sub_fig_network,
        '(B)': sub_fig_weights_left,
        '(C)': sub_fig_fr_top,
        '(D)': sub_fig_bp,
        }

    plot_network_architecture(sub_fig_network)
    plot_learned_weights(network_paths, sub_fig_weights_left, sub_fig_weights_right)
    plot_time_course(network_paths, sub_fig_fr_top, sub_fig_fr_bottom)
    plot_bistable_perception(network_paths, sub_fig_bp)

    add_sub_fig_labels(label_mapping)

    fig.tight_layout(
        pad=1.5,
        w_pad=1.0,
        h_pad=1.0)

    finish_plot(fig, results_dir('wta', 'wta_fig.pdf'))


def create_wta_layers_results_fig(network_paths):

    fig = plt.figure(figsize=(17 / 2.54, 10 / 2.54))

    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0]
    )

    sub_fig_fr = gs[0:, 0].subgridspec(
        4, 1,
        # hspace=0.35,
        # wspace=0.25
    )

    sub_fig_fr_dict = {
        'L2/3': fig.add_subplot(sub_fig_fr[0]),
        'L4': fig.add_subplot(sub_fig_fr[1]),
        'L5': fig.add_subplot(sub_fig_fr[2]),
        'L6': fig.add_subplot(sub_fig_fr[3]),
    }

    sub_fig_wsi = fig.add_subplot(gs[0, 1])
    sub_fig_wsi_interp = fig.add_subplot(gs[0, 2])
    sub_fig_divt = fig.add_subplot(gs[1, 1])
    sub_fig_divt_interp = fig.add_subplot(gs[1, 2])

    label_mapping = {
        '(A)': sub_fig_fr_dict['L2/3'],
        '(B)': sub_fig_wsi,
        '(C)': sub_fig_wsi_interp,
        '(D)': sub_fig_divt,
        '(E)': sub_fig_divt_interp,
        }

    plot_layer_results(network_paths, fig, sub_fig_fr, sub_fig_fr_dict, sub_fig_wsi, sub_fig_wsi_interp, sub_fig_divt, sub_fig_divt_interp)

    add_sub_fig_labels(label_mapping)

    fig.tight_layout(
        pad=3.0,
        w_pad=3.0,
        h_pad=1.0)

    finish_plot(fig, results_dir('wta', 'wta_layers_fig.pdf'))



if __name__ == '__main__':

    network_paths = [models_dir('wta', f'wta_{i}.pt') for i in range(1, 5)]
    history_paths = [models_dir('wta', f'wta_history_{i}.pt') for i in range(1, 5)]

    create_wta_general_results_fig(network_paths)
    create_wta_layers_results_fig(network_paths)

    # plot_training_history(history_paths)

