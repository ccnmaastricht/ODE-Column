import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.brain_network import BrainNetwork
from src.utils.set_seed import set_seed
import src.utils.plotting.plotstyle
from src.utils.plotting.plotstyle import panel_label
from src.utils.plotting.colors import *


def finish_plot(fig, filename):
    fig.savefig(
        filename,
        bbox_inches="tight",
        transparent=True)


def add_sub_fig_labels(mapping):
    for label, sub_fig in mapping.items():
        sub_fig.text(
            -0.15,
            1.05,
            label,
            transform=sub_fig.transAxes,
            **panel_label)


def plot_time_course(network_paths, sub_fig_top, sub_fig_bottom, network_to_plot=0):
    idx_path = network_paths[network_to_plot]
    config_path = '../../config/wta_test_params.toml'
    network = BrainNetwork.load(idx_path, model_config_path=config_path)

    with torch.no_grad():

        i = 0
        for stim in [[0., 0.], [0., 0.], [10., 30.], [0., 0.], [30., 10.], [0., 0.], [20., 20.], [20., 20.], [20., 20.], [20., 20.], [0., 0.]]:

            output = network.run(stim, adjoint=False, stochastic=True, reset_state=False)
            firing_rates = network.get_firing_rates(output, return_as_np_array=False)

            # Store results
            # TODO: get L2/3e firing rates immediately
            firing_rates = firing_rates[:, 0, [0, 8]]

            stim_tensor = torch.tensor(stim)
            stim_over_time = torch.tile(stim_tensor, (len(firing_rates), 1))

            if i == 1:
                time_course = firing_rates
                stim_time_course = stim_over_time
            elif i > 1:
                time_course = torch.concat([time_course, firing_rates], dim=0)
                stim_time_course = torch.concat([stim_time_course, stim_over_time], dim=0)
            i += 1

    dt = network.simulator.dt
    total_time = np.arange(time_course.shape[0]) * dt
    time_ticks = np.arange(0, total_time[-1], 2.0)

    # Top plot: firing rates
    sub_fig_top.plot(total_time, time_course[:, 0], color=myred2, label='Column A')
    sub_fig_top.plot(total_time, time_course[:, 1], color=myblue2, label='Column B')
    sub_fig_top.set_ylabel('L2/3e firing rates (Hz)')
    sub_fig_top.set_xticks([])
    sub_fig_top.set_yticks([0.0, 0.5, 1.0])
    sub_fig_top.tick_params(labelbottom=False)
    sub_fig_top.legend()
    sub_fig_top.grid(True, linewidth=0.4, linestyle='--', alpha=0.5)

    # Bottom plot: stimulus-driven input
    sub_fig_bottom.plot(total_time, stim_time_course[:, 0], color=myred2, label='Input A', linewidth=1.0)
    sub_fig_bottom.plot(total_time, stim_time_course[:, 1], color=myblue2, label='Input B', linewidth=1.0, linestyle='--')
    sub_fig_bottom.set_xlabel('Time (s)')
    sub_fig_bottom.set_ylabel('Input rates (Hz)')
    sub_fig_bottom.set_xticks(time_ticks)
    sub_fig_bottom.set_ylim(-5.0, 40.0)
    sub_fig_bottom.legend()
    sub_fig_bottom.grid(True, linewidth=0.4, linestyle='--', alpha=0.5)


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

    # plt.plot(A1_smooth)
    # plt.plot(A2_smooth)
    # plt.plot(A_diff)
    # plt.show()

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


def run_bistable_perception(network_path):
    config_path = '../../config/wta_bistable_perception_params.toml'
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
            firing_rates = network.get_firing_rates(output)

            # TODO: get L2/3e firing rates immediately
            firing_rates = firing_rates[:, :, [0, 8]]

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

            # print(input_grid[sample])
            # print(dom)
            # plt.plot(total_firing_rates[:, sample, 0])
            # plt.plot(total_firing_rates[:, sample, 1])
            # plt.show()


        dom_dur = np.reshape(dom_dur, (21, 21))

        return dom_dur


def plot_bistable_perception(network_paths, sub_fig, network_to_plot=0):
    idx_path = network_paths[network_to_plot]

    dominance_duration = run_bistable_perception(idx_path)

    colors = [myred2, myyellow, myblue2]

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

    cbar = sub_fig.figure.colorbar(heatmap, ax=sub_fig)
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
        if l_idx == 0 or l_idx == 2:  # Remove x-ticks for top subplots
            sub_sub_fig.set_xticklabels([])
        else:  # For bottom subplots: relabel x-axis from 0–600 to -100–500
            xlabels = xticks - 100  # Shift labels
            xlabels = xlabels // 10
            sub_sub_fig.set_xticklabels(xlabels)
        sub_sub_fig.set_xlim(0, 600)

    # sub_fig.set_xlabel('Time (ms)', labelpad=20)
    # sub_fig.set_ylabel('Firing rates (Hz)', labelpad=20)

    # Color bar
    cbar = fig.colorbar(sm, ax=list(sub_fig_dict.values()), orientation='horizontal', location='top', fraction=0.04, pad=0.1)
    cbar.set_label('Relative evidence in Hz')
    cbar.set_ticks([min(coherences), max(coherences)])
    cbar.set_ticklabels([f'{min(coherences):.2f}', f'{max(coherences):.2f}'])


def plot_layer_results(network_paths, fig, overlay_fig_4, sub_fig_4_dict, sub_fig_5,
                       sub_fig_6, sub_fig_7, sub_fig_8, network_to_plot=0):
    wsi_all_models = torch.zeros((4, len(network_paths)))  # shape = (4 columns, N models)
    divt_all_models = torch.zeros((4, len(network_paths)))

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

                # Todo: get the correct layers right away!
                firing_rates = network.get_firing_rates(output, return_as_np_array=False)

                fr_results = torch.Tensor(4, 600, len(coherences), 2)
                fr_results[0, :, :, :] = firing_rates[400:1000, :, [0, 8]]  # layer 2/3e
                fr_results[1, :, :, :] = firing_rates[400:1000, :, [2, 10]]  # layer 4e
                fr_results[2, :, :, :] = firing_rates[400:1000, :, [4, 12]]  # layer 5e
                fr_results[3, :, :, :] = firing_rates[400:1000, :, [6, 14]]  # layer 6e

                # Plot layer-wise firing rates; only for the specified network and winner=A
                if network_i == network_to_plot and winner_idx == 0:
                    plot_firing_rates_per_layer(fr_results, coherences, fig, overlay_fig_4, sub_fig_4_dict)


def plot_network_architecture(sub_fig_1):
    img = plt.imread("column WTA.png")
    sub_fig_1.imshow(img, interpolation="none")
    sub_fig_1.axis("off")


def create_wta_results_fig(network_paths, seed=1):

    set_seed(seed)

    # fig = plt.figure(figsize=(17 / 2.54, 13 / 2.54))
    # fig.subplots_adjust(
    #     left=0.07,
    #     right=0.98,
    #     bottom=0.08,
    #     top=0.95,
    #     wspace=0.4,
    #     hspace=0.4)
    #
    # gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1.4, 1.1, 1.1], height_ratios=[1.0, 0.8, 0.8])
    #
    # sub_fig_1 = fig.add_subplot(gs[0, 0])
    #
    # sub_fig_2 = gs[1, 0].subgridspec(2,1, height_ratios=[1.2, 0.8])
    # sub_fig_2_top = fig.add_subplot(sub_fig_2[0])
    # sub_fig_2_bottom = fig.add_subplot(sub_fig_2[1], sharex=sub_fig_2_top)
    #
    # sub_fig_3 = fig.add_subplot(gs[2, 0])
    #
    # sub_fig_4 = gs[0, 1:].subgridspec(2,2, hspace=0.2)
    # sub_fig_4_dict = {
    #     'L2/3': fig.add_subplot(sub_fig_4[0, 0]),
    #     'L4': fig.add_subplot(sub_fig_4[1, 0]),
    #     'L5': fig.add_subplot(sub_fig_4[0, 1]),
    #     'L6': fig.add_subplot(sub_fig_4[1, 1]),
    # }
    # overlay_fig_4 = fig.add_subplot(gs[0, 1:])
    # overlay_fig_4.set_facecolor('none')
    # overlay_fig_4.set_in_layout(False)
    # overlay_fig_4.set_frame_on(False)
    # overlay_fig_4.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    #
    # sub_fig_5 = fig.add_subplot(gs[1, 1])
    # sub_fig_6 = fig.add_subplot(gs[1, 2])
    # sub_fig_7 = fig.add_subplot(gs[2, 1])
    # sub_fig_8 = fig.add_subplot(gs[2, 2])

    fig = plt.figure(figsize=(17 / 2.54, 13 / 2.54))

    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        width_ratios=[1.4, 1.1, 1.1],
        height_ratios=[1.4, 0.8, 0.8]
    )

    # -------------------------
    # Left column
    # -------------------------
    sub_fig_1 = fig.add_subplot(gs[0, 0])

    sub_fig_2 = gs[1, 0].subgridspec(2, 1, height_ratios=[1.2, 0.8])
    sub_fig_2_top = fig.add_subplot(sub_fig_2[0])
    sub_fig_2_bottom = fig.add_subplot(sub_fig_2[1], sharex=sub_fig_2_top)

    sub_fig_3 = fig.add_subplot(gs[2, 0])

    # -------------------------
    # Subplot 4
    # -------------------------
    sub_fig_4 = gs[0, 1:].subgridspec(
        2, 2,
        hspace=0.35,  # increased vertical spacing (important)
        wspace=0.25
    )

    sub_fig_4_dict = {
        'L2/3': fig.add_subplot(sub_fig_4[0, 0]),
        'L4': fig.add_subplot(sub_fig_4[1, 0]),
        'L5': fig.add_subplot(sub_fig_4[0, 1]),
        'L6': fig.add_subplot(sub_fig_4[1, 1]),
    }

    # -------------------------
    # Right-side lower panels
    # -------------------------
    sub_fig_5 = fig.add_subplot(gs[1, 1])
    sub_fig_6 = fig.add_subplot(gs[1, 2])
    sub_fig_7 = fig.add_subplot(gs[2, 1])
    sub_fig_8 = fig.add_subplot(gs[2, 2])

    label_mapping = {
        '(A)': sub_fig_1,
        '(B)': sub_fig_2_top,
        '(C)': sub_fig_3,
        '(D)': sub_fig_4_dict['L2/3'],
        '(E)': sub_fig_5,
        '(F)': sub_fig_6,
        '(G)': sub_fig_7,
        '(H)': sub_fig_8,
        }

    plot_network_architecture(sub_fig_1)
    plot_time_course(network_paths, sub_fig_2_top, sub_fig_2_bottom)
    plot_bistable_perception(network_paths, sub_fig_3)
    plot_layer_results(network_paths, fig, None, sub_fig_4_dict, sub_fig_5, sub_fig_6, sub_fig_7, sub_fig_8)

    add_sub_fig_labels(label_mapping)

    fig.tight_layout(
        pad=1.5,
        w_pad=1.0,
        h_pad=1.0)

    finish_plot(fig, 'wta_fig.pdf')



if __name__ == '__main__':

    network_paths = ['wta.pt']

    # === Results to show ===
    # Time courses
    # Bistable perception
    # Rainbow plots
    # Wsi, Csi
    # Divergence timing

    # run_bistable_perception('wta.pt')

    create_wta_results_fig(network_paths)


