import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.brain_network import BrainNetwork
from src.utils.paths import models_path, results_path
from src.utils.plotting.helpers import *
import src.utils.plotting.plotstyle
from src.utils.plotting.colors import *



def plot_timecourse(network_path, sub_fig_timecourse, sub_fig_timecourse_stims):
    network = BrainNetwork.load(network_path)

    stims = torch.tensor([[0., 0., 15., 15., 15., 15., 15., 15.],
                          [0., 0., 0., 0., 0., 0., 15., 15.],
                          [0., 0., 0., 15., 15., 15., 15., 15.],
                          [15., 15., 15., 15., 15., 15., 15., 15.],
                          [0., 0., 0., 0., 0., 0., 0., 15.],
                          [0., 0., 0., 0., 0., 15., 15., 15.],
                          [0., 15., 15., 15., 15., 15., 15., 15.],
                          [0., 0., 0., 0., 15., 15., 15., 15.],
                          [0., 0., 0., 0., 0., 0., 0., 0.]])
    stims_summed = torch.tensor([0., 6., 0., 2., 0., 5., 0., 8.,
                                 0., 1., 0., 3., 0., 7., 0., 4., 0., 0.])
    stim_time_course = stims_summed.repeat_interleave(500)

    with torch.no_grad():

        i = 0
        for stim in stims:

            output = network.run(stim, adjoint=False, stochastic=False, reset_state=False)
            firing_rates = network.get_firing_rates(output, area='v2', population='L23e', return_as_np_array=False)

            if i == 0:
                time_course = firing_rates[:, 0, :]
            elif i > 0:
                time_course = torch.concat([time_course, firing_rates[:, 0, :]], dim=0)
            i += 1

    sub_fig_timecourse.plot(time_course[:, 1], color=my_green, label='Even column')
    sub_fig_timecourse.plot(time_course[:, 0], color=my_orange, label='Odd column')
    sub_fig_timecourse.set_ylabel('L2/3e firing rate (Hz)')
    sub_fig_timecourse.set_xticks([])
    sub_fig_timecourse.tick_params(labelbottom=False)
    sub_fig_timecourse.set_yticks([0.0, 0.5, 1.0, 1.5])
    legend = sub_fig_timecourse.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=1.0,
                       fancybox=False, borderaxespad=0, borderpad=0.3)
    spine_lw = sub_fig_timecourse.spines['top'].get_linewidth()
    legend.get_frame().set_linewidth(spine_lw)
    sub_fig_timecourse.grid(True, linewidth=0.4, linestyle='--', alpha=0.5)

    sub_fig_timecourse_stims.plot(stim_time_course, color='gray')
    sub_fig_timecourse_stims.set_xlabel('Time (s)')
    sub_fig_timecourse_stims.set_ylabel('Input rate (Hz)')
    sub_fig_timecourse_stims.set_ylim([-1.0, 10.0])
    sub_fig_timecourse_stims.set_yticks([0, 2, 4, 6, 8])
    sub_fig_timecourse_stims.set_xticks([0, 2000, 4000, 6000, 8000], [0, 2, 4, 6, 8])
    time = np.arange(len(stim_time_course))
    sub_fig_timecourse_stims.fill_between(time, stim_time_course, where=(stim_time_course % 2 == 0),
                         label='Even input', color=my_green, alpha=1.0, interpolate=True)
    sub_fig_timecourse_stims.fill_between(time, stim_time_course, where=(stim_time_course % 2 != 0),
                         label='Odd input', color=my_orange, alpha=1.0, interpolate=True)
    legend = sub_fig_timecourse_stims.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=1.0,
                       fancybox=False, borderaxespad=0, borderpad=0.3)
    legend.get_frame().set_linewidth(spine_lw)
    sub_fig_timecourse_stims.grid(True, linewidth=0.4, linestyle='--', alpha=0.5)


def plot_history(history_paths, sub_fig_history):
    colors = [excitatory_color, my_green, my_orange, inhibitory_color]

    train_ce = []
    acc_no_noise = []
    acc_low_noise = []
    acc_high_noise = []

    for path in history_paths:
        history = torch.load(path, weights_only=False)

        train_ce.append(np.array(history['train_losses']) - np.array(history['volatility']))
        acc_no_noise.append(history['accuracy_no_noise'])
        acc_low_noise.append(history['accuracy_low_noise'])
        acc_high_noise.append(history['accuracy_high_noise'])

    for i in range(len(train_ce)):
        train_steps = np.arange(len(train_ce[i]))
        sub_fig_history.plot(train_steps, train_ce[i], color=colors[0], alpha=0.4)

        test_steps = np.linspace(0, len(train_ce[i]) - 1, len(acc_no_noise[i]))

        sub_fig_history.plot(test_steps, acc_no_noise[i], color=colors[1], alpha=0.6)
        sub_fig_history.plot(test_steps, acc_low_noise[i], color=colors[2], alpha=0.6)
        sub_fig_history.plot(test_steps, acc_high_noise[i], color=colors[3], alpha=0.6)

    sub_fig_history.set_ylabel('CE loss & accuracy')
    sub_fig_history.set_xlabel('Epoch')
    sub_fig_history.set_xticks([0, 500, 1000])


def make_parity_results_figure(network_paths, history_paths):
    fig = plt.figure(figsize=(17 / 2.54, 12 / 2.54))

    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        width_ratios=[1.2, 1.0, 1.0],
        height_ratios=[1.2, 1.2, 0.8]
    )

    sub_fig_architecture = fig.add_subplot(gs[0, 0])
    sub_fig_history = fig.add_subplot(gs[0, 1])
    sub_fig_weights = fig.add_subplot(gs[0, 2])
    sub_fig_timecourse = fig.add_subplot(gs[1, 0:])
    sub_fig_timecourse_stims = fig.add_subplot(gs[2, 0:], sharex=sub_fig_timecourse)

    plot_timecourse(network_paths[0], sub_fig_timecourse, sub_fig_timecourse_stims)
    plot_history(history_paths, sub_fig_history)

    label_mapping = {
        '(A)': sub_fig_architecture,
        '(B)': sub_fig_history,
        '(C)': sub_fig_weights,
        '(D)': sub_fig_timecourse
        }

    add_sub_fig_labels(label_mapping)

    fig.tight_layout(
        pad=1.0,
        w_pad=1.0,
        h_pad=1.0)

    finish_plot(fig, results_path('parity', 'parity_fig.pdf'))
    finish_plot(fig, results_path('parity', 'parity_fig.svg'))



if __name__ == '__main__':

    network_paths = [models_path('parity', f'parity_{i}.pt') for i in range(1, 7)]
    history_paths = [models_path('parity', f'parity_history_{i}.pt') for i in range(1, 7)]

    make_parity_results_figure(network_paths, history_paths)
