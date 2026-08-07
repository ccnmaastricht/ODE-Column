import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

from src.brain_network import BrainNetwork
from src.utils.paths import models_path, results_path
from src.utils.plotting.helpers import *
import src.utils.plotting.plotstyle
from src.utils.plotting.colors import *



def plot_history(history_paths, sub_fig_history):

    colors = [excitatory_color, my_green, my_orange]

    for i, paths in enumerate(history_paths):

        for path in paths:
            history = torch.load(path, weights_only=False)

            test_loss = np.array(history['test_losses']) - np.array(history['test_fr_reg'])
            sub_fig_history.plot(test_loss, color=colors[i], alpha=0.6)

    sub_fig_history.set_ylabel('Test loss')
    sub_fig_history.set_ylim([0.01, 0.08])
    sub_fig_history.set_xlabel('Training iteration')


def visualize_learned_weights(network_paths, sub_fig_weights, title, color):
    context_weights = []

    # Collect all the learned weights
    for path in network_paths:

        network = BrainNetwork.load(path)
        weights = network.analysis.get_weights(conn_name='input_context_mt', return_as_np=True)

        # Reconfigure the weights to ease interpretability
        weights_reshaped = np.reshape(weights.T, (8, 8)).T
        n_cols = weights_reshaped.shape[1]
        cols = list(range(n_cols))
        cols.remove(6)
        cols.remove(7)
        cols[2:2] = [6, 7]
        weights_reshaped = weights_reshaped[:, cols]

        context_weights.append(weights_reshaped)

    weights_all = np.stack(context_weights)

    mean_matrix = weights_all.mean(axis=0)
    std_matrix = weights_all.std(axis=0)

    # Build annotation strings combining mean and std
    annot_labels = np.empty(mean_matrix.shape, dtype=object)
    for i in range(mean_matrix.shape[0]):
        for j in range(mean_matrix.shape[1]):
            annot_labels[i, j] = f"{mean_matrix[i, j]:.2f}\n±{std_matrix[i, j]:.2f}"

    colors = ['#ffffff', color]

    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging",
        colors,
        N=256)

    heatmap = sns.heatmap(
        mean_matrix,
        # annot=annot_labels,
        fmt='',
        cmap=cmap,
        norm=PowerNorm(gamma=0.5),
        ax=sub_fig_weights,
        cbar_kws={'shrink': 0.7}
    )
    sub_fig_weights.set_title(title)
    sub_fig_weights.set_ylabel("Population")
    sub_fig_weights.set_yticks(np.arange(8) + 0.5,
                               ['L2/3e', 'L2/3i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i'])
    sub_fig_weights.set_yticklabels(sub_fig_weights.get_yticklabels(), rotation=0)
    sub_fig_weights.set_xticks([])
    label_font = sub_fig_weights.yaxis.label.get_fontproperties()
    trans = sub_fig_weights.get_xaxis_transform()
    sub_fig_weights.text(4 / 2, -0.12, 'Relevant\ncontext',
                         ha='center', va='top', transform=trans, fontproperties=label_font)
    sub_fig_weights.text(4 + 4 / 2, -0.12, 'Irrelevant\ncontext',
                         ha='center', va='top', transform=trans, fontproperties=label_font)
    sub_fig_weights.axvline(4, color='black', linewidth=0.6)

    for spine in sub_fig_weights.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.6)

    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([0, 100, 200, 300])

    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.6)


def make_context_results_figure(network_paths, history_paths):
    fig = plt.figure(figsize=(17 / 2.54, 10 / 2.54))

    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0]
    )

    sub_fig_architecture = fig.add_subplot(gs[0, 0:2])
    sub_fig_history = fig.add_subplot(gs[0, 2])
    sub_fig_default_weights = fig.add_subplot(gs[1, 0])
    sub_fig_super_weights = fig.add_subplot(gs[1, 1])
    sub_fig_deep_weights = fig.add_subplot(gs[1, 2])

    plot_history(history_paths, sub_fig_history)
    visualize_learned_weights([network_paths[0][0]], sub_fig_default_weights,
                              'Feedback to L2/3, L5, L6', excitatory_color)
    visualize_learned_weights([network_paths[1][0]], sub_fig_super_weights,
                              'Feedback to L2/3', my_green)
    visualize_learned_weights(network_paths[2], sub_fig_deep_weights,
                              'Feedback to L5, L6', my_orange)

    label_mapping = {
        '(A)': sub_fig_architecture,
        '(B)': sub_fig_history,
        '(C)': sub_fig_default_weights,
        '(D)': sub_fig_super_weights,
        '(E)': sub_fig_deep_weights
        }

    add_sub_fig_labels(label_mapping)

    fig.tight_layout(
        pad=1.0,
        w_pad=1.0,
        h_pad=1.0)

    finish_plot(fig, results_path('context', 'context_fig.pdf'))
    finish_plot(fig, results_path('context', 'context_fig.svg'))


if __name__ == '__main__':

    network_paths = [[models_path('context', f'context_default_{i}.pt') for i in range(1, 11)],
                     [models_path('context', f'context_superficial_only_{i}.pt') for i in range(1, 11)],
                     [models_path('context', f'context_deep_only_{i}.pt') for i in range(1, 11)]]
    history_paths = [[models_path('context', f'context_default_history_{i}.pt') for i in range(1, 11)],
                     [models_path('context', f'context_superficial_only_history_{i}.pt') for i in range(1, 11)],
                     [models_path('context', f'context_deep_only_history_{i}.pt') for i in range(1, 11)]]

    make_context_results_figure(network_paths, history_paths)
