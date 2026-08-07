import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.brain_network import BrainNetwork
from src.utils.paths import models_path, results_path
import src.utils.plotting.plotstyle
from src.utils.plotting.colors import *
from src.utils.plotting.helpers import *
from train_digits import prepare_ds



def plot_training_history(history_paths, sub_fig_training_history, sub_fig_accuracy_history):

    colors = [excitatory_color, inhibitory_color, my_orange]

    train_ce = []
    test_ce = []
    accuracy = []

    for path in history_paths:
        history = torch.load(path, weights_only=False)

        train_ce.append(history['train_ce'])
        test_ce.append(history['test_ce'])
        accuracy.append(history['test_accuracy'])

    for i in range(len(train_ce)):
        train_steps = np.arange(len(train_ce[i]))
        sub_fig_training_history.plot(train_steps, train_ce[i], color=colors[0], alpha=0.4)

    for i in range(len(test_ce)):
        test_steps = np.linspace(0, len(train_ce[i]) - 1, len(test_ce[i]))
        sub_fig_training_history.plot(test_steps, test_ce[i], color=colors[1], alpha=0.6)

        sub_fig_accuracy_history.plot(test_steps, accuracy[i], color=colors[2], alpha=0.6)

    sub_fig_training_history.set_ylabel('Cross-entropy loss')
    sub_fig_training_history.set_xlabel('Epoch')
    sub_fig_training_history.set_xticks([0, 1250, 2500], [0, 50, 100])

    sub_fig_accuracy_history.set_ylabel('Test accuracy')
    sub_fig_accuracy_history.set_xlabel('Epoch')
    sub_fig_accuracy_history.set_xticks([0, 1250, 2500], [0, 50, 100])
    sub_fig_accuracy_history.set_ylim(0.0, 1.0)



def heatmap_model_output(model_preds, sub_fig_confusion_matrix, sub_fig_heatmap):
    y_true = model_preds[:, -1:]
    y_pred = np.argmax(model_preds[:, :-1], axis=1)

    colors = ['#ffffff', my_green]
    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging", colors, N=256)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=sub_fig_confusion_matrix, cmap=cmap, colorbar=False)

    # Activations heatmap
    logits = model_preds[:, :-1]
    labels = model_preds[:, -1]

    # Sort by label
    sorted_idx = labels.argsort()
    logits_sorted = logits[sorted_idx].T

    colors = ['#ffffff', my_orange]
    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging", colors, N=256)

    heatmap = sns.heatmap(logits_sorted, cmap=cmap, vmax=3.0, ax=sub_fig_heatmap, cbar_kws={'shrink': 0.5})
    sub_fig_heatmap.set_ylabel("V2 column activations")
    sub_fig_heatmap.set_xlabel("Sample (sorted by class)")
    sub_fig_heatmap.set_xticks([])
    sub_fig_heatmap.set_yticklabels(sub_fig_heatmap.get_yticklabels(), rotation=0)

    for spine in sub_fig_heatmap.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.6)

    cbar = heatmap.collections[0].colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.6)
    cbar.set_label('L2/3 firing rates (Hz)')
    cbar.set_ticks([0.0, 1.0, 2.0, 3.0])


def test_and_plot_digits(network_path, digits_to_include, seed, sub_fig_confusion_matrix, sub_fig_heatmap):
    device = torch.device('mps')
    network = BrainNetwork.load(network_path)

    # Get test set that was used during training
    _, X_test, _, y_test = prepare_ds(digits_to_include, padding=1, seed=seed)

    # Run the network on the test set again
    with torch.no_grad():
        output = network.run(X_test, device=device)

        model_predictions = network.read_out(output, mode='classification')
        heatmap_model_output(torch.concat((model_predictions.detach().cpu(), y_test.unsqueeze(1)), dim=-1).detach().numpy(),
                             sub_fig_confusion_matrix, sub_fig_heatmap)


def make_digits_results_fig(network_paths, history_paths, digits_to_include, seed):
    fig = plt.figure(figsize=(17 / 2.54, 12 / 2.54))

    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        width_ratios=[1.0, 1.0, 0.8],
        height_ratios=[0.4, 0.4, 1.0]
    )

    sub_fig_architecture = fig.add_subplot(gs[0:2, 0:2])
    sub_fig_training_history = fig.add_subplot(gs[0, 2])
    sub_fig_accuracy_history = fig.add_subplot(gs[1, 2])
    sub_fig_confusion_matrix = fig.add_subplot(gs[2, 0])
    sub_fig_heatmap = fig.add_subplot(gs[2, 1:])

    plot_training_history(history_paths, sub_fig_training_history, sub_fig_accuracy_history)

    test_and_plot_digits(network_paths[seed-1], digits_to_include, seed, sub_fig_confusion_matrix, sub_fig_heatmap)

    label_mapping = {
        '(A)': sub_fig_architecture,
        '(B)': sub_fig_training_history,
        '(C)': sub_fig_accuracy_history,
        '(D)': sub_fig_confusion_matrix,
        '(E)': sub_fig_heatmap
        }

    add_sub_fig_labels(label_mapping)

    fig.tight_layout(
        pad=1.0,
        w_pad=1.0,
        h_pad=1.0)

    finish_plot(fig, results_path('digits', 'digits_fig.pdf'))
    finish_plot(fig, results_path('digits', 'digits_fig.svg'))



if __name__ == '__main__':

    network_paths = [models_path('digits', f'digits_{i}.pt') for i in range(1, 6)]
    history_paths = [models_path('digits', f'digits_history_{i}.pt') for i in range(1, 6)]

    digits_to_include = [0,1,2,3,4,5,6,7,8,9]
    seed = 1

    make_digits_results_fig(network_paths, history_paths, digits_to_include, seed)
