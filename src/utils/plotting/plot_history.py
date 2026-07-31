import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_training_history(history_paths, time_unit="Epoch"):
    """
    Load saved training checkpoint history files and plot training and test loss
    trajectories across multiple runs using Matplotlib.

    Args:
        history_paths (list[str | Path]): List of file paths to saved history `.pt`/checkpoint
            files.
        time_unit (str, optional): Label for the horizontal time axis. Defaults to "Epoch".
    """
    train_histories = []
    test_histories = []

    # Load histories
    for path in history_paths:
        history = torch.load(path, weights_only=False)

        train_histories.append(np.asarray(history["train_losses"]))
        test_histories.append(np.asarray(history["test_losses"]))

    # Ensure equal number of epochs / time units
    train = np.stack(train_histories)
    test = np.stack(test_histories)

    epochs = np.arange(train.shape[1])

    train_mean = train.mean(axis=0)
    test_mean = test.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(4, 4), sharex=True, constrained_layout=True)

    # Training loss
    ax = axes[0]

    for run in train:
        ax.plot(epochs, run, color="C0", alpha=0.15, linewidth=1)

    ax.plot(epochs, train_mean, color="C0", linewidth=3, label="Mean")

    ax.set_title("Training loss")
    ax.set_xlabel(time_unit)
    ax.set_ylabel("Loss")

    # Test loss
    ax = axes[1]

    for run in test:
        ax.plot(epochs, run, color="C1", alpha=0.15, linewidth=1)

    ax.plot(epochs, test_mean, color="C1", linewidth=3, label="Mean")

    ax.set_title("Test loss")
    ax.set_xlabel(time_unit)
    ax.set_ylabel("Loss")

    plt.show()
