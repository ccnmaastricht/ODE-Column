import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.brain_network import BrainNetwork
from src.utils.paths import models_path
from train_digits import prepare_ds



def plot_history_measures(history_path):
    history = torch.load(history_path, weights_only=False)

    for measure, values in history.items():
        plt.plot(values)
        plt.title(measure)
        plt.show()

def heatmap_model_output(model_preds):
    y_true = model_preds[:, -1:]
    y_pred = np.argmax(model_preds[:, :-1], axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    ConfusionMatrixDisplay(cm).plot(cmap="magma")
    plt.title("Confusion Matrix")
    plt.show()

    # Activations heatmap
    logits = model_preds[:, :-1]
    labels = model_preds[:, -1]

    # Sort by label
    sorted_idx = labels.argsort()
    logits_sorted = logits[sorted_idx]

    plt.figure(figsize=(5, 10))
    sns.heatmap(logits_sorted, cmap="magma", vmax=20.0)
    plt.xlabel("Class")
    plt.ylabel("Sample (sorted by label)")
    plt.show()

def test_digits_network(network_path, digits_to_include, seed):
    network = BrainNetwork.load(network_path)

    # Get test set that was used during training
    _, X_test, _, y_test = prepare_ds(digits_to_include, padding=1, seed=seed)

    # Run the network on the test set again
    with torch.no_grad():
        output = network.run(X_test)
        model_predictions = network.read_out(output, mode='classification')

        heatmap_model_output(torch.concat((model_predictions, y_test.unsqueeze(1)), dim=-1).detach().numpy())



if __name__ == '__main__':

    network_path = models_path('digits', f'digits_1.pt')
    history_path = models_path('digits', f'digits_history_1.pt')

    digits_to_include = [0, 1]
    seed = 1

    test_digits_network(network_path, digits_to_include, seed)

    # plot_history_measures(history_path)
