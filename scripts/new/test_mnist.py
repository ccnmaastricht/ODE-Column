import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

from src.brain_network import BrainNetwork
from src.utils.paths import models_path
from train_mnist import get_data



def plot_history(history_path):
    history = torch.load(history_path, weights_only=False)

    for key, values in history.items():
        print(key)
        plt.plot(values)
        plt.show()

def test_and_plot_mnist(network_path, seed):
    # Get test set used during training
    _, X_val, y_val = get_data(64, seed)

    network = BrainNetwork.load(network_path)

    with torch.no_grad():
        output = network.run(X_val, device=torch.device('mps'))

    model_predictions = network.read_out(output, mode='classification')
    model_preds = model_predictions.detach().cpu().numpy()

    # Confusion matrix
    y_pred = np.argmax(model_preds, axis=1)
    cm = confusion_matrix(y_val, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    # Activations heatmap
    sorted_idx = y_val.argsort()
    model_preds_sorted = model_preds[sorted_idx].T

    heatmap = sns.heatmap(model_preds_sorted)
    plt.show()



if __name__ == '__main__':

    network_path = models_path('mnist', f'mnist_1.pt')
    history_path = models_path('mnist', f'mnist_history_1.pt')

    # plot_history(history_path)

    test_and_plot_mnist(network_path, seed=1)
