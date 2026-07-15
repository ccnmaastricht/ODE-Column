import torch
import numpy as np
import matplotlib.pyplot as plt

from src.brain_network import BrainNetwork
from src.utils.paths import data_path, models_path
from src.utils.plotting.plot_history import plot_training_history
from scripts.context_wta.train_context_wta import get_data



def visualize_results(pred_raw, true, stim, context, network, area='mt'):
    """
    Visualize the firing rates of L23e and the weights during training.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes_indices = [(0,0), (0,1), (1,0), (1,1)]

    # Plot firing rates
    pred = pred_raw.detach().numpy()

    for i in range(4):
        axes[axes_indices[i]].plot(pred[:, i], label='pred')
        axes[axes_indices[i]].plot(true[:, i], '--', label='true')
        axes[axes_indices[i]].set_title(f'FR column {i}, input={stim[i].item():.1f}, context={context[0 if i < 2 else 1].item():.1f}')
        axes[axes_indices[i]].set_ylim(0.0, 1.5)
    axes[axes_indices[0]].legend()

    # Plot recurrent + self-excitation + lateral inhibition weights
    recurrent_weights = network.connections[f'recurrent_{area}_{area}'].weights
    self_excitation_weights = network.connections[f'self_excitation_{area}_{area}'].weights
    lateral_weights = network.connections[f'lateral_{area}_{area}'].weights

    recurr_weights  = (recurrent_weights + self_excitation_weights + lateral_weights).detach().numpy()
    heatmap1 = axes[0, 2].imshow(recurr_weights, cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[0, 2])
    axes[0, 2].set_title("Recurrent weights")

    # Plot context weights
    context_weights_vec = network.connections[f'input_context_{area}'].weights.detach().numpy()
    context_weights = np.reshape(context_weights_vec.T, (8, 8))
    heatmap1 = axes[1, 2].imshow(context_weights, cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[1, 2])
    axes[1, 2].set_title("Top-down context weights")

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.show()

def test_context_network(network_path, seed):
    network = BrainNetwork.load(network_path)

    # Get test set that was used during training
    _, test_states, test_stims, test_contexts = get_data(32, network.params['model']['time_params'],
                                                         data_path('ds_wta.pt'), seed)

    # Run the network on the test set again
    with torch.no_grad():
        resting_state = network.run({'bottom_up': np.zeros_like(test_stims), 'context': np.zeros_like(test_contexts)})
        output = network.run({'bottom_up': test_stims, 'context': test_contexts}, stochastic=True, reset_state=False)
        model_predictions = network.read_out(output, mode='trajectory')

        for i in range(len(test_stims)):
            visualize_results(model_predictions[:, i], test_states[i], test_stims[i], test_contexts[i], network)



if __name__ == '__main__':

    network_path = models_path('context', f'context_1.pt')
    history_path = models_path('context', f'context_history_1.pt')

    seed = 1

    test_context_network(network_path, seed)
    # plot_training_history([history_path])
