import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from src.utils import load_config
from wwp_training import get_stim



def find_index(fr, threshold, winner_col):
    winner_idx = 0 if winner_col == 'A' else 8

    # Find indices where condition is True
    indices = (fr[:, winner_idx] > threshold).nonzero(as_tuple=True)[0]

    # Get the first one, if any
    if len(indices) > 0:
        index = indices[0].item()
        # print("First index:", index)
    else:
        index = 0
    return index

def compute_visuomotor_index(model, input_state, time_vec):
    with torch.no_grad():

        threshold = 1.0
        nr_sims = 1000

        RTs = torch.Tensor(nr_sims)
        av_frs = torch.Tensor(nr_sims, 8)
        ratios = torch.Tensor(nr_sims, 8)

        for i in range(nr_sims):

            # Determine the stimulus
            coherence = np.random.uniform(2.0, 10.0)
            muA = np.random.uniform(15.0, 25.0)
            muB = muA + np.random.choice([coherence, coherence * -1.0])

            stim = get_stim(torch.tensor([muA, muB]))
            winner = 'A' if muA > muB else 'B'

            # Run the model
            output = model.run_ode_stim_phases(input_state, stim, time_vec, 3, no_post=True)

            # Compute firing rates
            mem, adap = output[:, 0, :], output[:, 1, :]
            fr = model.compute_firing_rate_torch(mem - adap)

            # Find the threshold excedence point
            thresh_idx = find_index(fr, threshold, winner)

            # Compute reaction time
            RT = thresh_idx - 1000 + 100  # WW added 100 ms
            RTs[i] = RT

            # Compute average firing rate after 400-800 ms after threshold
            av_start = thresh_idx + 100
            av_end = thresh_idx + 200
            # print(av_start, av_end)
            av_fr = torch.mean(fr[av_start:av_end, :], dim=0)
            if winner == 'A':
                av_frs[i, :] = av_fr[:8]
            elif winner == 'B':
                av_frs[i, :] = av_fr[8:]

            # Compute ratio winner/loser
            if winner == 'A':
                ratio = torch.mean(fr[:, :8] / fr[:, 8:], dim=0)
            elif winner == 'B':
                ratio = torch.mean(fr[:, 8:] / fr[:, :8], dim=0)
            ratios[i, :] = ratio

        # Compute visuo-motor index
        for j in range(0, 8, 2):
            # print(np.corrcoef(RTs, av_frs[:, j]))
            print(np.corrcoef(RTs, [av_frs[:, j], av_frs[:, j + 1]]))

        print(torch.mean(ratios, dim=0))

def blend_with_white(c, factor=0.6):
    """Blend color c with white by a factor."""
    return (1 - factor) * np.array([1, 1, 1]) + factor * np.array(c)


if __name__ == '__main__':

    # Load model
    with open('../ww_trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    col_params = load_config('../config/model.toml')
    sim_params = load_config('../config/simulation.toml')

    # Initial state is always the same
    membrane = torch.tensor(sim_params['initial_conditions']['membrane_potential'])
    adaptation = torch.tensor(sim_params['initial_conditions']['adaptation'])
    input_state = torch.stack((membrane, adaptation))

    # Time
    time_steps = 1500
    time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)


    with torch.no_grad():

        # Run the model for different coherences (diff between input A and B)
        # coherences = [0.5, 1., 2., 4., 8., 16.]
        coherences = [0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]

        fr_results = torch.Tensor(4, len(coherences), 600, 2)

        for i, coherence in enumerate(coherences):

            # Determine the stimulus
            muA = 20.
            muB = muA - coherence
            stim = get_stim(torch.tensor([muA, muB]))

            # Run the model
            output = model.run_ode_stim_phases(input_state, stim, time_vec, 3, no_post=True)

            # Compute firing rates
            mem, adap = output[:, 0, :], output[:, 1, :]
            fr = model.compute_firing_rate_torch(mem - adap)

            fr_results[0, i, :, :] = fr[400:1000, [0, 8]]  # layer 2/3
            fr_results[1, i, :, :] = fr[400:1000, [2, 10]]  # layer 4
            fr_results[2, i, :, :] = fr[400:1000, [4, 12]]  # layer 5
            fr_results[3, i, :, :] = fr[400:1000, [6, 14]]  # layer 6



        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(7, 4))
        plt.subplots_adjust(hspace=0.5, wspace=0.2)

        colors = plt.get_cmap('rainbow', len(coherences))
        layers = ['Layer 2/3', 'Layer 4', 'Layer 5', 'Layer 6']

        # Normalize for color mapping for the colorbar
        norm = mcolors.Normalize(vmin=min(coherences), vmax=max(coherences))
        sm = cm.ScalarMappable(cmap=colors, norm=norm)
        sm.set_array([])

        # Font
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10

        for l_idx in range(4):

            if l_idx == 0:
                axes_ = axes[0, 0]
            elif l_idx == 1:
                axes_ = axes[1, 0]
            elif l_idx == 2:
                axes_ = axes[0, 1]
            elif l_idx == 3:
                axes_ = axes[1, 1]

            for c_idx in range(len(coherences)):
                color = colors(c_idx)

                axes_.plot(fr_results[l_idx, c_idx, :, 1], linestyle='--', color=color, zorder=1)
                axes_.plot(fr_results[l_idx, c_idx, :, 0], color=color, zorder=2)

            # Add y label indicating layer
            axes_.set_title(layers[l_idx])

            # Add vertical line at x=0 (stimulus)
            axes_.axvline(x=100, color='gray', linestyle='--', linewidth=0.8)

            # Remove the upper and right-most borders
            axes_.spines['top'].set_visible(False)
            axes_.spines['right'].set_visible(False)

            # Remove y-ticks
            axes_.set_yticks([])

            # Set x-ticks
            xticks = np.arange(0, 601, 100)
            if l_idx == 0 or l_idx == 2:  # Remove x-ticks for top subplots
                axes_.set_xticklabels([])
            else:  # For bottom subplots: relabel x-axis from 0–600 to -100–500
                axes_.tick_params(labelsize=8)
                xlabels = xticks - 100  # Shift labels
                axes_.set_xticklabels(xlabels)
            axes_.set_xticks(xticks)
            axes_.set_xlim(0, 600)

        fig.text(0.08, 0.5, 'Firing rates', va='center', rotation='vertical', fontsize=14)

        # Color bar
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', location='top', fraction=0.04, pad=0.1)
        cbar.set_label('Relative evidence in Hz')
        cbar.set_ticks([min(coherences), max(coherences)])
        cbar.set_ticklabels([f'{min(coherences):.2f}', f'{max(coherences):.2f}'])

        plt.show()
