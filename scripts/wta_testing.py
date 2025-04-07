import pickle
import torch
import matplotlib.pyplot as plt
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
    return index


if __name__ == '__main__':

    nr_sims = 1000

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
    time_steps = 3000
    time_vec = torch.linspace(0., time_steps * sim_params['time_step'], time_steps)


    # ### Run the model ###
    #
    # with torch.no_grad():
    #
    #     threshold = 1.0
    #
    #     RTs = torch.Tensor(nr_sims)
    #     av_frs = torch.Tensor(nr_sims, 8)
    #     ratios = torch.Tensor(nr_sims, 8)
    #
    #     for i in range(nr_sims):
    #
    #         # Determine the stimulus
    #         coherence   = np.random.uniform(2.0, 10.0)
    #         muA         = np.random.uniform(15.0, 25.0)
    #         muB         = muA + np.random.choice([coherence, coherence * -1.0])
    #
    #         stim = get_stim(torch.tensor([muA, muB]))
    #         winner = 'A' if muA > muB else 'B'
    #
    #         # print(muA)
    #         # print(muB)
    #         # print(winner)
    #
    #         # Run the model
    #         output = model.run_ode_stim_phases(input_state, stim, time_vec, 3, no_post=True)
    #
    #         # Compute firing rates
    #         mem, adap = output[:, 0, :], output[:, 1, :]
    #         fr = model.compute_firing_rate_torch(mem - adap)
    #
    #         # plt.plot(fr[:,0])
    #         # plt.plot(fr[:,8])
    #         # plt.show()
    #
    #         # Find the threshold excedence point
    #         thresh_idx = find_index(fr, threshold, winner)
    #
    #         # Compute reaction time
    #         RT = thresh_idx - 1000 + 100  # WW added 100 ms
    #         RTs[i] = RT
    #
    #         # Compute average firing rate after 400-800 ms after threshold
    #         av_start = thresh_idx + 400
    #         av_end = thresh_idx + 800
    #         av_fr = torch.mean(fr[av_start:av_end, :], dim=0)
    #         if winner == 'A':
    #             av_frs[i, :] = av_fr[:8]
    #         elif winner == 'B':
    #             av_frs[i, :] = av_fr[8:]
    #
    #         # Compute ratio winner/loser
    #         if winner == 'A':
    #             ratio = torch.mean(fr[:, :8] / fr[:, 8:], dim=0)
    #         elif winner == 'B':
    #             ratio = torch.mean(fr[:, 8:] / fr[:, :8], dim=0)
    #         ratios[i, :] = ratio
    #
    #     # Compute visuo-motor index
    #     for j in range (0, 8, 2):
    #         print(np.corrcoef(RTs, av_frs[:, j]))
    #
    #     print(torch.mean(ratios, dim=0))


    plt.plot([0.6681047, 0.62423317, 0.54781625, 0.71784251])
    plt.ylim(0.5, 0.8)
    plt.ylabel('Visuomotor index')
    plt.xlabel('Cortical layer depth')
    plt.show()

    plt.plot([429.57, 0.86766, 3.1629, 4.2308])
    plt.ylabel('Ratio winning column/losing column')
    plt.xlabel('Cortical layer depth')
    plt.show()
