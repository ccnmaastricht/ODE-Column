import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from pprint import pprint

from sympy.printing.pretty.pretty_symbology import line_width
from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from src.utils import *
from src.column_network_wta import ColumnAreaWTA
from src.column_network_xor import ColumnNetworkXOR
from wta_ode import set_stim_three_phases
from xor_ode import init_xor, make_stim, prep_stim_ode



def wta_timecourse(fn):

    # Load network
    with open(fn, 'rb') as f:
        _network = pickle.load(f)
    weights = _network.recurrent_weights

    col_params = load_config('../config/model.toml')
    network = ColumnAreaWTA(col_params, area='mt')

    network.recurrent_weights = torch.nn.Parameter(weights, requires_grad=False)

    # Time params
    dt = 1e-4
    phase = 0.5  # secs
    time_steps = int(phase/dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)
    network.time_vec = time_vec

    # Initial state is just zeros, except membrane potential
    initial_state = torch.zeros(48).unsqueeze(0)  # 48 = 8*2*3
    initial_state[:, :16] = torch.tile(torch.tensor([-1.5554, 8.9735, 12.0712, 12.5040, -5.2554, 10.4650, -30.8225, 12.6189]), (2,))

    with torch.no_grad():
        i = 0

        for stims in [[0., 0.], [0., 0.], [0., 0.], [10., 30.], [0., 0.], [30., 10.], [0., 0.], [20., 20.], [20., 20.], [20., 20.], [20., 20.], [0., 0.]]:  #

            muA = stims[0]
            muB = stims[1]

            # Set stim and time_vec
            stim = torch.zeros(time_steps, 16)
            stim[:, [2, 3]] = muA
            stim[:, [10, 11]] = muB
            network.stim = stim
            network.time_vec = time_vec

            ode_output = sdeint(network, initial_state, time_vec,
                                names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')
            comp_fr = compute_firing_rate(ode_output[:, 0, :16] - ode_output[:, 0, 16:32])
            if i == 0:
                time_course = comp_fr
                stim_time_course = stim
            else:
                time_course = torch.concat([time_course, comp_fr], dim=0)
                stim_time_course = torch.concat([stim_time_course, stim], dim=0)
            initial_state = ode_output[-1, :, :]
            i += 1

        with open('../wta_timecourse_plot.pkl', 'wb') as f:
            pickle.dump(time_course, f)

        # with open("../wta_timecourse_plot_28-07.pkl", 'rb') as f:  # this one was used for the CCN poster!
        #     time_course = pickle.load(f)

        time_course = time_course[time_steps:]
        stim_time_course = stim_time_course[time_steps:]

        time = np.arange(time_course.shape[0]) * dt

        plt.rcParams.update({
            'axes.titlesize': 28,  # Title size
            'axes.labelsize': 24,  # X and Y label size
            'xtick.labelsize': 20,  # X tick label size
            'ytick.labelsize': 20,  # Y tick label size
            'legend.fontsize': 24,  # Legend font size
            'font.size': 24  # Default text size
        })

        # Set the figure size (wide, not too tall)
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 13), sharex=True,
                                                 gridspec_kw={'height_ratios': [2.5, 1.0]})

        ax1.plot(time, time_course[:, 0], label='Column A', color='royalblue', linewidth=3)
        ax1.plot(time, time_course[:, 8], label='Column B', color='darkorange', linewidth=3)
        ax1.set_title('L2/3e firing rates in columns A & B')
        ax1.set_ylabel('Firing Rate')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax3.plot(time, stim_time_course[:, 2], label='Input 1', color='royalblue', linewidth=7)
        ax3.plot(time, stim_time_course[:, 10], label='Input 2', color='darkorange', linewidth=7, linestyle='--')
        ax3.set_title('Inputs')
        ax3.set_xlabel('Time (s)')
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3.set_ylabel('Hz')
        ax3.set_ylim(-5.0, 40.0)
        ax1.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)

        # Layout adjustment
        plt.tight_layout()
        plt.savefig('../wta_timecourse')
        plt.close(fig)


def xor_timecourse():

    # Initialize the network
    network, initial_state, time_vec, time_steps = init_xor()

    network.feedforward_target_weights['0'] = torch.nn.ParameterList([
        torch.tensor([0.0, 0.0, 28.0, 7.2, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 26.2, 13.2, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 32.4, 12.6, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 29.2, 15.2, 0.0, 0.0, 0.0, 0.0])
    ])
    network.feedforward_target_weights['1'] = torch.nn.ParameterList([
        torch.tensor([0.0, 0.0, 31.4, 8.8, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 23.2, 18.8, 0.0, 0.0, 0.0, 0.0])
    ])

    time_course = torch.Tensor(time_steps*5, 24)  # 5 stimuli * N time steps, 24 populations
    stim_time_course = torch.Tensor(time_steps*5, 2)

    four_stims = make_stim(shuffle=False)
    five_stims = torch.concat([four_stims[3].unsqueeze(0), four_stims], dim=0)  # add an extra (0,0) to start time course

    with torch.no_grad():
        for stim_iter, stim in enumerate(five_stims):
            stim_ode = prep_stim_ode(stim, time_vec)

            network.time_vec = time_vec
            network.stim = stim_ode
            # ode_output = odeint(network, initial_state, time_vec)
            ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

            initial_state = ode_output[-1, :, :]

            # Store results
            firing_rates = compute_firing_rate(ode_output[:, :, :24] - ode_output[:, :, 24:48]).squeeze(dim=1)
            time_course[stim_iter*time_steps:(stim_iter+1)*time_steps, :] = firing_rates
            stim_time_course[stim_iter*time_steps:(stim_iter+1)*time_steps, 0] = stim_ode[:, 0, 2]  # idx2 = layer 4
            stim_time_course[stim_iter*time_steps:(stim_iter+1)*time_steps, 1] = stim_ode[:, 1, 2]

        time_course = time_course[time_steps:]  # remove first resting state period
        stim_time_course = stim_time_course[time_steps:]

        time = np.arange(time_course.shape[0]) * 1e-3

        plt.rcParams.update({
            'axes.titlesize': 28,  # Title size
            'axes.labelsize': 24,  # X and Y label size
            'xtick.labelsize': 20,  # X tick label size
            'ytick.labelsize': 20,  # Y tick label size
            'legend.fontsize': 24,  # Legend font size
            'font.size': 24  # Default text size
        })

        # Set the figure size (wide, not too tall)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16), sharex=True, gridspec_kw={'height_ratios': [2.5, 2.5, 0.75]})

        ax1.plot(time, time_course[:, 0], label='Column A', color='royalblue', linewidth=3)
        ax1.plot(time, time_course[:, 8], label='Column B', color='darkorange', linewidth=3)
        ax1.set_title('L2/3e firing rates in columns A & B')
        ax1.set_ylabel('Firing Rate')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.plot(time, time_course[:, 16], label='Column C', color='forestgreen', linewidth=3)
        ax2.plot(time, np.ones(len(time)), label='Classification label = 1', color='forestgreen', linewidth=3, linestyle='--')
        # ax2.axhline(y=1, color='forestgreen', linestyle='--', linewidth=2)
        ax2.set_title('L2/3e firing rates in column C')
        ax2.set_ylabel('Firing Rate')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        ax3.plot(time, stim_time_course[:, 0], label='Input 1', color='black', linewidth=7)
        ax3.plot(time, stim_time_course[:, 1], label='Input 2', color='grey', linewidth=7, linestyle='--')
        ax3.set_title('Inputs')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Hz')
        ax3.set_ylim(-5, 30)
        ax3.grid(True, linestyle='--', alpha=0.5)

        # Layout adjustment
        plt.tight_layout()
        plt.savefig('../xor_timecourse')
        plt.close(fig)





if __name__ == '__main__':

    fn = '../___.pkl'

    # WTA time course showing WTA and bistable perception - used for CCN poster
    wta_timecourse(fn)

    # XOR time course - used for CCN poster
    xor_timecourse()
