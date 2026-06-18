from torchdiffeq import odeint

from src.utils import *
from parity_ode import prep_parity_stim



def plot_all_parity_cases(nr_output_cols, random_order, draw_numbers):

    if nr_output_cols == 1:
        # Load 8 bit classification trained network
        # with open('../results/_parity_old/parity_12_and_children/parity_12_only_L23_latin_no_adap_noise_2_0_only_L5_output/parity_post_training.pkl', 'rb') as f:
        #     network = pickle.load(f)
        with open('../results/parity_8bit_classification_seed1_onlyL5output_L23einhibi_pt2/parity_post_training.pkl', 'rb') as f:
            network = pickle.load(f)
    elif nr_output_cols == 2:
        # Load 4 bit trajectory trained network
        with open('../results/parity_successful_double_trajectory/parity_post_training.pkl', 'rb') as f:
            network = pickle.load(f)

    # Prepare time vector and initial state
    stim_duration = 0.5
    dt = 1e-3
    time_steps = int(stim_duration * 2 / dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)

    num_columns = sum(network.nr_columns_per_area)
    initial_state = torch.zeros(num_columns * 8 * 3)  # 3 state variables
    membrane_init = torch.tensor([-1.7997e-01, 8.3757e+00, 1.1346e+01, 1.1953e+01, -6.5426e+00, 1.0319e+01, -2.9719e+01, 1.2530e+01])
    num_populations_first_area = network.areas['0'].num_populations
    initial_state[:num_populations_first_area] = torch.tile(membrane_init, (num_populations_first_area//8,))
    initial_state = initial_state.unsqueeze(0)

    # Stimulus
    if random_order:
        stims = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 15., 15., 15., 15., 15., 15.],
                              [0., 0., 0., 0., 0., 0., 15., 15.],
                              [0., 0., 0., 15., 15., 15., 15., 15.],
                              [15., 15., 15., 15., 15., 15., 15., 15.],
                              [0., 0., 0., 0., 0., 0., 0., 15.],
                              [0., 0., 0., 0., 0., 15., 15., 15.],
                              [0., 15., 15., 15., 15., 15., 15., 15.],
                              [0., 0., 0., 0., 15., 15., 15., 15.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])
    else:
        stims = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 15.],
                              [0., 0., 0., 0., 0., 0., 15., 15.],
                              [0., 0., 0., 0., 0., 15., 15., 15.],
                              [0., 0., 0., 0., 15., 15., 15., 15.],
                              # [0., 0., 0., 15., 15., 15., 15., 15.],
                              # [0., 0., 15., 15., 15., 15., 15., 15.],
                              # [0., 15., 15., 15., 15., 15., 15., 15.],
                              # [15., 15., 15., 15., 15., 15., 15., 15.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])


    ### Run neural ODE ###
    with torch.no_grad():
        i = 0

        for stim in stims:
            stim_ode = prep_parity_stim(stim, time_vec, network.areas['0'].num_columns)
            network.stim = stim_ode

            ode_output = odeint(network, initial_state, time_vec)
            # ode_output = sdeint(network, initial_state, time_vec, names={'drift': 'forward', 'diffusion': 'diffusion'}, method='srk')

            split = network.network_as_area.num_populations
            firing_rates = compute_firing_rate(ode_output[:, :, :split] - ode_output[:, :, split:(split * 2)])

            if nr_output_cols > 1:
                final_column_1 = torch.sum((firing_rates[:, 0, -16:-8] * network.output_weights[-16:-8]) / network.output_scale, dim=-1)
            final_column_2 = torch.sum((firing_rates[:, 0, -8:] * network.output_weights[-8:]) / network.output_scale, dim=-1)

            if i == 0:
                if nr_output_cols > 1:
                    time_course_1 = final_column_1
                time_course_2 = final_column_2
                stim_time_course = stim_ode
            else:
                if nr_output_cols > 1:
                    time_course_1 = torch.concat([time_course_1, final_column_1], dim=0)
                time_course_2 = torch.concat([time_course_2, final_column_2], dim=0)
                stim_time_course = torch.concat([stim_time_course, stim_ode], dim=0)
            initial_state = ode_output[-1, :, :]
            i += 1

        # with open('../parity_timecourse_even.pkl', 'wb') as f:
        #     pickle.dump(time_course_1, f)
        # with open('../parity_timecourse_odd.pkl', 'wb') as f:
        #     pickle.dump(time_course_2, f)

    # Plotting
    plt.rcParams.update({
        'axes.titlesize': 18,  # Title size
        'axes.labelsize': 18,  # X and Y label size
        'xtick.labelsize': 16,  # X tick label size
        'ytick.labelsize': 16,  # Y tick label size
        'legend.fontsize': 12,  # Legend font size
        'font.size': 20  # Default text size
    })

    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True,
                                                 gridspec_kw={'height_ratios': [2.5, 1.5]})

    time = np.arange(len(time_course_2[1000:])) * dt

    if nr_output_cols > 1:
        axes[0].plot(time, time_course_2[1000:].detach().numpy(), label='Odd column', color='royalblue', linewidth=3)
    axes[0].plot(time, time_course_1[1000:].detach().numpy(), label='Even column', color='darkorange', linewidth=3)
    axes[0].set_title('Firing rates output column')
    axes[0].set_ylabel('Firing rate')
    if nr_output_cols > 1:
        axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    stim_time_course = torch.sum(stim_time_course[1000:], dim=1).detach().numpy()
    # stim_binary = torch.tensor([1. if i>0. else 0. for i in stim_time_course])
    stim_time_scaled = stim_time_course / 15
    axes[1].plot(time, stim_time_scaled, color='dimgrey', linewidth=3)
    axes[1].set_xlabel('Time (s)')
    # axes[1].set_yticks(np.arange(2), ['Off', 'On'])
    axes[1].set_ylim([-1, 10])
    axes[1].set_ylabel('Input')
    axes[1].set_title('Number of activated input units')
    axes[1].fill_between(time, stim_time_scaled, where=(stim_time_scaled % 2 == 0), label='Even input', color='darkorange', alpha=1.0, interpolate=True)
    axes[1].fill_between(time, stim_time_scaled, where=(stim_time_scaled % 2 != 0), label='Odd input', color='royalblue', alpha=1.0, interpolate=True)
    axes[1].legend(loc="upper right")
    axes[1].set_axisbelow(True)
    axes[1].xaxis.grid(True, linestyle='--', alpha=1.0)
    # Draw number in fig
    if draw_numbers:
        num_stimuli = 8 if nr_output_cols==1 else 4
        for i in range(num_stimuli):
            stim = int(stim_time_course[(i*1000)+670]/15)
            axes[1].text(i+0.67, stim+1.5, str(stim), color='dimgrey')

    plt.savefig('../parity_timecourse')
    plt.show()



if __name__ == '__main__':
    plot_all_parity_cases(nr_output_cols=2, random_order=False, draw_numbers=False)

    # Note: still needs to be updated after changes

