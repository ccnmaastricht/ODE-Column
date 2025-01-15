from DMF import *
from DMF_single_column import *
import matplotlib.pyplot as plt
import seaborn as sns


# Run DMF single column; DMF double columns
single = False
double = False


# Visualization
def heatmap(mtx):
    sns.heatmap(mtx, linewidth=0.5)
    plt.show()

def plot_firing_rates(R):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(R[0], label='excitatory')
    axes[0, 0].plot(R[1], label='inhibitory')
    axes[0, 0].set_title("Layer 2/3")
    axes[0, 0].legend()

    axes[0, 1].plot(R[2])
    axes[0, 1].plot(R[3])
    axes[0, 1].set_title("Layer 4")

    axes[1, 0].plot(R[4])
    axes[1, 0].plot(R[5])
    axes[1, 0].set_title("Layer 5")

    axes[1, 1].plot(R[6])
    axes[1, 1].plot(R[7])
    axes[1, 1].set_title("Layer 6")

    plt.show()



if __name__ == '__main__':

    column_model = SingleColumnDMF(area='MT')
    stim = ...
    firing_rates = column_model.simulate(stim)



    # Run the original single and/or double DMF
    if single:

        # Initialize the parameters
        params = get_params_single(J_local=0.13, area='MT')

        # Intialize the starting state (all zeros?)
        state = {}
        M = params['M']   # number of populations (=8)
        state['I'] = np.zeros(M)    # input current
        state['A'] = np.zeros(M)    # adaptation
        state['H'] = np.zeros(M)    # membrane potential
        state['R'] = np.zeros(M)    # rate
        state['N'] = np.zeros(M)    # noise

        # Initialize the stimulation
        stim = np.zeros(M)
        stim = set_vis_single(stim, column='H', nu=20.0, params=params)  # let's go with the horizontal column


        # Total time steps
        T = 1000

        # Array for saving firing rate
        R = np.zeros((M, T))

        # Run simulation
        # note: stim does not change for the entirety of the simulation
        for t in range(T):
            state = update_single(state, params, stim)
            R[:, t] = state['H']

        # Plot the firing rate for each layer
        #plot_firing_rates(R)

        # Plot the firing rate trajectories in a heatmap
        plt.figure(figsize=(12, 6))  # Set the figure size
        plt.imshow(R, aspect='auto', cmap='viridis', interpolation='nearest')

        # Add a colorbar
        plt.colorbar(label="Value")

        # Add labels
        plt.title("Firing rates over time")
        plt.xlabel("Time")
        plt.ylabel("Layers")

        # Show the plot
        plt.show()


    if double:

        # Initialize the parameters
        params = get_params(J_local=0.13, J_lateral=0.172, area='MT')

        # Intialize the starting state (all zeros?)
        state = {}
        M = params['M']  # number of populations (=16)
        state['I'] = np.zeros(M)  # input current
        state['A'] = np.zeros(M)  # adaptation
        state['H'] = np.zeros(M)  # membrane potential
        state['R'] = np.zeros(M)  # rate
        state['N'] = np.zeros(M)  # noise

        # Initialize the stimulation
        stim = np.zeros(M)
        stim = set_vis(stim, column='H', nu=20.0, params=params)  # for horizontal column
        # stim = set_vis(stim, column='V', nu=20.0, params=params)  # for vertical column

        # Total time steps
        T = 1000

        # Array for saving firing rate
        R = np.zeros((M, T))

        # Run simulation
        # note: stim does not change for the entirety of the simulation
        for t in range(T):
            state = update(state, params, stim)
            R[:, t] = state['R']

        # Plot the firing rate for each layer
        # plot_firing_rates(R[:8])
        # plot_firing_rates(R[8:])

        # Plot the firing rate trajectories in a heatmap
        plt.figure(figsize=(12, 6))  # Set the figure size
        plt.imshow(R, aspect='auto', cmap='viridis', interpolation='nearest')

        # Add a colorbar
        plt.colorbar(label="Value")

        # Add labels
        plt.title("Firing rates over time")
        plt.xlabel("Time")
        plt.ylabel("Layers")

        # Show the plot
        plt.show()
