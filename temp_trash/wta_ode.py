import os
from scipy.linalg import block_diag

from torch.utils.data import TensorDataset, DataLoader

from torchsde import sdeint_adjoint
from torchdiffeq import odeint_adjoint

from src.utils import *
from src.ww_model import DM
from temp_trash.column_network_wta import ColumnAreaWTA


def visualize_results(pred, true, stim, network, train_loss, test_loss, weights, seed):
    '''
    Visualize the firing rates of L23e during training.
    '''
    if not os.path.exists(f'../results_old/wta_seed_{seed}'):
        os.makedirs(f'../results_old/wta_seed_{seed}')
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    fig.text(0.2, 0.03, f"Input column 1: {stim[0]:.1f}", ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    fig.text(0.4, 0.03, f"Input column 2: {stim[1]:.1f}", ha='center', fontsize=10, color='#ff7f0e', fontweight='bold')
    fig.text(0.8, 0.03, f"Validation loss: {test_loss:.2f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.6, 0.03, f"Training loss: {train_loss:.2f}", ha='center', fontsize=10, fontweight='bold')

    # Plot firing rate
    col1_pred_fr_all = pred[:, :8]
    col2_pred_fr_all = pred[:, 8:]
    col1_pred_fr = torch.sum(col1_pred_fr_all * network.output_weights, dim=-1)
    col2_pred_fr = torch.sum(col2_pred_fr_all * network.output_weights, dim=-1)

    col1_true_fr = true[:, 0]
    col2_true_fr = true[:, 1]

    axes[0].plot(col1_true_fr.cpu().numpy(), '--', label='true col 1')
    axes[0].plot(col2_true_fr.cpu().numpy(), '--', label='true col 2')
    axes[0].plot(col1_pred_fr.cpu().numpy(), label='pred col 1')
    axes[0].plot(col2_pred_fr.cpu().numpy(), label='pred col 2')
    axes[0].set_title("Firing rates in layer 2/3")

    # Plot current weights
    heatmap1 = axes[1].imshow(weights[-1], cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[1])
    axes[1].set_title("Current weights")

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig('../results_old/wta_seed_{}/{:02d}'.format(seed, len(weights)))
    plt.close(fig)

def random_input_pair():
    '''
    Makes a random pair of inputs for two columns.
    '''
    muA = np.random.uniform(15.0, 35.0)
    muB = muA + np.random.uniform(5, 10.)

    mu_vals = [muA, muB]
    np.random.shuffle(mu_vals)
    return mu_vals

def make_input_pairs():
    mu_values = np.linspace(15, 45, 100)

    pairs = []
    for a in mu_values:
        for b in mu_values:
            if 5 <= abs(a - b) <= 10:
                pairs.append([a, b])
    return pairs

def make_ds_ww(ds_file, nr_samples, time_steps):
    '''
    Make a dataset of Wang-Wong training samples. If filename
    already exists, it will load the existing dataset.
    '''
    if not os.path.exists('../data'):
        os.makedirs('../data')
    if os.path.exists(ds_file):
        with open(ds_file, 'rb') as f:
            ds = pickle.load(f)
    else:

        input_pairs = make_input_pairs()
        nr_samples = len(input_pairs)

        ds = {
            'states': torch.Tensor(nr_samples, time_steps, 2),
            'stims': torch.Tensor(nr_samples, 2)
        }

        dm = DM()  # Wang Wong model

        for i in range(nr_samples):

            # Random input
            # muA, muB = random_input_pair
            muA, muB = input_pairs[i]

            R = dm.run_sim(muA, muB)
            R = R[:, ::10]  # only take every tenth time sample
            R = R[:, :time_steps]  # lose any extra time samples

            R_t = torch.tensor(R).transpose(0, 1)
            ds['states'][i, :, :] = R_t
            ds['stims'][i, :] = torch.tensor([muA, muB])

        with open(ds_file, 'wb') as f:
            pickle.dump(ds, f)
    return ds['states'], ds['stims']

def get_data(nr_samples, batch_size, time_steps, fn):
    '''
    Gets the training dataset made with Wang-Wong model,
    scales it down to match our L23e firing rates.
    '''
    states_raw, stims_raw = make_ds_ww(fn, nr_samples, time_steps)

    # In case nr_samples is higher than nr of samples in saved file, duplicate the dataset
    if len(states_raw) >= nr_samples:
        states, stims = states_raw[:nr_samples], stims_raw[:nr_samples]
    elif len(states_raw) < nr_samples:
        nr_epochs = int(np.ceil(nr_samples / len(states_raw)))
        states, stims = torch.tile(states_raw, (nr_epochs, 1, 1)), torch.tile(stims_raw, (nr_epochs, 1))

    states = states / 30.  # scale down wang-wong firing rates to match with our L23e

    ds = TensorDataset(states, stims)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader

def set_stim_whole_column(raw_stim):
    '''
    Extent the given input stimulus to all eight populations.
    '''
    stim = torch.repeat_interleave(raw_stim, repeats=8, dim=1)
    return stim

def init_network(column_network, area, num_columns, batch_size, device):
    '''
    Initialize the one area column network, initial state and time vector.
    '''
    # Time steps for three stimulus phases (pre- and post-stimulus phase)
    dt = 1e-4
    stim_phase = 0.05
    time_steps = int((stim_phase * 3) / dt)  # add pre- and post-stimulus phase

    # Column network setup
    col_params = load_config('../config/wta_params.toml')
    network = column_network(col_params, area=area, num_columns=num_columns)

    # Initial state
    initial_state = torch.zeros(batch_size, num_columns*8*2)
    initial_state[:, :num_columns*8] = torch.tile(torch.tensor([-1.7997e-01, 8.3757e+00, 1.1346e+01, 1.1953e+01,
                                                                -6.5426e+00, 1.0319e+01, -2.9719e+01, 1.2530e+01]), (batch_size, num_columns,))

    # Time vector
    time_vec = torch.linspace(0., time_steps * dt, time_steps)
    return network.to(device), initial_state.to(device), time_vec.to(device)

def run_sample(network, time_vec, initial_state, stim_raw, device, with_noise):
    '''
    Runs one stimulus sample through the network
    '''
    stim = set_stim_whole_column(stim_raw)
    network.set_stim(stim)

    if with_noise:
        ode_output = sdeint_adjoint(network,
                            initial_state,
                            time_vec,
                            names={'drift': 'forward', 'diffusion': 'diffusion'},
                            method='srk').to(device)
    else:
        ode_output = odeint_adjoint(network,
                            initial_state,
                            time_vec).to(device)

    return ode_output

# new stuff
def get_rand_conn_matrix(network, perturb_param):
    A_2 = np.array(  # do this differently
        [[6.44611506e+03, 3.16266339e+03, 2.59783557e+03,
          6.01647220e+02, 4.65442775e+02, -0.00000000e+00,
          1.20821057e+02, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00],
         [8.76141151e+03, 2.52076805e+03, 9.05567745e+02,
          3.72758277e+02, 1.11284772e+03, -0.00000000e+00,
          6.66551844e+01, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00],
         [4.68473750e+02, 1.01159651e+02, 1.43766957e+03,
          1.02243162e+03, 9.52985102e+01, 9.34009482e-01,
          7.34173713e+02, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00],
         [4.33959810e+03, 4.96475974e+01, 2.33314168e+03,
          1.22667421e+03, 4.68583174e+01, -0.00000000e+00,
          1.76921469e+03, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00],
         [6.41241958e+03, 1.09781663e+03, 1.46141982e+03,
          4.02999089e+01, 1.22986559e+03, 1.45119002e+03,
          3.26415477e+02, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00],
         [3.41567571e+03, 4.66153650e+02, 7.34266895e+02,
          1.55271380e+01, 8.77145757e+02, 1.18139903e+03,
          1.36787175e+02, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00],
         [9.52907301e+02, 1.13201356e+02, 6.01429382e+02,
          1.18012323e+02, 8.34982317e+02, 6.19381861e+01,
          6.39900796e+02, 8.27452569e+02, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00],
         [2.24721162e+03, 1.71033333e+01, 9.60509643e+01,
          3.52604611e+00, 3.98216406e+02, 2.50041995e+01,
          1.07794130e+03, 5.05374243e+02, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00],
         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, 6.44611506e+03,
          3.16266339e+03, 2.59783557e+03, 6.01647220e+02,
          4.65442775e+02, -0.00000000e+00, 1.20821057e+02,
          -0.00000000e+00],
         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, 8.76141151e+03,
          2.52076805e+03, 9.05567745e+02, 3.72758277e+02,
          1.11284772e+03, -0.00000000e+00, 6.66551844e+01,
          -0.00000000e+00],
         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, 4.68473750e+02,
          1.01159651e+02, 1.43766957e+03, 1.02243162e+03,
          9.52985102e+01, 9.34009482e-01, 7.34173713e+02,
          -0.00000000e+00],
         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, 4.33959810e+03,
          4.96475974e+01, 2.33314168e+03, 1.22667421e+03,
          4.68583174e+01, -0.00000000e+00, 1.76921469e+03,
          -0.00000000e+00],
         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, 6.41241958e+03,
          1.09781663e+03, 1.46141982e+03, 4.02999089e+01,
          1.22986559e+03, 1.45119002e+03, 3.26415477e+02,
          -0.00000000e+00],
         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, 3.41567571e+03,
          4.66153650e+02, 7.34266895e+02, 1.55271380e+01,
          8.77145757e+02, 1.18139903e+03, 1.36787175e+02,
          -0.00000000e+00],
         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, 9.52907301e+02,
          1.13201356e+02, 6.01429382e+02, 1.18012323e+02,
          8.34982317e+02, 6.19381861e+01, 6.39900796e+02,
          8.27452569e+02],
         [-0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -0.00000000e+00, 2.24721162e+03,
          1.71033333e+01, 9.60509643e+01, 3.52604611e+00,
          3.98216406e+02, 2.50041995e+01, 1.07794130e+03,
          5.05374243e+02]])

    A_raw = A_2[:8, :8]
    A_diag = np.diag(np.diag(A_raw))
    A = A_raw - A_diag

    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)

    rng = np.random.default_rng()
    B = A * (1 + perturb_param * rng.normal(size=A.shape))
    B = np.clip(B, 1e-12, None)

    # Sinkhorn algorithm
    for _ in range(5000):
        B *= row_sums[:, None] / B.sum(axis=1, keepdims=True)
        B *= col_sums[None, :] / B.sum(axis=0, keepdims=True)

    # print(np.allclose(B.sum(axis=1), row_sums))
    # print(B.sum(axis=1) - row_sums)
    # print(np.allclose(B.sum(axis=0), col_sums))
    # print(B.sum(axis=0) - col_sums)
    #
    # print(np.sum(A != B))
    # print(np.sum(abs(A - B)))
    #
    # fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    # heatmap1 = axes[0].imshow(A + A_diag, cmap="viridis", interpolation="nearest")
    # heatmap2 = axes[1].imshow(B + A_diag, cmap="viridis", interpolation="nearest")
    # fig.colorbar(heatmap2, ax=axes[1])
    # plt.show()

    # Extent 8x8 connections to 16x16, i.e. two columns

    # B = np.array([[6.44611506e+03, 2.97641736e+03, 2.71606571e+03, 6.18350231e+02,
    #     5.11499648e+02, 9.73190684e-13, 1.26077066e+02, 8.11417735e-13],
    #    [8.82406026e+03, 2.52076805e+03, 8.99901238e+02, 3.66855765e+02,
    #     1.05762360e+03, 1.01762814e-12, 7.07995698e+01, 8.48468378e-13],
    #    [5.55731312e+02, 1.11906834e+02, 1.43766957e+03, 9.59527651e+02,
    #     1.01605231e+02, 1.22384794e+00, 6.92476378e+02, 8.78980871e-13],
    #    [4.70560130e+03, 4.35943444e+01, 2.09188067e+03, 1.22667421e+03,
    #     4.10005724e+01, 8.58419423e-13, 1.65638350e+03, 7.15724837e-13],
    #    [6.28890331e+03, 1.20702163e+03, 1.48410838e+03, 4.49314429e+01,
    #     1.22986559e+03, 1.46099534e+03, 3.03601330e+02, 7.87490395e-13],
    #    [3.14419829e+03, 5.40355095e+02, 8.32140386e+02, 1.84464212e+01,
    #     9.48459692e+02, 1.18139903e+03, 1.61956446e+02, 8.55011479e-13],
    #    [9.91090311e+02, 1.12801194e+02, 6.08519556e+02, 1.62759744e+02,
    #     7.54152810e+02, 5.31472514e+01, 6.39900796e+02, 8.27452569e+02],
    #    [2.08811279e+03, 1.56491584e+01, 9.70961186e+01, 3.33127755e+00,
    #     4.16450246e+02, 2.36999731e+01, 1.22071430e+03, 5.05374243e+02]])

    # B = np.array([[6.44611506e+03, 3.55632735e+03, 3.39208266e+03, 2.47700157e-12,
    #     7.59734679e-12, 5.87835630e-13, 1.12918823e-12, 4.16227410e-12],
    #    [9.67180023e+03, 2.52076805e+03, 3.23829322e+01, 1.84109465e+01,
    #     1.47670778e+03, 7.76388650e-14, 1.99385509e+01, 5.49735710e-13],
    #    [2.39997456e-12, 7.72002461e+00, 1.43766957e+03, 1.61403088e+03,
    #     8.00227691e+02, 4.92655204e-01, 6.76664488e-13, 2.49423702e-12],
    #    [7.30321552e+03, 1.78659568e-13, 3.53733743e-13, 1.22667421e+03,
    #     1.87269563e+02, 1.93127837e-13, 1.04797530e+03, 1.36747579e-12],
    #    [1.10423169e-11, 1.31331835e+03, 4.51414625e+03, 4.52784184e+02,
    #     1.22986559e+03, 1.52543293e+03, 2.98387972e+03, 1.14760198e-11],
    #    [5.12035656e+03, 1.16395798e+02, 3.77801930e+02, 1.77424790e+00,
    #     1.55880761e-12, 1.18139903e+03, 2.92277869e+01, 8.54006633e-13],
    #    [1.43618124e+03, 1.08889465e+01, 3.98663767e+02, 8.69205554e+01,
    #     7.39424219e+02, 1.03921358e+01, 6.39900796e+02, 8.27452569e+02],
    #    [3.06614401e+03, 3.09514283e+00, 1.46345152e+01, 2.81716805e-01,
    #     6.27162553e+02, 2.74869223e+00, 1.50987234e+02, 5.05374243e+02]])

    # B = np.array([[6.44611506e+03, 1.83233516e+03, 2.64909095e+03, 9.16192284e+02,
    #   1.49082499e+03, 3.05662146e-13, 1.36901289e+01, 4.62765008e+01],
    #  [9.49922455e+03, 2.52076805e+03, 8.91547509e+02, 2.70836915e+02,
    #   4.84473211e+02, 4.83219540e-13, 5.82502210e-13, 7.31582556e+01],
    #  [4.40337592e+02, 1.71450472e+02, 1.43766957e+03, 3.11171003e-12,
    #   6.15722460e+02, 2.95549951e+00, 1.03325586e+03, 1.58749369e+02],
    #  [3.48238594e+03, 1.19049178e+02, 2.53090653e+03, 1.22667421e+03,
    #   1.64546647e+02, 6.08197564e-13, 2.14949247e+03, 9.20796225e+01],
    #  [6.78151147e+03, 1.32936210e+03, 1.00850797e+03, 4.64305917e+01,
    #   1.22986559e+03, 1.46350935e+03, 1.17534067e+02, 4.27058884e+01],
    #  [3.42267229e+03, 7.95385746e+02, 1.14198146e+03, 3.21449261e+00,
    #   1.62804692e+01, 1.18139903e+03, 1.71222817e+02, 9.47990521e+01],
    #  [1.50883936e+03, 6.65342359e+02, 1.14123072e+02, 9.27996612e+02,
    #   5.78272524e-12, 5.79853806e+01, 6.39900796e+02, 2.35636651e+02],
    #  [1.46272638e+03, 9.48205958e+01, 3.93554568e+02, 9.53163816e+00,
    #   1.05894402e+03, 1.46161821e+01, 7.46813255e+02, 5.89421473e+02]])

    B += A_diag
    blocks = [B] * 2  # *2 columns
    rand_recurr_synapse_counts = block_diag(*blocks)

    # Multiply with synapse strength
    rand_recurr_synapse_counts = torch.tensor(rand_recurr_synapse_counts, dtype=torch.float32)
    rand_recurr_weights = rand_recurr_synapse_counts * network.recurrent_synaptic_strength
    return rand_recurr_weights

def compute_pd_deviation_penalty(network, pd_original_connectivity):
    '''
    Computes the mean absolute error between the original Potjans and
    Diesmann connectivity profile and the current connectivity profile.
    '''
    curr_internal_connections = network.recurrent_weights
    abs_diff = abs(curr_internal_connections - pd_original_connectivity)
    deviation_penalty = torch.mean(abs_diff)

    return deviation_penalty

def train_wta(nr_samples,
              batch_size,
              fn,
              device,
              seed,
              with_noise=True,
              adjust_pd=False,
              scrambled_pd=False):
    '''
    Learn the lateral connections between two cortical columns using
    data from Wang-Wong (WTA dynamics) as a training target.
    '''
    # Initialize network, initial state and time vector
    network, initial_state, time_vec = init_network(ColumnAreaWTA, 'mt', 2, batch_size, device)
    time_steps = len(time_vec)
    network.set_time_vec(time_vec)

    # Get the train and test data
    data_loader = get_data(nr_samples, batch_size, time_steps, fn)

    pd_original_connectivity = network.recurrent_weights.clone().detach()

    if scrambled_pd:
        # Re-set the recurrent connections to scrambled version
        rand_recurr_weights = get_rand_conn_matrix(network, perturb_param=5e-1) # 1e-1 tiny difference; 1e+0 noticable difference
        print(torch.mean(abs(pd_original_connectivity - rand_recurr_weights)))
        network.recurrent_weights = rand_recurr_weights

    if adjust_pd:
        # Set column-intrinsic connectivity as learnable parameter
        network.initialize_recurrent_weights()
        optimizer = torch.optim.RMSprop([{'params': network.lat_in_weights, 'lr': 10.0},
                                         {'params': network.recurrent_weights, 'lr': 1.0}], alpha=0.9)
    else:
        optimizer = torch.optim.RMSprop([network.lat_in_weights], lr=10.0, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # higher gamma = slower decay

    # Store weights for visualization
    weights = []

    # bias = 0

    for iter, (true_states, stim_batch) in enumerate(data_loader):
        optimizer.zero_grad()
        network.constrain_recurr_weights()
        true_states = true_states.to(device)

        # bias_stim = 0
        # for i in stim_batch:
        #     if i[0] > i[1]:
        #         bias_stim += 1
        # print(bias_stim / len(stim_batch))
        #
        # mean_stims = torch.mean(stim_batch, dim=0)
        # if mean_stims[0] - mean_stims[1] > 0:
        #     bias += 1
        #     stop = 0

        pred_states = run_sample(network, time_vec, initial_state[:-1], stim_batch[:-1], device, with_noise)

        # Compute loss between pred and true
        firing_rates = compute_firing_rate(pred_states[:, :, :16] - pred_states[:, :, 16:32])
        hub_loss = huber_loss_wta(firing_rates, true_states[:-1], network)
        loss = hub_loss

        if adjust_pd:
            penalty = compute_pd_deviation_penalty(network, pd_original_connectivity)
            # loss += (penalty * 0.1)  # penalty weight!
            # print(loss.item())
            # if iter == 188:
            #     print(penalty.item())

        print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))

        loss.backward()
        optimizer.step()
        if iter > (nr_samples / batch_size) / 2:
            scheduler.step()

        # Validate network and visualize results_old
        with torch.no_grad():
            # Save current weights
            network.constrain_recurr_weights()
            curr_weights = network.W.detach().cpu().numpy()
            # weights.append(curr_weights - pd_original_connectivity.cpu().numpy())
            weights.append(curr_weights)

            # Run test sample
            pred_state = run_sample(network, time_vec, initial_state[-1].unsqueeze(0), stim_batch[-1].unsqueeze(0), device, with_noise)

            # Visualize final test sample
            test_state = true_states[-1]
            pred_state_fr = compute_firing_rate(pred_state[:, 0, :16] - pred_state[:, 0, 16:32])
            test_loss = huber_loss_wta(pred_state_fr.unsqueeze(1), test_state.unsqueeze(0), network)
            visualize_results(pred_state_fr, test_state, stim_batch[-1], network, loss.item(), test_loss, weights, seed)

    # print(bias/iter)

    return network




if __name__ == '__main__':

    for seed in range (1, 11, 1):

        device = torch.device('cpu')
        # ds_target = '../data/ds_wta_6000_15_35_5_10.pkl'
        ds_target = '../data/ds_wta_NEW.pkl'

        network = train_wta(nr_samples=12000,
                            batch_size=32,
                            fn=ds_target,
                            device=device,
                            seed=seed,
                            with_noise=True,
                            adjust_pd=False,
                            scrambled_pd=False,
                            )

        save_pkl_file(f'../trained_wta_models/wta_seed_{seed}.pkl', network)

# Note: shuffle=TRUE for dataloader