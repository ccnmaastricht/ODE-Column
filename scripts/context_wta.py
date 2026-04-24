import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

from torchsde import sdeint, sdeint_adjoint
from torchdiffeq import odeint, odeint_adjoint

from wta_ode import init_network, make_ds_ww, random_input_pair, set_stim_whole_column
from src.utils import *
from src.column_network_wta import ColumnAreaContextWTA





def visualize_results(pred_raw, true, stim, context, network, train_loss, test_loss, iter, seed):
    '''
    Visualize the firing rates of L23e and the weights during training.
    '''
    if not os.path.exists(f'../results/context_seed_{seed}'):
        os.makedirs(f'../results/context_seed_{seed}')
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes_indices = [(0,0), (0,1), (1,0), (1,1)]

    # Plot firing rates
    pred = pred_raw.detach().numpy()

    for i in range(4):
        axes[axes_indices[i]].plot(pred[:, i*8], label='pred')
        axes[axes_indices[i]].plot(true[:, i], '--', label='true')
        axes[axes_indices[i]].set_title(f'FR column {i}, input={stim[i].item():.1f}, context={context[0 if i < 2 else 1].item():.1f}')
        axes[axes_indices[i]].set_ylim(0.0, 1.5)
    axes[axes_indices[0]].legend()

    # Plot recurrent + lateral inhibition weights
    recurr_weights = (network.lat_in_weights + network.recurrent_weights).detach().numpy()
    heatmap1 = axes[0, 2].imshow(recurr_weights, cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[0, 2])
    axes[0, 2].set_title("Recurrent weights")

    # Plot feedback weights
    fb_weights_vec = network.feedback_weights.detach().numpy()
    fb_weights = np.reshape(fb_weights_vec, (8, 8))
    heatmap1 = axes[1, 2].imshow(fb_weights, cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[1, 2])
    axes[1, 2].set_title("Feedback weights")

    # Loss in text
    fig.text(0.6, 0.03, f"Validation loss: {test_loss:.3f}", ha='center', fontsize=10, fontweight='bold')
    fig.text(0.4, 0.03, f"Training loss: {train_loss:.3f}", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig(f'../results/context_seed_{seed}/{iter:02d}')
    plt.close(fig)

def get_data(nr_samples, batch_size, time_steps, fn):
    '''
    Gets the training dataset made with Wang-Wong model.
    '''
    raw_states, raw_stims = make_ds_ww(fn, nr_samples, time_steps)

    # In case nr_samples is higher than nr of samples in saved file, duplicate the dataset
    if len(raw_states) >= nr_samples:
        raw_states, raw_stims = raw_states[:nr_samples], raw_stims[:nr_samples]
    elif len(raw_states) < nr_samples:
        nr_epochs = int(np.ceil(nr_samples / len(raw_states)))
        raw_states, raw_stims = torch.tile(raw_states, (nr_epochs, 1, 1)), torch.tile(raw_stims, (nr_epochs, 1))

    raw_states = raw_states / 30.  # scale down wang-wong firing rates to match with our L23

    states = []
    stims = []
    contexts = []

    for i in range(nr_samples):

        # Get the winner and loser column trajectories
        winner_column_id = torch.argmax(raw_stims[i]).item()
        winner_traj = raw_states[i, :, winner_column_id]

        loser_column_id = abs(winner_column_id - 1)
        loser_traj = raw_states[i, :, loser_column_id]

        # Choose context: 0 = (A,B), 1 = (C,D)
        context = torch.randint(0, 2, (1,)).item()
        if context == 0:
            pair = [0, 1]  # columns A, B
        elif context == 1:
            pair = [2, 3]  # columns C, D

        # Store states
        sample_states = torch.stack([loser_traj] * 4, dim=-1)  # initialize all 4 columns as losers
        sample_states[:, pair[winner_column_id]] = winner_traj  # assign winner trajectory
        states.append(sample_states)

        # Store stimuli
        sample_stimuli = torch.concat([torch.tensor(random_input_pair())] * 2)  # initialize all stimuli as new random pairs
        sample_stimuli[pair] = raw_stims[i]  # assign the context-relevant stimuli
        stims.append(sample_stimuli)

        # Context encoding (one-hot)
        context_vec = torch.zeros(2)
        context_vec[context] = 5.0
        # context_vec = torch.repeat_interleave(context_vec, repeats=2)
        contexts.append(context_vec)

    states = torch.stack(states)
    stims = torch.stack(stims)
    contexts = torch.stack(contexts)

    # Store all data in a dataloader
    ds = TensorDataset(states, stims, contexts)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader

def run_sample(network, time_vec, initial_state, stim_raw, context_raw, with_noise):
    '''
    Runs one stimulus sample through the network
    '''
    stim    = set_stim_whole_column(stim_raw)
    context = context_raw.unsqueeze(-1).expand(-1, -1, network.num_populations)  # set 2D context vector for all populations
    network.set_stim_and_context(stim, context)

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

def train_context_wta(nr_samples,
                      batch_size,
                      fn,
                      device,
                      seed,
                      with_noise=True):
    '''
    description
    '''
    # Initialize network, initial state and time vector
    network, initial_state, time_vec = init_network(ColumnAreaContextWTA, 4, batch_size, device)
    time_steps = len(time_vec)
    network.set_time_vec(time_vec)

    # Get the train and test data
    data_loader = get_data(nr_samples, batch_size, time_steps, fn)

    optimizer = torch.optim.RMSprop([{'params': network.lat_in_weights, 'lr': 10.0},
                                     {'params': network.fb_weights, 'lr': 1.0}], alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # higher gamma = slower decay

    for iter, (true_states, stim_batch, context_batch) in enumerate(data_loader):
        optimizer.zero_grad()
        network.constrain_weights()
        true_states = true_states.to(device)

        pred_states = run_sample(network,
                                 time_vec,
                                 initial_state[:-1],
                                 stim_batch[:-1],
                                 context_batch[:-1],
                                 with_noise)

        # Compute loss between pred and true
        firing_rates = compute_firing_rate(pred_states[:, :, :32] - pred_states[:, :, 32:])
        hub_loss = huber_loss_wta(firing_rates, true_states[:-1], network)

        # Add regularization term
        fb_reg = 5e-6 * (network.feedback_weights ** 2).mean()  # L2 on feedback weights
        fb_reg += 1e-6 * (network.lat_in_weights ** 2).mean()  # L2 on lateral weights
        # fb_reg = 1e-8 * (network.feedback_weights.sum(dim=1) ** 2).mean()  # column-wise constraints
        # fb_reg = 1e-3 * (firing_rates ** 2).mean()  # L2 on firing rates
        loss = hub_loss + fb_reg

        print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))

        loss.backward()

        # if iter > 180:
        #
        #     for i in range(10):
        #         plt.plot(firing_rates[:, i, :].detach().numpy())
        #         plt.show()
        #
        #     # for i in range(firing_rates.shape[1]):
        #     #     print(i, torch.max(firing_rates[:, i, :]).item())
        #
        #     idx = torch.argmax(firing_rates)
        #     coords = torch.unravel_index(idx, firing_rates.shape)
        #     print(coords, torch.max(firing_rates).item())
        #
        #     for name, param in network.named_parameters():
        #         if param.requires_grad:
        #             print(name, torch.max(param.grad))

        optimizer.step()
        # scheduler.step()

        # Validate network and visualize results
        with torch.no_grad():
            network.constrain_weights()

            # Run test sample
            pred_state = run_sample(network,
                                    time_vec,
                                    initial_state[-1].unsqueeze(0),
                                    stim_batch[-1].unsqueeze(0),
                                    context_batch[-1].unsqueeze(0),
                                    with_noise)

            # Visualize final test sample
            test_state = true_states[-1]
            pred_state_fr = compute_firing_rate(pred_state[:, 0, :32] - pred_state[:, 0, 32:])
            test_loss = huber_loss_wta(pred_state_fr.unsqueeze(1), test_state.unsqueeze(0), network)
            visualize_results(pred_state_fr, test_state, stim_batch[-1], context_batch[-1], network, loss.item(), test_loss.item(), iter, seed)

    return network



if __name__ == '__main__':

    # for seed in range(1, 11, 1):
    seed = 2
    set_seed(seed)
    device = torch.device('cpu')
    ds_target = '../data/ds_wta_NEW.pkl'

    network = train_context_wta(nr_samples=12000,
                                batch_size=32,
                                fn=ds_target,
                                device=device,
                                seed=seed,
                                with_noise=True
                                )

    save_pkl_file(f'../trained_wta_models/wta_model_seed_{seed}.pkl', network)

