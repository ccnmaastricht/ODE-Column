import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from src.brain_network import BrainNetwork
from src.utils.paths import config_path, data_path, models_path
from src.utils.set_seed import set_seed
from src.utils.loss_functions import huber_loss_wta, compute_fr_ceiling_penalty
from scripts.wta.train_wta import make_input_pairs, make_ds_ww, initialize_self_excitation_connection



def visualize_results(pred_raw, true, stim, context, network, itr, area):
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
    recurrent_weights = network.connections[f'recurrent_{area}_{area}'].W
    self_excitation_weights = network.connections[f'self_excitation_{area}_{area}'].W
    lateral_weights = network.connections[f'lateral_{area}_{area}'].W

    recurr_weights  = (recurrent_weights + self_excitation_weights + lateral_weights).detach().numpy()
    heatmap1 = axes[0, 2].imshow(recurr_weights, cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[0, 2])
    axes[0, 2].set_title("Recurrent weights")

    # Plot context weights
    context_weights_vec = network.connections[f'input_context_{area}'].W.detach().numpy()
    context_weights = np.reshape(context_weights_vec.T, (8, 8))
    heatmap1 = axes[1, 2].imshow(context_weights, cmap="viridis", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[1, 2])
    axes[1, 2].set_title("Top-down context weights")

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.15)
    plt.savefig(f'./results/context_{itr}')
    plt.close(fig)

def random_input_pair():
    """
    Make a random pair of inputs for two columns.
    """
    muA = np.random.uniform(15.0, 35.0)
    muB = muA + np.random.uniform(5, 10.)

    mu_vals = [muA, muB]
    np.random.shuffle(mu_vals)
    return mu_vals

def split_dataset(states, stims, contexts, seed, test_fraction=0.1):
    """
    Randomly partition the dataset into train and test sets.
    """
    generator = torch.Generator().manual_seed(seed)

    n = len(states)
    perm = torch.randperm(n, generator=generator)

    n_test = int(round(test_fraction * n))

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return (
        states[train_idx],
        stims[train_idx],
        contexts[train_idx],
        states[test_idx],
        stims[test_idx],
        contexts[test_idx],
    )

def get_data(batch_size, network_time_params, fn, seed):
    """
    Get the Wang-Wong dataset and extends it to apply to a four-column architecture
    with an extra context variable deciding the winner-column. Creates a random
    train/test split.
    """
    dt = network_time_params['dt']
    sim_time = network_time_params['sim_time']
    time_steps = int(round(sim_time / dt))

    raw_states, raw_stims = make_ds_ww(fn, time_steps)

    # Scale down Wang-Wong firing rates
    raw_states = raw_states / 30.

    # Create a shuffled list of irrelevant input pairs
    irrelevant_pairs = make_input_pairs()
    rng = np.random.default_rng(seed)
    rng.shuffle(irrelevant_pairs)

    # Convert data from two columns to four columns
    states = []
    stims = []
    contexts = []

    pair_counter = 0

    for i in range(len(raw_states)):

        # Get the winner and loser column trajectories
        winner_column_id = torch.argmax(raw_stims[i]).item()
        loser_column_id = 1 - winner_column_id

        winner_traj = raw_states[i, :, winner_column_id]
        loser_traj = raw_states[i, :, loser_column_id]

        # Create one sample for each context
        for context, pair in enumerate([[0, 1], [2, 3]]):
            # Initialize all four columns as losers
            sample_states = torch.stack([loser_traj] * 4, dim=-1)

            # Replace the winning column within the active pair
            sample_states[:, pair[winner_column_id]] = winner_traj
            states.append(sample_states)

            # Get the next irrelevant input pair
            irrelevant_pair = torch.tensor(
                irrelevant_pairs[pair_counter % len(irrelevant_pairs)],
                dtype=raw_stims.dtype)
            pair_counter += 1

            # Initialize all four stimuli with the irrelevant pair
            sample_stimuli = torch.cat([irrelevant_pair, irrelevant_pair])

            # Replace the context-relevant pair with the original stimuli
            sample_stimuli[pair] = raw_stims[i]
            stims.append(sample_stimuli)

            # One-hot context vector
            context_vec = torch.zeros(2)
            context_vec[context] = 5.0
            contexts.append(context_vec)

    states = torch.stack(states)
    stims = torch.stack(stims)
    contexts = torch.stack(contexts)

    train_states, train_stims, train_contexts, test_states, test_stims, test_contexts = split_dataset(states, stims, contexts, seed)
    ds = TensorDataset(train_states, train_stims, train_contexts)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_states, test_stims, test_contexts

def initialize_network(area, nr_inputs, nr_contexts):
    """
    Initialize a network for context-dependent decision-making.
    """
    config = config_path('context_params.toml')
    general_config = config_path('general_params_wta.toml')
    network = BrainNetwork.from_toml(config, general_config)

    network.add_area(area, nr_inputs)

    network.add_input_connection(area, nr_inputs, unique_id='bottom_up',
                                 receptive_field_size=1, stride=1, trainable=False, std=0.0)
    network.add_input_connection(area, nr_contexts, unique_id='context')

    network.add_lateral_connection(area, receptive_field_size=2, stride=2)
    network.add_custom_connection(connection_name='self_excitation', source=area, target=area,
                                  initializer=initialize_self_excitation_connection)
    network.add_output_connection(area)

    return network

def apply_context_weight_constraints(context_to, network, area, nr_inputs, nr_contexts):
    """
    Apply context weight constraints if specified. If 'deep_only', the context (i.e. feedback)
    weights are only target L5e, L5i, L6e, L6i. If 'superficial_only', they only target L23e, L23i.
    """
    if context_to != 'default':
        assert context_to in ['deep_only', 'superficial_only'], (f"The only acceptable options for context_to are 'default',"
                                                               f"'deep_only' and 'superficial_only'. ")

        if context_to == 'deep_only':
            constrained_mask = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.]).unsqueeze(1)
        elif context_to == 'superficial_only':
            constrained_mask = torch.tensor([1., 1., 0., 0., 0., 0., 0., 0.]).unsqueeze(1)

        with torch.no_grad():
            constrained_mask = torch.tile(constrained_mask, (nr_inputs, nr_contexts))

            network.connections[f'input_context_{area}'].weights.mul_(constrained_mask)
            network.connections[f'input_context_{area}'].mask.copy_(constrained_mask)

def train_context_wta(
        fn_target_data,
        seed,
        batch_size=32,
        lr_lateral=1e+1,
        lr_context=1e+0,
        fr_reg_lambda=1e-6,
        num_epochs=5,
        test_freq=10,
        train_with_adjoint=True,
        train_with_noise=True,
        context_to='default',
        device=torch.device('cpu')):
    """
    Train a BrainNetwork to perform context-dependent decision-making. Uses the Wang-Wong
    model activity as a training target. Set `context_to` to 'default', 'deep_only' or
    'superficial_only'.
    """
    set_seed(seed)

    # Initialize the network
    area = 'mt'
    nr_inputs = 4
    nr_contexts = 2

    network = initialize_network(area, nr_inputs, nr_contexts)
    apply_context_weight_constraints(context_to, network, area, nr_inputs, nr_contexts)

    # Prepare training data and optimizer
    train_loader, test_states, test_stims, test_contexts = get_data(
        batch_size, network.params['model']['time_params'], fn_target_data, seed)
    test_states, test_contexts = test_states.to(device), test_contexts.to(device)

    optimizer = torch.optim.RMSprop([{'params': network.connections[f'lateral_{area}_{area}'].weights, 'lr': lr_lateral},
                                     {'params': network.connections[f'self_excitation_{area}_{area}'].weights, 'lr': lr_lateral},
                                     {'params': network.connections[f'input_context_{area}'].weights, 'lr': lr_context}], alpha=0.9)

    def run_batch(stims, context, true_states, itr=None, plotting=False):
        """
        Run a batch of stimuli-context pairs through the network and computes the loss
        between the network's activity and the target activity.
        """
        output = network.run({'bottom_up': stims, 'context': context},
                             adjoint=train_with_adjoint, stochastic=train_with_noise, device=device)

        # Compute loss between predicted and true states
        pred_states = network.read_out(output, mode='trajectory')
        huber_loss = huber_loss_wta(pred_states, true_states)

        firing_rates = network.get_firing_rates(output, return_as_np_array=False)
        fr_reg = fr_reg_lambda * compute_fr_ceiling_penalty(firing_rates)
        loss = huber_loss + fr_reg

        if plotting:
            for i in range(16):
                visualize_results(pred_states[:, i], true_states[i], stims[i], context[i], network, itr + f'_{i}', area=area)

        return loss, fr_reg

    # Store losses
    train_losses = []
    train_fr_reg = []
    test_losses = []
    test_fr_reg = []

    # Start training loop
    for epoch in range(num_epochs):

        for itr, (true_states, stim_batch, context_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            loss, fr_reg = run_batch(stim_batch, context_batch, true_states.to(device))

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_fr_reg.append(fr_reg.item())

            # Test
            if itr % test_freq == 0:
                with torch.no_grad():

                    test_loss, test_fr_r = run_batch(test_stims, test_contexts, test_states, itr=f'{epoch}_{itr}', plotting=True)

                    print('Epoch {:02d} | Iter {:02d} | Train Loss {:.4f} | Test Loss {:.4f}'.format(
                        epoch, itr // test_freq, loss.item(), test_loss.item()))

                    test_losses.append(test_loss.item())
                    test_fr_reg.append(test_fr_r.item())

    # Store training history and trained network
    history = {'train_losses': train_losses,
               'train_fr_reg': train_fr_reg,
               'test_losses': test_losses,
               'test_fr_reg': test_fr_reg}

    torch.save(history, models_path('context', f'context_{context_to}_history_{seed}.pt'))
    network.save(models_path('context', f'context_{context_to}_{seed}.pt'))



if __name__ == '__main__':

    fn_target_data      = data_path('ds_wta.pt')
    num_epochs          = 3
    test_freq           = 10
    train_with_adjoint  = False
    train_with_noise    = True
    device              = torch.device('cpu')

    context_constraints = ['default', 'superficial_only', 'deep_only']

    for cc in context_constraints:
        for seed in range(1, 11):

            print(f'Context to: {cc} || Seed: {seed}')

            train_context_wta(
                fn_target_data,
                seed,
                num_epochs=num_epochs,
                test_freq=test_freq,
                train_with_adjoint=train_with_adjoint,
                train_with_noise=train_with_noise,
                context_to=cc,
                device=device)
