import torch
import math

# TODO: remove unused loss functions


def huber_loss_wta(pred_states, true):
    """
    Compute Huber loss (smooth L1 loss) between predicted state trajectories and
    target trajectories for winner-take-all dynamics.

    Args:
        pred_states (torch.Tensor): Model predicted state trajectories, shape
            `(time_steps, batch_size, state_dim)`.
        true (torch.Tensor): Simulated ground truth trajectory tensor, shape
            `(batch_size, time_steps, state_dim)`.

    Returns:
        torch.Tensor: Computed Smooth L1 loss scalar.
    """
    true_reshaped = true.transpose(0, 1).contiguous()

    # Compute loss between model prediction and target data
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)
    return hub_loss(pred_states, true_reshaped)

def compute_suppression_penalty(model_predictions, labels, num_classes):
    """
    Compute contrastive suppression penalty by averaging model activations for non-target
    class labels to facilitate contrastive learning.

    Args:
        model_predictions (torch.Tensor): Model prediction outputs, shape
            `(batch_size, num_classes)`.
        labels (torch.Tensor): Class label indices, shape `(batch_size,)`.
        num_classes (int): Total number of output classification categories.

    Returns:
        torch.Tensor: Computed mean suppression penalty scalar.
    """
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    suppression = ((1 - one_hot_labels) * model_predictions).mean()
    return suppression

def compute_L2_regularization(network):
    """
    Compute L2 weight regularization penalty over all trainable connection parameters
    in the network.

    Args:
        network (BrainNetwork): Network module containing trainable connections.

    Returns:
        torch.Tensor: Average squared weight regularization penalty scalar.
    """
    total_sq_sum = 0.0
    total_count = 0

    for _, conn in network.connections.items():
        if not conn.trainable:
            continue
        total_sq_sum = total_sq_sum + (conn.W ** 2).sum()
        total_count += conn.W.numel()

    return total_sq_sum / total_count

def compute_fr_volatility_penalty(rates, dt=1.0, max_mean_sq_d=0.1):
    """
    Compute penalty for firing rate trajectories with rate-of-change volatility
    exceeding a maximum mean squared derivative threshold.

    Args:
        rates (torch.Tensor): Firing rate trajectory tensor, shape
            `(time_steps, batch_size, total_populations)`.
        dt (float, optional): Integration time step size in seconds. Defaults to 1.0.
        max_mean_sq_d (float, optional): Maximum allowed mean squared derivative
            threshold. Defaults to 0.1.

    Returns:
        torch.Tensor: Firing rate volatility penalty scalar.
    """
    d_rate = (rates[1:] - rates[:-1]) / dt
    mean_sq_deriv = d_rate.pow(2).mean(dim=0)  # mean squared derivative

    excess = torch.relu(mean_sq_deriv - max_mean_sq_d)
    penalty = excess.sum()
    return penalty

def compute_ei_ratio_penalty(network, target=0.61):
    """
    Compute structural penalty if the ratio of excitatory-targeting to inhibitory-targeting
    synaptic drive exceeds a biological reference ratio.

    Args:
        network (BrainNetwork): Network module containing trainable connections.
        target (float, optional): Target baseline excitatory-to-inhibitory balance
            ratio. Defaults to 0.61.

    Returns:
        torch.Tensor: Total E/I balance ratio penalty scalar across network connections.
    """
    total_ei_penalty = 0.0

    E_idx = [0, 2, 4, 6]
    I_idx = [1, 3, 5, 7]

    for _, conn in network.connections.items():
        if not conn.trainable:
            continue

        weights = conn.W
        num_columns = weights.shape[0] // 8

        # (columns, 8 populations, source)
        W = weights.view(num_columns, 8, -1)

        # Sum excitatory and inhibitory drive over all layers
        E_drive = W[:, E_idx, :].abs().sum(dim=(1, 2))
        I_drive = W[:, I_idx, :].abs().sum(dim=(1, 2))

        # excess = target * E_drive - I_drive
        excess = E_drive - ((E_drive + I_drive) * target)
        penalty = torch.relu(excess).pow(2).mean()

        total_ei_penalty += penalty

    return total_ei_penalty

def compute_smart_ei_ratio_penalty(network, target=0.61, eps=1e-6, beta=100.0):
    """
    Compute LogSumExp-scaled structural penalty for log-ratio deviations of excitatory
    and inhibitory synaptic drive across columns and layers.

    Args:
        network (BrainNetwork): Network module containing trainable connections.
        target (float, optional): Target baseline excitatory-to-inhibitory balance
            ratio. Defaults to 0.61.
        eps (float, optional): Epsilon value added for numerical division stability.
            Defaults to 1e-6.
        beta (float, optional): Smooth maximum temperature scaling parameter. Defaults
            to 100.0.

    Returns:
        torch.Tensor: LogSumExp-aggregated E/I balance penalty scalar.
    """
    all_excess = []

    for _, conn in network.connections.items():
        if not conn.trainable:
            continue

        W = conn.W
        num_columns = W.shape[0] // 8
        W = W.view(num_columns, 4, 2, -1)  # (columns, layers, [E,I], source)

        E_drive = W[:, :, 0, :].abs().sum(-1) + eps
        I_drive = W[:, :, 1, :].abs().sum(-1) + eps

        # Equal contribution, regardless of magnitude
        log_excess = torch.log(E_drive / I_drive) - math.log(target / (1 - target))
        excess = torch.relu(log_excess)

        all_excess.append(excess.flatten())
    excess_all = torch.cat(all_excess)

    # Penalize highest offenders the strongest
    penalty = (torch.logsumexp(beta * excess_all, dim=0) - math.log(excess_all.numel())) / beta
    return penalty
