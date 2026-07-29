import torch
import math



def huber_loss_wta(pred_states, true):
    """
    Computes Huber loss, a loss function suited for trajectories.
    """
    true_reshaped = true.transpose(0, 1).contiguous()

    # Compute loss between model prediction and WangWong simulated data
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)
    return hub_loss(pred_states, true_reshaped)

def compute_suppression_penalty(model_predictions, labels, num_classes):
    """
    Computes the penalty for high activations for incorrect class labels.
    Facilitates contrastive learning.
    """
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    suppression = ((1 - one_hot_labels) * model_predictions).mean()
    return suppression

def compute_L2_regularization(network):
    """
    Computes the penalty for large weights (i.e. L2 regularization).
    """
    total_sq_sum = 0.0
    total_count = 0

    for _, conn in network.connections.items():
        if not conn.trainable:
            continue
        total_sq_sum = total_sq_sum + (conn.W ** 2).sum()
        total_count += conn.W.numel()

    return total_sq_sum / total_count

def compute_ei_ratio_penalty(network, target=0.61):
    """
    Computes the ratio of connections targeting the excitatory against
    inhibitory populations in the target columns. Produces a penalty if
    the ratio is significantly higher than the biologically realistic
    standard ratio (=0.61).
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

        excess = target * E_drive - I_drive
        penalty = torch.relu(excess).pow(2).mean()

        total_ei_penalty += penalty

    return total_ei_penalty

def compute_smart_ei_ratio_penalty(network, target=0.61, eps=1e-6, beta=100.0):
    """
    Computes the ratio of connections targeting the excitatory against
    inhibitory populations in the target columns. Produces a penalty if
    the ratio is significantly higher than the biologically realistic
    standard ratio (=0.61).
    Beta: smaller penalizes the mean, bigger penalizes the maximum.
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

def min_max(firing_rates):
    """
    Function to binary classify final firing rates by means of
    min-maxing. Thus, the maximum final firing rate will receive
    score=1 and the minimum will receive score=0.
    """
    max_val = torch.max(firing_rates)
    min_val = torch.min(firing_rates)
    return (firing_rates - min_val) / (max_val - min_val)

def fr_to_binary(firing_rates, scaling_factor=1.0):
    """
    Function to binary classify final firing rates. Loosely
    z-scores the input and passes it to a sigmoid function to
    obtain values between 0 and 1.
    """
    threshold = torch.mean(firing_rates)
    sd_fr = torch.std(firing_rates) / scaling_factor

    fr_normalized = (firing_rates - threshold) / sd_fr
    fr_sigmoid = torch.sigmoid(fr_normalized)
    return fr_sigmoid
