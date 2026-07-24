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
    total_L2_reg = 0

    for name, conn in network.connections.items():
        if conn.trainable:

            L2_reg = (conn.W ** 2).mean()
            total_L2_reg += L2_reg

    return total_L2_reg

def compute_ei_ratio_penalty(network, target=0.61, eps=1e-6, beta=10.0):
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
        log_excess = torch.log(E_drive / I_drive) - math.log(target)
        excess = torch.relu(log_excess)

        all_excess.append(excess.flatten())
    excess_all = torch.cat(all_excess)

    # Penalize highest offenders the strongest
    penalty = torch.logsumexp(beta * excess_all, dim=0) / beta
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
