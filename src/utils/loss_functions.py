import torch



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

            L2_reg = (conn.weights ** 2).mean()
            total_L2_reg += L2_reg

    return total_L2_reg

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

        weights = conn.weights
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
