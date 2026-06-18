import torch



def huber_loss_wta(pred_states, true, network):
    """
    Computes Huber loss, a loss function suited for trajectories.
    """
    num_columns = true.shape[2]

    true_reshaped = true.transpose(0, 1).contiguous()
    pred_reshaped = torch.zeros_like(true_reshaped)

    for i_col in range(num_columns):
        fr_pred_col = pred_states[:, :, i_col*8 : (i_col+1)*8]
        fr_pred_col_sum = torch.sum(fr_pred_col * network.output_weights, dim=2)
        pred_reshaped[:, :, i_col] = fr_pred_col_sum

    # Compute loss between model prediction and WangWong simulated data
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)
    return hub_loss(pred_reshaped, true_reshaped)

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
