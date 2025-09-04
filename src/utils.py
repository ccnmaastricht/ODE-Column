import tomllib
import torch


def load_config(filepath: str) -> dict:
    '''
    Load and return configuration from TOML file.
    '''
    with open(filepath, 'rb') as f:
        return tomllib.load(f)


def compute_firing_rate(x):
    '''
    Compute the firing rates from (membrane potential - adaptation)
    '''
    a, b, d = 48.0, 981.0, 0.0089  # gain, threshold, noise factor
    x_nom = a * x - b
    exp_input = -d * x_nom
    exp_input = soft_clamp(exp_input)
    exp_term = torch.exp(exp_input)

    denom = 1 - exp_term
    x_activ = x_nom / denom
    return x_activ

def soft_clamp(x, max_val=80):
    return max_val * torch.tanh(x / max_val)


def torch_interp(x, xp, fp):
    """
    Interpolates fp at points x, given base points xp.
    """
    x = torch.clamp(x, xp[0], xp[-1])  # clamp x to the valid range of xp

    idx = torch.searchsorted(xp, x, right=True)
    idx = torch.clamp(idx, 1, len(xp) - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    slope = (y1 - y0) / (x1 - x0).unsqueeze(-1)
    return y0 + slope * (x.unsqueeze(-1) - x0.unsqueeze(-1))


### Loss functions ###

def min_max(firing_rates):
    '''
    Function to binary classify final firing rates by means of
    min-maxing. Thus, the maximum final firing rate will receive
    score=1 and the minimum will receive score=0.
    '''
    max_val = torch.max(firing_rates)
    min_val = torch.min(firing_rates)
    return (firing_rates - min_val) / (max_val - min_val)

def fr_to_binary(firing_rates, scaling_factor=1.0):
    '''
    Function to binary classify final firing rates. Loosely
    z-scores the input and passes it to a sigmoid function to
    obtain values between 0 and 1.
    '''
    threshold = torch.mean(firing_rates)
    sd_fr = torch.std(firing_rates) / scaling_factor

    fr_normalized = (firing_rates - threshold) / sd_fr
    fr_sigmoid = torch.sigmoid(fr_normalized)
    return fr_sigmoid

def huber_loss_wta(pred_states, true, network):
    '''
    Computes Huber loss, a loss function suited for trajectories.
    '''
    mem_pred, adap_pred = pred_states[:, :, 0, :16], pred_states[:, :, 0, 16:32]
    fr_pred = compute_firing_rate(mem_pred - adap_pred)
    fr_pred_A = fr_pred[:, :, :8]
    fr_pred_B = fr_pred[:, :, 8:]
    fr_pred_A_sum = torch.sum(fr_pred_A * network.output_weights, dim=2)
    fr_pred_B_sum = torch.sum(fr_pred_B * network.output_weights, dim=2)
    fr_pred_sum = torch.stack([fr_pred_A_sum, fr_pred_B_sum], dim=2)

    # Compute loss between ode prediction and WangWong simulated data
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)
    return hub_loss(fr_pred_sum, true)
