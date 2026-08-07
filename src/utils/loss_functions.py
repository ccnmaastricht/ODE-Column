import torch
import math



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

def compute_fr_ceiling_penalty(rates, max_fr=0.0, exponent=4):
    """
    Compute penalty for firing rates exceeding a maximum threshold, using a
    power penalty on the excess above the threshold.

    Args:
        rates (torch.Tensor): Firing rate tensor, shape
            `(time_steps, batch_size, total_populations)`.
        max_fr (float, optional): Maximum allowed firing rate threshold.
            Defaults to 100.0.
        exponent (int, optional): Power to which the excess above `max_fr`
            is raised. Defaults to 4.

    Returns:
        torch.Tensor: Firing rate ceiling penalty scalar.
    """
    return (torch.relu(rates - max_fr) ** exponent).mean()

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
