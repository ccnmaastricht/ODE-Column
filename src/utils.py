import tomllib
import argparse
import numpy as np
import torch
from pprint import pprint



def compute_firing_rate_torch(x):
    '''
    Compute the firing rates torch-friendly.
    '''
    a, b, d = 48.0, 981.0, 0.0089  # gain, threshold, noise factor
    x_nom = a * x - b
    exp_input = -d * x_nom
    exp_input = soft_clamp(exp_input)
    exp_term = torch.exp(exp_input)

    denom = 1 - exp_term
    x_activ = x_nom / denom

    if torch.isnan(x_activ).any():
        print("⚠️ NaN detected in firing rate computation!")
        print(f"x_nom max: {x_nom.max().item()}, min: {x_nom.min().item()}")
        print(f"exp_input max: {exp_input.max().item()}, min: {exp_input.min().item()}")
        print(f"exp_term max: {exp_term.max().item()}, min: {exp_term.min().item()}")
        print(f"exp_term argmax: {exp_term.argmax().item()}, argmin: {exp_term.argmin().item()}")

    return x_activ


def soft_clamp(x, max_val=80):
    return max_val * torch.tanh(x / max_val)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.description = 'Run a simulation of the coupled columns model.'
    parser.add_argument('--region',
                        type=str,
                        default='mt',
                        help='Region of the brain to simulate.')

    return parser.parse_args()


def parse_region() -> str:
    """Parse and return the region of the brain to simulate."""
    region = parse_args().region
    if region not in [
            'mt', 'v1', 'v2', 'v3', 'v3a', 'mstd', 'lip', 'fef', 'fst'
    ]:
        raise ValueError(
            'Invalid region. Please choose from mt, v1, v2, v3, v3a, mstd, lip, fef, or fst.'
        )
    return region


def load_config(filepath: str) -> dict:
    """Load and return configuration from TOML file."""
    with open(filepath, 'rb') as f:
        return tomllib.load(f)


def gain_function(x: np.array, a: float, b: float, d: float) -> np.array:
    """Compute gain function for the model."""
    return (a * x - b) / (1 - np.exp(-d * (a * x - b)))


def make_rand_stim_three_phases(num_populations, time_vec):
    """Make a random stimulus input for three phases: pre,
    stimulus and post."""
    rand_diff = np.random.uniform(-20.0, 20.0)
    input_A = 32.0 + rand_diff
    input_B = 32.0 - rand_diff
    raw_stim = torch.tensor([input_A, input_B])

    stim_vector = set_stim_three_phases(num_populations, time_vec, raw_stim)
    return stim_vector


def set_stim_three_phases(num_populations, time_vec, raw_stim):
    """Extent the given input stimulus to fit the time vector
    in three phases: pre-, stimulus, and post-."""
    stim = create_feedforward_input(num_populations, raw_stim[0], raw_stim[1])

    stim_vector = torch.zeros((len(time_vec), num_populations))
    stim_onset = int(len(time_vec) / 3)
    stim_offset = int(stim_onset + len(time_vec) / 3)
    stim_vector[stim_onset:stim_offset, :] = stim
    return stim_vector


def create_feedforward_input(num_populations: int,
                             input_colA: float,
                             input_colB: float) -> np.array:
    """Create feedforward input array for the simulation."""
    layer_4_indices = [[2, 3], [10, 11]]
    feedforward_rate = torch.zeros(num_populations)
    feedforward_rate[layer_4_indices[0]] = input_colA
    feedforward_rate[layer_4_indices[1]] = input_colB
    return feedforward_rate


def compute_firing_rate(membrane_potential: np.array, adaptation: np.array,
                        gain_function_parameters: dict) -> np.array:
    """Compute firing rate for the model"""
    return gain_function(membrane_potential - adaptation,
                         gain_function_parameters.gain,
                         gain_function_parameters.threshold,
                         gain_function_parameters.noise_factor)

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
    fr_pred_classic = compute_firing_rate_torch(mem_pred - adap_pred)
    fr_split = (pred_states.shape[3] // 3) * 2
    fr_pred = pred_states[:, :, 0, fr_split:]
    fr_pred_A = fr_pred[:, :, :8]
    fr_pred_B = fr_pred[:, :, 8:]
    fr_pred_A_sum = torch.sum(fr_pred_A * network.output_weights, dim=2)
    fr_pred_B_sum = torch.sum(fr_pred_B * network.output_weights, dim=2)
    fr_pred_sum = torch.stack([fr_pred_A_sum, fr_pred_B_sum], dim=2)

    # Compute loss between ode prediction and WangWong simulated data
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)
    return hub_loss(fr_pred_sum, true)

def huber_loss_membrane(y_pred, y_true):
    '''
    Computes Huber loss, a loss function suited for trajectories.
    Uses the membrane potential of the model prediction and target.
    '''
    hub_loss = torch.nn.SmoothL1Loss(beta=1.0)
    return hub_loss(y_pred[:, :, 0, :], y_true[:, :, 0, :])  # idx 0 = membrane potential

def mse_halfway_point(pred, true, odefunc):
    '''
    Computes the mean squared error between membrane potenials
    of column A and B at the halfway time point, i.e. in the
    middle of the stimulus phase.
    '''
    halfpoint = int(pred.shape[1] / 2)  # int(pred.shape[1] - 1)
    mem_pred, adap_pred = pred[:, halfpoint, 0, [0, 8]], pred[:, halfpoint, 1, [0, 8]]
    mem_true, adap_true = true[:, halfpoint, 0, [0, 8]], true[:, halfpoint, 1, [0, 8]]
    fr_pred = compute_firing_rate_torch(mem_pred - adap_pred)
    fr_true = compute_firing_rate_torch(mem_true - adap_true)
    return torch.mean(abs(fr_pred - fr_true))
