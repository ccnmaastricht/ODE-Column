import tomllib
import argparse
import numpy as np
import torch

from dataclasses import dataclass


@dataclass
class GainFunctionParams:
    gain: float
    threshold: float
    noise_factor: float


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


def create_feedforward_input(num_populations: int, layer_4_indices: tuple,
                             input_colA: float,
                             input_colB: float) -> np.array:
    """Create feedforward input array for the simulation."""
    feedforward_rate = np.zeros(num_populations)
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

def huber_loss(y_pred, y_true):
    '''
    Computes Huber loss, a loss function suited for trajectories.
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
    fr_pred = odefunc.compute_firing_rate_torch(mem_pred - adap_pred, odefunc.gain_function_parameters)
    fr_true = odefunc.compute_firing_rate_torch(mem_true - adap_true, odefunc.gain_function_parameters)
    return torch.mean(abs(fr_pred - fr_true))
    # return torch.mean(abs(pred[:, halfpoint, 0, [0, 8]] - true[:, halfpoint, 0, [0, 8]]))
