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
                             base_line_drive: float,
                             delta_drive: float) -> np.array:
    """Create feedforward input array for the simulation."""
    feedforward_rate = np.zeros(num_populations)
    feedforward_rate[layer_4_indices[0]] = (base_line_drive + delta_drive / 2)
    feedforward_rate[layer_4_indices[1]] = (base_line_drive - delta_drive / 2)
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
    return hub_loss(y_pred, y_true)

def mse_halfway_point(pred, true):
    '''
    Computes the mean squared error between membrane potenials
    of column A and B at the halfway time point, i.e. in the
    middle of the stimulus phase.
    '''
    halfpoint = int(pred.shape[1] / 2)
    return torch.mean(abs(pred[:, halfpoint, 0, [0, 8]] - true[:, halfpoint, 0, [0, 8]]))
