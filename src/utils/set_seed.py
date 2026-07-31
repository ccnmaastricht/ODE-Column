import torch
import numpy as np
import random


def set_seed(seed):
    """
    Set pseudorandom random seeds across PyTorch, NumPy, and standard library random
    generators for reproducible simulation runs.

    Args:
        seed (int): Integer seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)