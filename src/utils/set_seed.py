import torch
import numpy as np
import random



def set_seed(seed):
    """
    Sets the random seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)