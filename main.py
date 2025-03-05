import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchdiffeq import odeint


class WeightsDMF(nn.Module):
    def __init__(self, in_features, out_features):
        super(WeightsDMF, self).__init__()

        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.1)  # 8x8 matrix
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.matmul(x, self.weights.T) + self.bias


# Custom Activation Function
class ThresholdActivation(nn.Module):
    def forward(self, x):

        # gain, threshold and noise factor
        a, b, d = 48, 981, 0.0089

        x_nom = np.float64((a * x - b))
        x_activ = x_nom / (1 - np.exp(-d * x_nom))

        return x_activ


# Example Model Using Custom Layers
class TwoColumnODE(nn.Module):
    def __init__(self):
        super(TwoColumnODE, self).__init__()
        self.custom_linear = WeightsDMF(8, 8)  # Custom 8x8 weight layer
        self.activation = ThresholdActivation()  # Custom activation function

    def forward(self, x):
        x = self.custom_linear(x)
        x = self.activation(x)
        return x



if __name__ == '__main__':
    data = torch.rand((100, 8)) * 2 - 1
    sample = data[0]

    odemodel = TwoColumnODE()
    layer = WeightsDMF(8, 8)
    print(layer.weights)
    print(layer(sample))
    print(layer.weights)
