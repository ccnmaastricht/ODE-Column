import torch

from src.brain_network import BrainNetwork
from src.utils.paths import models_path



path = models_path('parity', 'parity_1.pt')

network = BrainNetwork.load(path)

stim = torch.zeros(1, 8)
network.run(stim)

network.analysis.visualize_weights(get_constrained_weights=True)
