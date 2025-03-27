import torch
import torch.nn as nn

targets = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
preds = torch.tensor([0.001, 0.999, 0.997, 0.001, 0.997, 0.002, 0.999, 0.999, 0.002, 0.001])

print(torch.cat((preds, preds[5:], preds[:5]), dim=0))

