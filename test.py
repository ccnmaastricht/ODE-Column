import numpy as np
import torch
import pickle
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import torch
import torch.fft
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time



ds_file = 'ds_uc_5000_omg.pkl'

with open(ds_file, 'rb') as f:
  ds = pickle.load(f)

def dtw_loss(y_pred, y_true):
  y_pred, y_true = y_pred.unsqueeze(0), y_true.unsqueeze(0)
  batch_size = y_pred.shape[0]
  loss = 0.0

  blep = y_pred[0].shape

  for i in range(batch_size):
      pred_seq = y_pred[i].detach().cpu().numpy().squeeze()
      true_seq = y_true[i].detach().cpu().numpy().squeeze()

      distance, _ = fastdtw(pred_seq, true_seq, dist=2)  # Using Minkowski distance with p=2
      loss += distance
  return torch.tensor(loss / batch_size, requires_grad=True)

def fourier_loss(y_pred, y_true):
    fft_pred = torch.fft.fft(y_pred, dim=-1)
    fft_true = torch.fft.fft(y_true, dim=-1)

    loss = torch.mean(torch.abs(torch.abs(fft_pred) - torch.abs(fft_true)))  # Magnitude difference
    return loss

a, b, c, d = 21, 22, 23, 0

# Plot samples
plt.plot(ds[:, a, 0])
plt.plot(ds[:, b, 0])
plt.plot(ds[:, c, 0])
plt.show()

# MAE loss
start = time.time()
print(torch.mean(torch.abs(ds[:, a, 0] - ds[:, b, 0])).item())
print(torch.mean(torch.abs(ds[:, a, 0] - ds[:, c, 0])).item())
print(torch.mean(torch.abs(ds[:, a, 0] - ds[:, a, 0])).item())
mae = time.time()
print(mae - start)

# Huber loss
huber_loss_fn = torch.nn.SmoothL1Loss(beta=1.0)
print(huber_loss_fn(ds[:, a, 0], ds[:, b, 0]).item())
print(huber_loss_fn(ds[:, a, 0], ds[:, c, 0]).item())
print(huber_loss_fn(ds[:, a, 0], ds[:, a, 0]).item())
huber = time.time()
print(huber - mae)

# Fourier loss
print(fourier_loss(ds[:, a, 0], ds[:, b, 0]).item())
print(fourier_loss(ds[:, a, 0], ds[:, c, 0]).item())
print(fourier_loss(ds[:, a, 0], ds[:, a, 0]).item())
fourier = time.time()
print(fourier - huber)

# DTW loss
print(dtw_loss(ds[:, a, 0], ds[:, b, 0]).item())
print(dtw_loss(ds[:, a, 0], ds[:, c, 0]).item())
print(dtw_loss(ds[:, a, 0], ds[:, a, 0]).item())
dtw = time.time()
print(dtw - fourier)


# Some extra code no longer using

# # Training data (true y)
# def time_varying_input(t, a, b, c, d):
#     return a * torch.sin(t * b + c) + d
#
# def A_matrix(xy, u):
#     x, y, u = xy[:,0].item(), xy[:,1].item(), u.item()
#     A = torch.tensor([
#         [(u - x**2 - y**2),     2.0],
#         [-2.0,                  (u - x**2 - y**2)]
#     ])
#     return A
#
# class Lambda(nn.Module):
#     def forward(self, t, xyu, u):
#         xy = xyu[:,:-1]
#         u = torch.tensor([[u]])
#         forward_xy = torch.mm(xy, A_matrix(xy, u))  # apply matrix A
#         forward_xyu = torch.concat((forward_xy, u), dim=1)
#         return forward_xyu
