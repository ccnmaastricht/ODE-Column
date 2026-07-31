import torch


def torch_interp(x, xp, fp):
    """
    Perform 1D linear interpolation of function values `fp` evaluated at target query
    coordinates `x` given sample coordinates `xp` using PyTorch tensors.

    Args:
        x (torch.Tensor): Target evaluation query coordinates tensor.
        xp (torch.Tensor): Strictly increasing 1D sample coordinates tensor.
        fp (torch.Tensor): Sample function values tensor matching `xp` length.

    Returns:
        torch.Tensor: Linearly interpolated function values evaluated at coordinates `x`.
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