import torch



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