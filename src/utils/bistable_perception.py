import numpy as np



"""
Source: 
Evers, K. S., Peters, J. C., Goebel, R., & Senden, M. (2025). 
Layered structure of cortex explains reversal dynamics in bistable perception. 
Scientific Reports, 15(1), 36878.
"""



def running_mean(x, N, outliers=False):
    """
    Computes average of last N timepoints and replaces outliers with 0.
    Args:
    x (array):          input
    N (int):            window size
    outliers (bool):    remove outliers
    """
    if outliers==False:
        mean = np.mean(x)
        for i in range(len(x)):
            if x[i] > mean*10:
                x[i] = 0
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def dominance_time(A1, A2, dt=1e-4, cutoff=.1, thresh=0.0001, sliding_window=10000):
    """
    Args:
    A1 (array):         activity of column 1; shape=(num_populations, num_time_steps)
    A2 (array):         activity of column 2; shape=(num_populations, num_time_steps)
    dt (float):         time step
    cutoff (float):     cutoff for dominance interval

    Returns:
    DT (array):         dominance intervals
    """
    # get switching points
    A1_smooth = running_mean(A1, N=sliding_window)
    A2_smooth = running_mean(A2, N=sliding_window)
    A_diff = A1_smooth - A2_smooth

    sign_diff = np.sign(A_diff)
    switch_inds = np.where(np.diff(sign_diff) != 0)[0]
    switch_times = switch_inds * dt

    DT_signed = []
    for i in range(len(switch_times) - 1):
        start = switch_inds[i]
        end = switch_inds[i + 1]
        dur = (end - start) * dt
        if dur >= cutoff:
            dominant = np.sign(np.mean(A_diff[start:end]))
            DT_signed.append(dominant * dur)

    if len(DT_signed) > 0:
        return np.array(DT_signed)

    # No switches or too short
    return np.array([np.sign(np.mean(A_diff)) * len(A1) * dt])

