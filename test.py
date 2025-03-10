import numpy as np

from scipy import signal, ndimage, stats

import warnings

warnings.filterwarnings("ignore")


# import IPython


def running_mean(x, N, outliers=False):
    """
    Args:
    x (array):          input
    N (int):            window size
    outliers (bool):    remove outliers

    Returns:
    (array):            running mean

    """

    if outliers == False:
        mean = np.mean(x)
        for i in range(len(x)):
            if x[i] > mean * 10:
                x[i] = 0
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# dominance time
def dominance_time(A1, A2, dt=1e-4, cutoff=.1):
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
    A1_smooth = running_mean(A1, N=10000)
    A2_smooth = running_mean(A2, N=10000)
    A_diff = A1_smooth - A2_smooth
    ind = np.where(abs(A_diff) <= 0.0001)[0]
    switch = ind * dt  # in seconds
    switch = np.round(switch, 2)
    switch = np.unique(switch)

    # get switching intervals
    if len(switch) > 1:
        DT = np.empty(len(switch) - 1)
        for k in range(len(switch) - 1):
            DT_ = switch[k + 1] - switch[k]
            if DT_ >= cutoff:
                DT[k] = DT_  # in seconds
            else:
                DT[k] = np.nan
        not_nan_ind = ~ np.isnan(DT)
        DT = DT[not_nan_ind]
        if len(DT) >= 1:
            return DT
        else:
            return np.array([len(A1) * dt])
    else:
        return np.array([len(A1) * dt])  # WTA


def alternation_rate(A1, A2, dt=1e-4, cutoff=.1):
    """
    Args:
    A1 (array):         activity of column 1; shape=(num_populations, num_time_steps)
    A2 (array):         activity of column 2; shape=(num_populations, num_time_steps)
    dt (float):         time step
    cutoff (float):     cutoff for dominance interval

    Returns:
    AR (float):         alternation rate
    """

    A_diff = running_mean(A1, N=1000) - running_mean(A2, N=1000)

    AL = 0

    k = 0
    for t in range(len(A_diff)):
        if k == 0:
            current = np.sign(A_diff[t])
            k += 1
        else:
            if np.sign(A_diff[t]) != current and k * dt >= cutoff:
                k = 0
                AL += 1
            else:
                k += 1

    AL /= (len(A_diff) * dt)

    return AL


def predominance_time(A1, A2, dt=1e-4):
    """
    Args:
    A1 (array):         activity of column 1; shape=(num_populations, num_time_steps)
    A2 (array):         activity of column 2; shape=(num_populations, num_time_steps)
    dt (float):         time step

    Returns:
    A1_PD (float):      predominance time of column 1
    A2_PD (float):      predominance time of column 2
    """

    A_diff = running_mean(A1, N=1000) - running_mean(A2, N=1000)

    A1_PD = []
    A2_PD = []

    K = []
    k = 0
    for t in range(len(A_diff)):
        if k == 0:
            current = np.sign(A_diff[t])
            k += 1
        else:
            if np.sign(A_diff[t]) == current:
                k += 1
            else:
                if current == 1:
                    A1_PD.append(k * dt)
                else:
                    A2_PD.append(k * dt)
                k = 0  # reset
        K.append(k)

    return np.mean(A1_PD), np.mean(A2_PD)


def fit_gamma(y, t_sim, hist=True, plot=True, fig=None, color='#c44343ff'):
    """
    Args:
    y (array):          data
    t_sim (float):      simulation time
    hist (bool):        plot histogram
    plot (bool):        plot pdf
    fig (figure):       figure
    color (str):        color

    Returns:
    param (tuple):      gamma distribution parameters
    moment (list):      gamma distribution moments
    [x, pdf_fitted]:    pdf of fitted gamma distribution
    """

    gamma = stats.gamma

    x = np.linspace(0.00001, y.max() + y.max() * 0.25, 100)
    x = np.linspace(0.00001, 100, 100)

    y = y[y != 0]

    if len(y) > 1 and not np.min(y) == np.max(y):
        hist_, bins = np.histogram(y, density=True, bins=30)  # histogram
        corr = t_sim / 0.0000001 + bins  # correct for time
        height = hist_ / corr[:-1]  # correct for bins
        height = (height - np.min(height)) / 0.0000001 + (np.max(height) - np.min(height))  # normalize histogram

        # fit
        try:
            param = gamma.fit(y, floc=0)  # fit gamma distribution
            pdf_fitted = gamma.pdf(x, *param)  # pdf of fitted gamma distribution
            corr = t_sim / x  # correct for time
            pdf_fitted = pdf_fitted / corr  # correct for bins
            pdf_fitted = (pdf_fitted - np.min(pdf_fitted)) / (np.max(pdf_fitted) - np.min(pdf_fitted))  # normalize pdf

            if plot:
                import pylab as plt
                if fig is None:
                    fig = plt.figure()
                ax = plt.subplot(111)
                ax.plot(x, pdf_fitted, color=color, lw=2)

                plt.xlim(left=0, right=y.max() + y.max() * 0.25)
                plt.xticks(np.linspace(np.min(y), np.max(y), 4), np.linspace(np.min(y), np.max(y), 4).astype(int),
                           fontsize=20)
                if hist:
                    plt.bar(x=bins[:-1], height=height, width=np.max(bins[:-1]) / 30, color=color, alpha=0.8)
                plt.yticks([], [])  # yticks have no interpretable meaning
                plt.xlabel('Dominance Duration [s]', fontsize=20)

                ax.spines['bottom'].set_linewidth(3)
                ax.spines['left'].set_linewidth(3)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                plt.tight_layout()

            # moments
            if not np.isnan(pdf_fitted).any():
                moment = [
                    np.where(pdf_fitted == np.max(pdf_fitted))[0][0],  # mode
                    stats.moment(pdf_fitted, moment=2),  # variance
                    stats.moment(pdf_fitted, moment=3),  # skewness
                    stats.moment(pdf_fitted, moment=4)  # kurtosis
                ]
                return param, moment, [x, pdf_fitted]
            else:
                param = tuple([None, None, None])
                moment = list([None, None, None, None])
                return param, moment, [x, None]

        except:
            param = tuple([None, None, None])
            moment = list([None, None, None, None])
            return param, moment, [x, None]

    else:
        param = tuple([None, None, None])
        moment = list([None, None, None, None])
        return param, moment, [x, None]


