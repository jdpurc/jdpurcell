import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

def visualise_statistic(
        stat_series: np.ndarray[int],   # series to plot
        include_zero: bool = False,     # whether to include 0 (i.e. do we care if there is no topple)
        log: bool = False,              # whether to log the counts and stats
        ax = None,                      # plot axes for formatting
        marker = 'o',                   # default to a scatter plot
        pmf: Callable = None,           # pmf for MLE
        mle_params: tuple[float] = None,# fitted MLE params
        pmf_kwargs: dict = {},          # other pmf arguments
        mle_plot_kwargs: dict = {},     # line plot kwargs
        **kwargs                        # plot keyword arguments for formatting
):
    # get the frequency by size of event
    stat, counts = np.unique(stat_series, return_counts=True)

    # remove 0 if requested
    if not include_zero and stat[0] == 0:
        stat = stat[1:]
        counts = counts[1:]

    freq = counts / np.sum(counts)

    if ax is None:
        # generate the required axes
        ax = plt.gca()

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.plot(stat, freq, marker=marker, **kwargs)

    if pmf is not None:
        x = np.arange(1, max(stat)+1)
        y = pmf(params=mle_params, x=x, loglik=False, **pmf_kwargs)
        ax.plot(x, y, **mle_plot_kwargs)
