import matplotlib.pyplot as plt
import numpy as np

def visualise_statistic(
        stat_series: np.ndarray[int],   # series to plot
        include_zero: bool = False,     # whether to include 0 (i.e. do we care if there is no topple)
        log: bool = False,              # whether to log the counts and stats
        ax = None,                      # plot axes for formatting
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
        stat = np.log(stat)
        freq = np.log(freq)

    ax.scatter(stat, freq, **kwargs)