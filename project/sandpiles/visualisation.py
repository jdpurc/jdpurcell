import matplotlib.pyplot as plt
import numpy as np

def visualise_statistic(
        stat_series: np.ndarray[int],   # series to plot
        include_zero: bool = False,     # whether to include 0 (i.e. do we care if there is no topple)
        log: bool = False,              # whether to log the counts and stats
        ax = None,                      # plot axes for formatting
        **kwargs                        # plot keyword arguments for formatting
):
    stat, counts = np.unique(stat_series, return_counts=True)

    if ax is None:
        # generate the required axes
        ax = plt.gca()

    if log:
        stat = np.log(stat)
        counts = np.log(counts)

    if not include_zero and stat[0] == 0:
        ax.scatter(stat[1:], counts[1:], **kwargs)

    else:
        ax.scatter(stat, counts, **kwargs)