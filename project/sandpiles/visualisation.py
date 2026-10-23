import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

def visualise_statistic(
        stat_series: np.ndarray[int],       # series to plot
        include_zero: bool = False,         # whether to include 0 (i.e. do we care if there is no topple)
        log: bool = False,                  # whether to log the counts and stats
        ax = None,                          # plot axes for formatting
        marker = 'o',                       # default to a scatter plot
        pmf: Callable = None,               # pmf for MLE
        mle_params: tuple[float] = None,    # fitted MLE params
        pmf_kwargs: dict = {},              # other pmf arguments
        mle_plot_kwargs: dict = {},         # line plot kwargs
        x_max_MLE: int = None,              # optional x_max for MLE
        **kwargs                            # plot keyword arguments for formatting
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
        if x_max_MLE is None:
            x_max_MLE = max(stat)+1
        x = np.arange(1, x_max_MLE)
        y = pmf(params=mle_params, x=x, loglik=False, **pmf_kwargs)
        ax.plot(x, y, **mle_plot_kwargs)


def visualise_sandpile(board: np.ndarray[np.int8, np.int8], 
                       ax = None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    im = ax.imshow(board, vmin=0, vmax = max(4, np.max(board)))

    r, c = board.shape

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(c), labels=range(1, c+1),
                rotation=45, rotation_mode="xtick")
    ax.set_yticks(range(r), labels=range(r, 0, -1))

    # Loop over data dimensions and create text annotations.
    for i in range(r):
        for j in range(c):
            text = ax.text(j, i, str(board[i, j]),
                        ha="center", va="center", color="w")

    fig.tight_layout()


def visualise_mle_fit(
        example_subset: list[int],      # subset of gridsizes to plot
        sims,                           # simulation results
        pmf,                            # pmf to use for the MLE
        topple_params,                  # optimisation results
        area_params,                    # optimisation results
        x_max_topple,                   # s_max
        x_max_area,                     # a_max
        subplots = None,
        
):
    if subplots is None:
        fig, ax = plt.subplots(len(example_subset), 2, figsize=(10, 10))
    else:
        fig, ax = subplots

    # visualise for L=30
    for i, L_example in enumerate(example_subset):

        visualise_statistic(
            stat_series = sims[L_example].topples[sims[L_example].topples <= x_max_topple[L_example]],
            log=True,
            ax = ax[i, 0],
            pmf = pmf,
            mle_params = topple_params[L_example],
            pmf_kwargs = {"x_max": x_max_topple[L_example]},
            mle_plot_kwargs={"color": "black", 
                            "label": r"MLE: $\hat{\tau} = $" + f"{topple_params[L_example][0]:.2f}", 
                            "marker": "o",
                            "ms": 1.5},
            ms = 1.5,
            label = f"L = {L_example}",
            linestyle="None",
        )
        ax[i, 0].legend()
        
        ax[i, 0].set_ylabel("Frequency")
        ax[i, 0].set_xlim(0.5, 10**5)
        ax[i, 0].set_ylim(10**(-6), 10**0)


        visualise_statistic(
            stat_series = sims[L_example].area[sims[L_example].area <= x_max_area[L_example]],
            log=True,
            ax = ax[i, 1],
            pmf = pmf,
            mle_params = area_params[L_example],
            pmf_kwargs = {"x_max": x_max_area[L_example]},
            mle_plot_kwargs={"color": "black", 
                            "label": r"MLE: $\hat{\tau} = $" + f"{area_params[L_example][0]:.2f}",
                            "marker": "o",
                            "ms": 1.5},
            ms = 1.5,
            label = f"L = {L_example}",
            color = "C1",
            linestyle = "None"
        )
        ax[i, 1].set_xlim(0.5, 10**5)
        ax[i, 1].set_ylim(10**(-6), 10**0)
        ax[i, 1].axvline(x_max_area[L_example], linestyle="dashed", color="C1", label=r"$a_{max}(L)$")
        ax[i, 1].legend()

    ax[0, 0].set_title("Distribution of Avalanche Size")
    ax[-1, 0].set_xlabel("Avalanche Size")

    ax[0, 1].set_title("Distribution of Avalanche Area")
    ax[-1, 1].set_xlabel("Area")