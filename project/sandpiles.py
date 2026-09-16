import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import collections
import scipy.optimize
import mpmath

class ToppleStatistics():
    topples: int = 0
    area: int = 0
    loss: int = 0
    length: int = 0

class Simulation():

    board: np.ndarray[tuple[int, int]]
    topples: np.ndarray[int]
    area: np.ndarray[int]
    loss: np.ndarray[int]
    length: np.ndarray[int]


    def __init__(
            self,
            shape: tuple[int, int],      # dimensions of the board
            n_steps: int,                # number of grains dropped
            seed: int,                   # random seed
    ):
        # initialize the empty board
        self.board = np.zeros(shape)

        self.topples = np.zeros(n_steps)
        self.area = np.zeros(n_steps)
        self.loss = np.zeros(n_steps)
        self.length = np.zeros(n_steps)

        np.random.default_rng(seed=seed)

        for i in tqdm(range(0, n_steps)):

            # drop a grain randomly:
            row = np.random.randint(low=0, high=shape[0])
            col = np.random.randint(low=0, high=shape[1])
            self.board[row, col] += 1

            if self.board[row, col] == 4:
                self.board, stats = self._simulate_topple(self.board, shape, (row, col))
                self.topples[i] = stats.topples
                self.area[i] = stats.area
                self.length[i] = stats.length
                self.loss[i] = stats.loss


    def _simulate_topple(
            self,
            board: np.ndarray[tuple[int, int]],     # current board
            shape: tuple[int, int],                 # dimensions of the board
            coordinate: tuple[int, int]             # coordinate causing the topple
    ) -> tuple[np.ndarray[tuple[int, int]], ToppleStatistics]:

        queue = collections.deque()
        queue.append(coordinate)

        unique_positions = set()

        stats = ToppleStatistics()

        while queue:
            current = queue.popleft()

            # topple the pile (if required)
            if board[current[0], current[1]] >= 4:

                # compute statistics
                stats.topples += 1
                unique_positions.add(current)
                loss_delta = 0
                stats.length = max(stats.length, abs(current[0]-coordinate[0]) + abs(current[1] - coordinate[1]))

                board[current[0], current[1]] -= 4
                if current[0] - 1 >= 0:
                    board[current[0] - 1, current[1]] += 1
                    queue.append((current[0] - 1, current[1]))
                else:
                    loss_delta += 1

                if current[0] + 1 < shape[0]:
                    board[current[0] + 1, current[1]] += 1
                    queue.append((current[0] + 1, current[1]))
                else:
                    loss_delta += 1

                if current[1] - 1 >= 0:
                    board[current[0], current[1] - 1] += 1
                    queue.append((current[0], current[1] - 1))
                else:
                    loss_delta += 1

                if current[1] + 1 < shape[1]:
                    board[current[0], current[1] + 1] += 1
                    queue.append((current[0], current[1] + 1))
                else: 
                    loss_delta += 1

                stats.loss += loss_delta

        stats.area = len(unique_positions)
        return board, stats
    

def visualise_board(
        board: np.ndarray[tuple[int, int]]
):
    cmap = ListedColormap(["white", "yellow", "orange", "red"])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)


    plt.imshow(
        board,
        cmap = cmap,
        norm=norm
    )
    plt.colorbar()

    plt.show()

def visualise_statistic(
        stat_series: np.ndarray[int],   # series to plot
        include_zero: bool = False,     # whether to include 0 (i.e. do we care if there is no topple)
        log: bool = False,              # whether to log the counts and stats
        **kwargs                        # plot keyword arguments for formatting
):
    stat, counts = np.unique(stat_series, return_counts=True)

    if log:
        stat = np.log(stat)
        counts = np.log(counts)

    if not include_zero and stat[0] == 0:
        plt.scatter(stat[1:], counts[1:], **kwargs)

    else:
        plt.scatter(stat, counts, **kwargs)


def compute_powerlaw_MLEs(
        stat_series: np.ndarray[int]    # sequence of observations
):
    # MLE is solved by zeta'(s) / zeta(s) = -1/n sum(log(y_i))

    nonzero = stat_series[stat_series > 0]
    log_sum = 1/nonzero.size * sum(np.log(nonzero))

    print(log_sum)

    def root_function(s: float) -> float:
        return mpmath.zeta(s, derivative=1) / mpmath.zeta(s, n=0) + log_sum

    print(root_function(1.1), root_function(2), root_function(10))

    # Solve using root finding:
    s = mpmath.findroot(
        f = root_function,
        x0 = 2
    )

    return s

    
def neg_exp_trunc_pwr_log_likelihood(
        params: tuple[float, float],
        x: np.ndarray[int],     # vector of data
) -> float:
    alpha, l = params
    sum_x = np.sum(x)
    log_sum_x = np.sum(np.log(x))
    n = x.size
    constant = float(mpmath.polylog(alpha, np.exp(-l)))

    loglik = -(n * np.log(constant) + alpha * log_sum_x + l * sum_x)

    print(params, loglik)

    return -loglik


def compute_exp_trunc_pwr_mles(
        x: np.ndarray[int]      # vector of data
) -> tuple[float, float]:
    return scipy.optimize.minimize(
        fun = neg_exp_trunc_pwr_log_likelihood,
        args=(x,), 
        x0 = (2, 0.1),
        bounds=[
            (0, None),  # alpha > 0
            (0, None)   # lambda > 0
        ]
    )



