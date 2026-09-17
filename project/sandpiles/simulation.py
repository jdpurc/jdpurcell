import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
from numba import njit

# Run the abelian sandpiles simulation
@njit
def simulate_sandpiles(
        shape: tuple[int, int],                     # finite grid size
        n_steps: int,                               # number of simulation steps
        seed: int,                                  # random seed
        board: np.ndarray[np.int8, np.int8] = None  # optional board to continue an earlier simulation
) -> tuple[np.ndarray[np.int8, np.int8], np.ndarray[np.int64], np.ndarray[np.int64], np.ndarray[np.int64], np.ndarray[np.int64]]:

    # initialise the board to be empty
    if board is None:
        board = np.zeros(shape, dtype=np.int8)

    # initialise the statistics
    topples = np.zeros(n_steps, dtype=np.int64)
    area = np.zeros(n_steps, dtype=np.int64)
    loss = np.zeros(n_steps, dtype=np.int64)
    length = np.zeros(n_steps, dtype=np.int64)

    np.random.seed(seed)

    # initialise queue here to avoid initialisations in the simulation step for efficiency
    queue = np.empty((shape[0] * shape[1], 2), dtype=np.int64)

    # keep track of the last time each square was toppled
    last_toppled = np.zeros(board.shape, dtype=np.int64)

    # simulate
    for i in range(0, n_steps):
        topples[i], area[i], loss[i], length[i] = simulate_step(board, queue, last_toppled, i)

    return board, topples, area, loss, length

               
# Run one step of the abelian sandpile model
@njit
def simulate_step(
        board: np.ndarray[int, int],                # state of the board prior to the new grain drop
        queue: np.ndarray[int, int] = None,         # pre-loaded queue
        last_toppled: np.ndarray[int, int] = None,  # keep track of last time space was toppled
        iter: int = 1,                              # current iteration
) -> tuple[int, int, int, int]:

    shape = board.shape

    if queue is None:
        queue = np.empty((shape[0] * shape[1], 2), dtype=np.int64)

    if last_toppled is None:
        last_toppled = np.zeros(board.shape, dtype=np.int64)

    # drop a grain randomly:
    row = np.random.randint(low=0, high=shape[0])
    col = np.random.randint(low=0, high=shape[1])
    board[row, col] += 1

    # check for a topple
    if board[row, col] == 4:
        topple, area, loss, length = _simulate_topple(board, np.array((row, col)), queue, last_toppled, iter)
    else:
        topple, area, loss, length = 0, 0, 0, 0

    return topple, area, loss, length


# simulate the toppling function in the abelian samdpile model
@njit
def _simulate_topple(
            board: np.ndarray[tuple[int, int]],     # current board
            coordinate: np.ndarray[int],            # coordinate causing the topple
            queue: np.ndarray[int, int],            # pre-loaded queue
            last_toppled: np.ndarray[int, int],     # when each position was last toppled
            iter: int                               # current iteration
    ) -> tuple[int, int, int, int]:

    shape = board.shape

    # queue implementation is a circular array, to allow optimising with numba njit
    qhead = 0
    queue[0] = coordinate
    qtail = 1
    qsize = 1

    # maximum elements that can be stored in the queue (should equal number of elements in grid)
    maxsize = queue.shape[0]


    topple, area, loss, length = 0, 0, 0, 0

    # define a helper function here to avoid code repetition
    def topple_helper(x, y, board, shape, queue, qtail, qsize, maxsize, loss):
        if x >= 0 and x < shape[0] and y >= 0 and y < shape[1]:

            board[x, y] += 1

            # if neighbour should topple, add it to the queue
            if board[x, y] == 4:
                queue[qtail, 0] = x
                queue[qtail, 1] = y
                qtail = (qtail + 1) % maxsize
                qsize += 1

        else:
            # grains are lost to the edge
            loss += 1

        return qtail, qsize, loss
        

    while qsize > 0:

        # dequeue and adjust indexes
        current = queue[qhead]
        qhead = (qhead + 1) % maxsize
        qsize -= 1

        # verify that a toppling is required
        if board[current[0], current[1]] < 4:
            continue

        # perform the toppling
        board[current[0], current[1]] -= 4

        topple += 1
        l1_distance = abs(current[0] - coordinate[0]) + abs(current[1] - coordinate[1])
        if l1_distance > length:
            length = l1_distance

        # if this is new area for this avalance, increase the area
        if last_toppled[current[0], current[1]] != iter:
            area += 1
            last_toppled[current[0], current[1]] = iter

        # propagate the toppling
        qtail, qsize, loss = topple_helper(current[0] - 1, current[1], board, shape, queue, qtail, qsize, maxsize, loss)
        qtail, qsize, loss = topple_helper(current[0] + 1, current[1], board, shape, queue, qtail, qsize, maxsize, loss)
        qtail, qsize, loss = topple_helper(current[0], current[1] - 1, board, shape, queue, qtail, qsize, maxsize, loss)
        qtail, qsize, loss = topple_helper(current[0], current[1] + 1, board, shape, queue, qtail, qsize, maxsize, loss)
        
    return topple, area, loss, length
    
    

class ToppleStatistics():
    topples: int = 0
    area: int = 0
    loss: int = 0
    length: int = 0

class SandpileSimulation():

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

