import numpy as np
from scipy.special import logsumexp
# ---
# Distribution pmf/pdfs
# ---

# 1. Discrete power law distribution
# p(x; s, x_max) = 1/C x^(-s)
#   x = 1, 2, ... x_max
#   s in (0, inf)

def power_law_pmf(
        params: tuple[float],   # parameter s
        x: np.ndarray[int],     # data
        x_max: int,             # hard cutoff (inclusive)
        loglik: bool = False    # log-pmf if true
) -> np.ndarray[float]:         # outputs vector of (log-)pmf values computed
    s, = params

    # compute the constant using log-sum-exp to avoid numeric over/underflow
    log_C = logsumexp(-s * np.log(np.arange(1, x_max + 1)))

    if loglik:
        return -log_C - s * np.log(x)
    else:
        return 1 / np.exp(log_C) * np.power(x, -s)


# 2. Exponentially Truncated Discrete Power Law Distribution
# p(x; tau, c, x_max) = 1 / C x^(-tau) e^(- x/c)
#   x = 1, 2, ... x_max
#   tau in (0, inf)
#   c in (0, inf)

def exp_power_law_pmf(
        params: tuple[float, float],    # parameters tau, c
        x: np.ndarray[int],             # data
        x_max: int,                     # hard cutoff (inclusive)
        loglik: bool = False            # log-pmf if true
) -> np.ndarray[float]:                 # outputs vector of (log-)pmf values computed
    tau, c = params

    k = np.arange(1, x_max + 1)
    log_C = logsumexp(-tau * np.log(k) - k/c)

    if loglik:
        return -log_C - tau * np.log(x) - x/c
    
    else: 
        return 1/np.exp(log_C) * np.power(x, -tau) * np.exp(-x/c)