import numpy as np
from collections.abc import Callable
import mpmath
import scipy.optimize


# ---
# Distribution pmf/pdfs
# ---

# 1. Discrete power law distribution = zeta distributiom
# p(x; s) = 1/zeta(s) x^(-s)
#   x = 1, 2, ...
#   s in (1, inf)

def zeta_pmf(
        params: tuple[float],   # parameter s
        x: np.ndarray[int],     # data
        log: bool = False       # log-pmf if true
) -> np.ndarray[float]:
    s, = params
    if log:
        return -np.log(float(mpmath.zeta(s))) - s * np.log(x)
    else:
        return 1 / float(mpmath.zeta(s))*np.power(x, -s)


# 2. Exponentially Truncated Discrete Power Law Distribution
# p(x; a, l) = 1 / Li_a(e^-l) x^(-a) e^(-l x)
#   x = 1, 2, ...
#   a in (0, inf)
#   l in (0, inf)

def etdpl_pmf(
        params: tuple[float, float],    # parameters a, l
        x: np.ndarray[int],             # data
        log: bool = False               # log-pmf if true
) -> np.ndarray[float]:
    a, l = params
    if log:
        return -np.log(float(mpmath.polylog(a, np.exp(-l)))) - a * log(x) - l * x
    else:
        return 1 / float(mpmath.polylog(a, np.exp(-l))) * np.power(x, -a) * np.exp(-l * x)


# ---
# Computing the MLE
# ---

def compute_MLE(
    x: np.ndarray[int],
    pmf: Callable[[tuple[float, ], np.ndarray[int], bool], np.ndarray[float]],
    p0: tuple[float, ]
):
    def neg_loglik(params: tuple[float, ]) -> float:
        return - np.sum(pmf(params, x, p0))

    # minimize negative log likelihood
    return scipy.optimize.minimize(
        neg_loglik,
        p0
    )


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




