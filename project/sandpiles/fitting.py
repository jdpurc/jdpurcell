import numpy as np
from collections.abc import Callable
import mpmath
import scipy.optimize
from typing import Any


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
        loglik: bool = False    # log-pmf if true
) -> np.ndarray[float]:
    s, = params
    if loglik:
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
        loglik: bool = False            # log-pmf if true
) -> np.ndarray[float]:
    a, l = params
    if loglik:
        return -np.log(float(mpmath.polylog(a, np.exp(-l)))) - a * np.log(x) - l * x
    else:
        return 1 / float(mpmath.polylog(a, np.exp(-l))) * np.power(x, -a) * np.exp(-l * x)

# 2. Super-Exponentially Truncated Discrete Power Law Distribution
# p(x; a, l) = 1 / C x^(-a) e^(-l x^b)
#   x = 1, 2, ...
#   a in (0, inf)
#   l in (0, inf)
#   b in (1, inf)

def setdpl_pmf(
        params: tuple[float, float, float],     # parameters a, l, b
        x: np.ndarray[int],                     # data
        loglik: bool = False                    # log-pmf if true
) -> np.ndarray[float]:
    a, l, b = params
    # approximate the constant C by summing up until x = 1000
    const = np.sum(np.power(np.arange(1, 100_000), -a) * np.exp(-l * np.power(np.arange(1, 100_000), b)))
    if loglik:
        return -np.log(const) - a * np.log(x) - l * np.power(x, b)
    else:
        return 1 / float(const) * np.power(x, -a) * np.exp(-l * np.power(x, b))

# ---
# Computing the MLE
# ---

def compute_MLE(
    x: np.ndarray[int],                     # observed data
    pmf: Callable,                          # pmf function f(params, x, log=True)
    p0: tuple[float, ],                     # parameter initial guess
    bounds: tuple[tuple[float, float], ]    # parameter space
) -> Any:
    def neg_loglik(params: tuple[float, ]) -> float:
        return - np.sum(pmf(params, x, True))

    # minimize negative log likelihood
    return scipy.optimize.minimize(
        fun = neg_loglik,
        x0 = p0,
        bounds=bounds,
    )
