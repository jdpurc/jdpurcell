import numpy as np
from collections.abc import Callable
import mpmath
import scipy.optimize
from scipy.special import logsumexp
from typing import Any







# 2. Super-Exponentially Truncated Discrete Power Law Distribution
# p(x; a, l) = 1 / C x^(-a) e^(-l x^b)
#   x = 1, 2, ...
#   a in (0, inf)
#   l in (0, inf)
#   b in (1, inf)

def setdpl_const(a: float, 
                 l: float, 
                 b: float, 
                 N: int = 100_000,
                 ) -> float:

    return np.sum(np.power(np.arange(1, N), -a) * np.exp(-l * np.power(np.arange(1, N), b)))

def setdpl_pmf(
        params: tuple[float, float, float],     # parameters a, l, b
        x: np.ndarray[int],                     # data
        loglik: bool = False,                   # log-pmf if true
) -> np.ndarray[float]:
    a, l, b = params
    # approximate the constant C
    const = setdpl_const(a, l, b)
    if loglik:
        val = -np.log(const) - a * np.log(x) - l * np.power(x, b)
    else:
        val = 1 / float(const) * np.power(x, -a) * np.exp(-l * np.power(x, b))

    return val

def setdpl_logjac(
        params: tuple[float, float, float],     # parameters a, l, b
        x: np.ndarray[int],                     # data
) -> float:
    a, l, b = params
    n = x.size

    # compute the gradients
    derv_a = n * a * np.exp(np.log(setdpl_const(a+1, l, b)) - np.log(setdpl_const(a, l, b))) - np.sum(np.log(x))
    derv_l = n * np.exp(np.log(setdpl_const(a - b, l, b)) - np.log(setdpl_const(a, l, b))) - np.sum(np.power(x, b))
    derv_b = b * n * l * np.exp(np.log(setdpl_const(a - b + 1, l, b)) - np.log(setdpl_const(a, l, b))) - l * b * np.sum(np.power(x, b-1))

    print(a, l, b, derv_a, derv_l, derv_b)

    return np.array([derv_a, derv_l, derv_b])




# ---
# Computing the MLE
# ---

def compute_MLE(
    pmf: Callable,                          # pmf function f(params, x, log=True)
    p0: tuple[float, ],                     # parameter initial guess
    bounds: tuple[tuple[float, float], ],   # parameter space
    scale: tuple[float] = None,             # rescale parameters to help optimisation
    pmf_kwargs: dict = {},                  # keyword arguments to pass to pmf
    logjac: Callable | None = None,         # jacobian of log-likelihood
    **kwargs                                # other kwargs to pass to optimisation
) -> Any:
    if scale is None:
        scale = np.ones(len(p0))
    else:
        scale = np.array(scale)
    
    def neg_loglik(params: tuple[float, ]) -> float:
        params = np.array(params)
        scaled_params = params*scale
        return - np.sum(pmf(params=scaled_params, loglik=True, **pmf_kwargs))

    if logjac is not None:
        def neg_logjac(params: tuple[float, ]) -> float:
            return -logjac(params, pmf_kwargs['x'])
        jac = neg_logjac

    else:
        jac = False

    # minimize negative log likelihood
    return scipy.optimize.minimize(
        fun = neg_loglik,
        x0 = p0,
        bounds=bounds,
        jac = jac,
        **kwargs
    ).x * scale
