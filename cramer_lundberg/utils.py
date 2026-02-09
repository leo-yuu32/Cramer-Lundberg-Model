"""
Shared helpers and distribution wrappers for Cramér-Lundberg model.
"""

import numpy as np
from scipy import stats


def get_distribution(name: str, params: dict):
    """Return a frozen scipy.stats distribution given name and params."""
    name = name.lower().strip()
    if name == "exponential":
        # params: mu (mean)
        return stats.expon(scale=params["mu"])
    if name == "gamma":
        # params: shape (alpha), rate (beta)
        return stats.gamma(a=params["alpha"], scale=1.0 / params["beta"])
    if name == "lognormal":
        # params: mu_log, sigma_log (shape, scale in scipy: s=sigma, scale=exp(mu))
        return stats.lognorm(s=params["sigma_log"], scale=np.exp(params["mu_log"]))
    if name == "pareto":
        # params: alpha (shape), x_m (scale = minimum)
        return stats.pareto(b=params["alpha"], scale=params["x_m"])
    raise ValueError(f"Unknown distribution: {name}")


def get_claim_mean(name: str, params: dict) -> float:
    """Return the mean of the claim distribution."""
    dist = get_distribution(name, params)
    return float(dist.mean())


def format_ci(lower: float, upper: float, decimals: int = 4) -> str:
    """Format confidence interval as string (lower, upper)."""
    return f"({lower:.{decimals}f}, {upper:.{decimals}f})"
