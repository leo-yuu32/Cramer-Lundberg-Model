"""
Core Cramér-Lundberg model simulation engine.
"""

import numpy as np
from utils import get_distribution


class CramerLundbergModel:
    """Cramér-Lundberg surplus process X(t) = u + rt - sum(B_i)."""

    def __init__(self, u, c, lambda_, claim_dist, claim_params):
        self.u = float(u)
        self.c = float(c)
        self.lambda_ = float(lambda_)
        self.claim_dist_name = claim_dist
        self.claim_params = dict(claim_params)
        self.claim_dist = get_distribution(claim_dist, claim_params)

    @staticmethod
    def compute_premium_rate(lambda_, mu, theta):
        """c = (1 + θ) λ μ."""
        return (1.0 + float(theta)) * float(lambda_) * float(mu)

    def simulate_paths(self, n_paths, T, seed=None):
        """
        Simulate n_paths surplus paths over [0, T].
        Surplus is linear between claims; ruin can only occur at claim times.
        """
        rng = np.random.default_rng(seed)
        times_list = []
        surpluses_list = []
        ruin_occurred = np.zeros(n_paths, dtype=bool)
        ruin_times = np.full(n_paths, np.inf)
        deficit_at_ruin = np.zeros(n_paths)
        min_surplus = np.full(n_paths, np.inf)

        for i in range(n_paths):
            t_cum = 0.0
            arrivals = []
            while t_cum < T:
                dt = rng.exponential(1.0 / self.lambda_)
                t_cum += dt
                if t_cum < T:
                    arrivals.append(t_cum)
            arrivals = np.array(arrivals) if arrivals else np.array([])
            n_claims = len(arrivals)

            if n_claims == 0:
                t_vals = np.array([0.0, T])
                u_vals = np.array([self.u, self.u + self.c * T])
            else:
                claim_sizes = self.claim_dist.rvs(size=n_claims, random_state=rng)
                cum_claims = np.cumsum(claim_sizes)
                surplus_at_claims = (
                    self.u + self.c * arrivals - cum_claims
                )
                t_vals = np.concatenate([[0.0], arrivals, [T]])
                u_start = np.array([self.u])
                u_end = (
                    self.u + self.c * T - cum_claims[-1]
                    if n_claims > 0
                    else self.u + self.c * T
                )
                u_vals = np.concatenate([u_start, surplus_at_claims, [u_end]])

                below = np.where(surplus_at_claims < 0)[0]
                if len(below) > 0:
                    first_ruin_idx = int(below[0])
                    ruin_occurred[i] = True
                    ruin_times[i] = arrivals[first_ruin_idx]
                    deficit_at_ruin[i] = -surplus_at_claims[
                        first_ruin_idx
                    ]

            times_list.append(t_vals)
            surpluses_list.append(u_vals)
            min_surplus[i] = np.min(u_vals)

        return {
            "t": times_list,
            "U_t": surpluses_list,
            "ruin_occurred": ruin_occurred,
            "tau": ruin_times,
            "deficit_at_ruin": deficit_at_ruin,
            "min_surplus": min_surplus,
        }
