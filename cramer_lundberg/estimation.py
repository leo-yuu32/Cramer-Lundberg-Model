"""
Monte Carlo estimation, confidence intervals, and bootstrap.
"""

import numpy as np


def estimate_ruin_probability(ruin_occurred):
    """
    Point estimate and 95% CIs (Wald and Wilson) for ruin probability.
    Wilson: p̃ = (n p̂ + z²/2)/(n + z²), ñ = n + z²,
    CI = p̃ ± z sqrt(p̃(1-p̃)/ñ).
    """
    ruin_occurred = np.asarray(ruin_occurred, dtype=bool)
    n = int(ruin_occurred.size)
    psi_hat = float(np.mean(ruin_occurred))
    z = 1.96  # 95%
    # Wald
    se_wald = np.sqrt(psi_hat * (1 - psi_hat) / n) if n > 0 else 0.0
    wald_lower = max(0.0, psi_hat - z * se_wald)
    wald_upper = min(1.0, psi_hat + z * se_wald)
    wald_ci = (wald_lower, wald_upper)
    # Wilson
    p_tilde = (n * psi_hat + z**2 / 2) / (n + z**2)
    n_tilde = n + z**2
    se_wilson = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    wilson_ci = (max(0.0, p_tilde - se_wilson), min(1.0, p_tilde + se_wilson))
    return {
        "psi_hat": psi_hat,
        "n": n,
        "wald_ci": wald_ci,
        "wilson_ci": wilson_ci,
    }


def estimate_ruin_time_stats(tau_values, ruin_occurred):
    """Mean, median, std, and CLT-based CI for ruin time τ (cond. on ruin)."""
    tau_values = np.asarray(tau_values)
    ruin_occurred = np.asarray(ruin_occurred, dtype=bool)
    tau_cond = tau_values[ruin_occurred]
    n_ruin = int(tau_cond.size)
    if n_ruin == 0:
        return {
            "E_tau": None,
            "median_tau": None,
            "std_tau": None,
            "ci_tau": None,
        }
    mean_tau = float(np.mean(tau_cond))
    median_tau = float(np.median(tau_cond))
    std_tau = float(np.std(tau_cond, ddof=1))
    se_mean = std_tau / np.sqrt(n_ruin)
    z = 1.96
    ci_tau = (mean_tau - z * se_mean, mean_tau + z * se_mean)
    return {
        "E_tau": mean_tau,
        "median_tau": median_tau,
        "std_tau": std_tau,
        "ci_tau": ci_tau,
    }


def estimate_deficit_stats(deficit_at_ruin, ruin_occurred):
    """Mean deficit at ruin and CI (conditional on ruin)."""
    deficit_at_ruin = np.asarray(deficit_at_ruin)
    ruin_occurred = np.asarray(ruin_occurred, dtype=bool)
    def_cond = deficit_at_ruin[ruin_occurred]
    n_ruin = int(def_cond.size)
    if n_ruin == 0:
        return {"mean_deficit": None, "ci_deficit": None}
    mean_d = float(np.mean(def_cond))
    std_d = float(np.std(def_cond, ddof=1))
    se = std_d / np.sqrt(n_ruin)
    z = 1.96
    ci_deficit = (mean_d - z * se, mean_d + z * se)
    return {"mean_deficit": mean_d, "ci_deficit": ci_deficit}


def compute_convergence(ruin_occurred, n_points=50):
    """Running ψ̂(u) and Wilson CI at n_points sample sizes from 100 to N."""
    ruin_occurred = np.asarray(ruin_occurred, dtype=bool)
    N = int(ruin_occurred.size)
    k_vals = np.linspace(100, N, n_points, dtype=int)
    k_vals = np.unique(np.clip(k_vals, 1, N))
    sample_sizes = []
    estimates = []
    ci_lower = []
    ci_upper = []
    z = 1.96
    for k in k_vals:
        sub = ruin_occurred[:k]
        n, p = k, float(np.mean(sub))
        p_tilde = (n * p + z**2 / 2) / (n + z**2)
        n_tilde = n + z**2
        se = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        sample_sizes.append(n)
        estimates.append(p)
        ci_lower.append(max(0.0, p_tilde - se))
        ci_upper.append(min(1.0, p_tilde + se))
    return {
        "sample_sizes": np.array(sample_sizes),
        "estimates": np.array(estimates),
        "ci_lower": np.array(ci_lower),
        "ci_upper": np.array(ci_upper),
    }


def bootstrap_ci(data, stat_fn, n_bootstrap=2000, alpha=0.05):
    """Percentile bootstrap CI: stat_fn takes 1D array, returns scalar."""
    data = np.asarray(data).ravel()
    n = data.size
    boot_stats = []
    rng = np.random.default_rng()
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats.append(stat_fn(data[idx]))
    boot_stats = np.array(boot_stats)
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


def analyze_ruin_decay(
    u_range, c, lambda_, claim_dist, claim_params, n_paths, T, seed_base=42
):
    """
    Analyze empirical ruin probability decay across initial surplus values.

    Args:
        u_range: Array-like of initial surplus values to test
        c: Premium rate
        lambda_: Claim intensity
        claim_dist: Claim distribution name
        claim_params: Claim distribution parameters dict
        n_paths: Number of paths per u value
        T: Time horizon
        seed_base: Base seed (each u gets seed_base + u_index)

    Returns:
        tuple: (u_values, psi_values) as numpy arrays
    """
    from simulation import CramerLundbergModel

    u_values = np.asarray(u_range)
    psi_values = []

    for i, u_val in enumerate(u_values):
        model = CramerLundbergModel(
            u_val, c, lambda_, claim_dist, claim_params
        )
        results = model.simulate_paths(n_paths, T, seed=seed_base + i)
        psi_hat = float(np.mean(results["ruin_occurred"]))
        psi_values.append(psi_hat)

    return u_values, np.array(psi_values)
