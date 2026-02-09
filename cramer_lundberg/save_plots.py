"""
Export all Cramér-Lundberg plots as PDFs for LaTeX.
Usage:  python save_plots.py
Output: figures/*.pdf
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from simulation import CramerLundbergModel
from estimation import (
    estimate_ruin_probability,
    compute_convergence,
)
from intervals import wilson_ci
import plots

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Model parameters (match your defaults) ──────────────────────────
u = 50
lambda_ = 1.0
theta = 0.2
claim_dist = "Exponential"
claim_params = {"mu": 1.0}
claim_mean = 1.0
c = CramerLundbergModel.compute_premium_rate(lambda_, claim_mean, theta)
n_paths = 50_000
T = 100
seed = 42


def save(fig, name):
    path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.write_image(path, format="pdf", width=900, height=500)
    print(f"  Saved {path}")


# ── 1. Sample paths ─────────────────────────────────────────────────
print("1/8  Sample paths")
model = CramerLundbergModel(u, c, lambda_, claim_dist, claim_params)
sim = model.simulate_paths(n_paths, T, seed=seed)
fig = plots.plot_sample_paths(sim, n_display=50, T=T)
save(fig, "sample_paths")

# ── 2. Ruin probability vs surplus ──────────────────────────────────
print("2/8  Ruin probability vs surplus")
u_vals = np.linspace(0, 2 * u, 20)
psi_hats, ci_lo, ci_hi = [], [], []
for i, u_val in enumerate(u_vals):
    m = CramerLundbergModel(u_val, c, lambda_, claim_dist, claim_params)
    r = m.simulate_paths(n_paths, T, seed=seed + i)
    est = estimate_ruin_probability(r["ruin_occurred"])
    psi_hats.append(est["psi_hat"])
    ci_lo.append(est["wilson_ci"][0])
    ci_hi.append(est["wilson_ci"][1])
fig = plots.plot_ruin_probability_vs_surplus(
    u_vals, np.array(psi_hats), np.array(ci_lo), np.array(ci_hi)
)
save(fig, "ruin_prob_vs_surplus")

# ── 3. Convergence ──────────────────────────────────────────────────
print("3/8  Convergence")
conv = compute_convergence(sim["ruin_occurred"], n_points=50)
fig = plots.plot_convergence(conv)
save(fig, "convergence")

# ── 4. Ruin time distribution ──────────────────────────────────────
print("4/8  Ruin time distribution")
fig = plots.plot_ruin_time_distribution(sim["tau"], sim["ruin_occurred"])
save(fig, "ruin_time_dist")

# ── 5. Claim size distribution ──────────────────────────────────────
print("5/8  Claim size distribution")
fig = plots.plot_claim_size_distribution(claim_dist, claim_params)
save(fig, "claim_size_dist")

# ── 6. Sensitivity heatmap ─────────────────────────────────────────
print("6/8  Sensitivity heatmap")
from utils import get_claim_mean
theta_vals = np.linspace(0.1, 0.5, 6)
u_vals_sens = np.linspace(10, 100, 6)
mu = get_claim_mean(claim_dist, claim_params)
psi_mat = np.zeros((len(u_vals_sens), len(theta_vals)))
for j, th in enumerate(theta_vals):
    c_th = CramerLundbergModel.compute_premium_rate(lambda_, mu, th)
    for i, u_s in enumerate(u_vals_sens):
        m = CramerLundbergModel(u_s, c_th, lambda_, claim_dist, claim_params)
        r = m.simulate_paths(2000, T, seed=seed + i * 1000 + j)
        psi_mat[i, j] = np.mean(r["ruin_occurred"])
fig = plots.plot_sensitivity_heatmap(theta_vals, u_vals_sens, psi_mat)
save(fig, "sensitivity_heatmap")

# ── 7. CI comparison (Wald vs Wilson) ──────────────────────────────
print("7/8  CI comparison")
est = estimate_ruin_probability(sim["ruin_occurred"])
fig = plots.plot_ci_comparison(est["wald_ci"], est["wilson_ci"], est["psi_hat"])
save(fig, "ci_comparison")

# ── 8. Minimum surplus search ──────────────────────────────────────
print("8/9  Minimum surplus search (99.5% Wilson CI)")
u_search = np.linspace(0, 200, 40)
psi_s, ci_lo_s, ci_hi_s = [], [], []
for i, u_val in enumerate(u_search):
    m = CramerLundbergModel(u_val, c, lambda_, claim_dist, claim_params)
    r = m.simulate_paths(n_paths, T, seed=seed + i)
    p = float(np.mean(r["ruin_occurred"]))
    L, U = wilson_ci(p, n_paths, confidence=0.995)
    psi_s.append(p)
    ci_lo_s.append(L)
    ci_hi_s.append(U)
psi_s = np.array(psi_s)
ci_lo_s = np.array(ci_lo_s)
ci_hi_s = np.array(ci_hi_s)
fig = plots.plot_min_surplus_search(u_search, psi_s, ci_lo_s, ci_hi_s, threshold=0.005)
save(fig, "min_surplus_search")

# ── 9. Zoomed intersection ────────────────────────────────────────
print("9/9  Minimum surplus search — zoomed intersection")
fig = plots.plot_min_surplus_zoomed(u_search, psi_s, ci_lo_s, ci_hi_s, threshold=0.005)
save(fig, "min_surplus_zoomed")

print(f"\nDone — all plots saved to {FIGURES_DIR}/")
