"""
Cramér-Lundberg Ruin Simulator — Streamlit app.
Main entry point: layout, tabs, sidebar.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from simulation import CramerLundbergModel
from utils import get_claim_mean, format_ci
from estimation import (
    estimate_ruin_probability,
    estimate_ruin_time_stats,
    estimate_deficit_stats,
    compute_convergence,
)
from intervals import wilson_ci
import plots

st.set_page_config(
    page_title="Cramér-Lundberg Simulator",
    layout="wide",
    page_icon="📊",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    div[data-testid="stMetric"] { background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)


def get_claim_params_from_sidebar(claim_dist):
    """Build claim_params dict from sidebar inputs for selected distribution."""
    if claim_dist == "Exponential":
        return {"mu": st.session_state.get("exp_mu", 1.0)}
    if claim_dist == "Gamma":
        return {"alpha": st.session_state.get("gamma_alpha", 2.0), "beta": st.session_state.get("gamma_beta", 1.0)}
    if claim_dist == "Lognormal":
        return {"mu_log": st.session_state.get("ln_mu", 0.0), "sigma_log": st.session_state.get("ln_sigma", 1.0)}
    if claim_dist == "Pareto":
        return {"alpha": st.session_state.get("pareto_alpha", 2.5), "x_m": st.session_state.get("pareto_xm", 1.0)}
    return {"mu": 1.0}


def _hashable_params(p):
    return tuple(sorted((k, float(v)) if isinstance(v, (int, float, np.floating)) else (k, v) for k, v in p.items()))


@st.cache_data(show_spinner=False)
def run_simulation(u, c, lambda_, claim_dist, claim_params_hashable, n_paths, T, seed):
    """Cached simulation run keyed by all inputs."""
    claim_params = dict(claim_params_hashable)
    model = CramerLundbergModel(u, c, lambda_, claim_dist, claim_params)
    return model.simulate_paths(n_paths, T, seed=seed)


@st.cache_data(show_spinner=False)
def run_u_sweep(u_values, c, lambda_, claim_dist, claim_params_hashable, n_paths, T, seed):
    """Run simulations at each u in u_values; return psi estimates and CIs."""
    claim_params = dict(claim_params_hashable)
    psi_hats, ci_lowers, ci_uppers = [], [], []
    for i, u in enumerate(u_values):
        model = CramerLundbergModel(u, c, lambda_, claim_dist, claim_params)
        res = model.simulate_paths(n_paths, T, seed=seed + i)
        est = estimate_ruin_probability(res["ruin_occurred"])
        psi_hats.append(est["psi_hat"])
        ci_lowers.append(est["wilson_ci"][0])
        ci_uppers.append(est["wilson_ci"][1])
    return np.array(psi_hats), np.array(ci_lowers), np.array(ci_uppers)


@st.cache_data(show_spinner=False)
def run_min_surplus_sweep(u_values, c, lambda_, claim_dist, claim_params_hashable, n_paths, T, seed):
    """Sweep over u values, returning psi_hat and 99.5% Wilson CIs from intervals.py."""
    claim_params = dict(claim_params_hashable)
    psi_hats, ci_lowers, ci_uppers = [], [], []
    for i, u_val in enumerate(u_values):
        model = CramerLundbergModel(u_val, c, lambda_, claim_dist, claim_params)
        res = model.simulate_paths(n_paths, T, seed=seed + i)
        psi_hat = float(np.mean(res["ruin_occurred"]))
        L, U = wilson_ci(psi_hat, n_paths, confidence=0.995)
        psi_hats.append(psi_hat)
        ci_lowers.append(L)
        ci_uppers.append(U)
    return np.array(psi_hats), np.array(ci_lowers), np.array(ci_uppers)


@st.cache_data(show_spinner=False)
def run_sensitivity_sweep(theta_values, u_values, lambda_, claim_dist, claim_params_hashable, n_paths, T, seed):
    """Sensitivity heatmap: psi over (theta, u) grid. Use smaller n_paths for speed."""
    claim_params = dict(claim_params_hashable)
    psi_matrix = np.zeros((len(u_values), len(theta_values)))
    mu = get_claim_mean(claim_dist, claim_params)
    for j, theta in enumerate(theta_values):
        c = CramerLundbergModel.compute_premium_rate(lambda_, mu, theta)
        for i, u in enumerate(u_values):
            model = CramerLundbergModel(u, c, lambda_, claim_dist, claim_params)
            res = model.simulate_paths(n_paths, T, seed=seed + i * 1000 + j)
            psi_matrix[i, j] = np.mean(res["ruin_occurred"])
    return psi_matrix


# ----- Sidebar -----
st.sidebar.title("📐 Model parameters")
st.sidebar.markdown("**Surplus process** $U(t) = u + ct - \\sum_{i=1}^{N(t)} X_i$")

u = st.sidebar.slider("Initial surplus $u$", 0, 200, 50, 1)

premium_mode = st.sidebar.radio("Premium", ["Specify safety loading θ", "Specify premium rate c"], index=0, key="premium_mode_v2")

lam = st.sidebar.number_input("Claim intensity λ", min_value=0.01, value=1.0, step=0.1, format="%.2f")
claim_dist = st.sidebar.selectbox("Claim distribution", ["Exponential", "Gamma", "Lognormal", "Pareto"])

c1, c2 = st.sidebar.columns(2)
with c1:
    if claim_dist == "Exponential":
        st.number_input("Mean μ", key="exp_mu", min_value=0.01, value=1.0, step=0.1, format="%.2f")
    elif claim_dist == "Gamma":
        st.number_input("Shape α", key="gamma_alpha", min_value=0.01, value=2.0, step=0.1, format="%.2f")
        st.number_input("Rate β", key="gamma_beta", min_value=0.01, value=1.0, step=0.1, format="%.2f")
    elif claim_dist == "Lognormal":
        st.number_input("μ_log", key="ln_mu", value=0.0, step=0.1, format="%.2f")
        st.number_input("σ_log", key="ln_sigma", min_value=0.01, value=1.0, step=0.1, format="%.2f")
    elif claim_dist == "Pareto":
        st.number_input("Shape α", key="pareto_alpha", min_value=1.01, value=2.5, step=0.1, format="%.2f")
        st.number_input("x_m", key="pareto_xm", min_value=0.01, value=1.0, step=0.1, format="%.2f")
with c2:
    if claim_dist == "Gamma":
        pass  # second column used for beta above
    elif claim_dist == "Lognormal":
        pass
    elif claim_dist == "Pareto":
        pass

claim_params = get_claim_params_from_sidebar(claim_dist)
claim_mean = get_claim_mean(claim_dist, claim_params)

if premium_mode == "Specify safety loading θ":
    theta_input = st.sidebar.number_input("Safety loading θ", min_value=-0.99, value=0.2, step=0.05, format="%.2f")
    c = CramerLundbergModel.compute_premium_rate(lam, claim_mean, theta_input)
else:
    c = st.sidebar.number_input("Premium rate $c$", min_value=0.01, value=float((1 + 0.2) * lam * claim_mean), step=0.1, format="%.2f")
    theta_input = (c / (lam * claim_mean)) - 1.0 if lam * claim_mean > 0 else 0.0

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation settings**")
n_paths = st.sidebar.slider("Number of paths $N$", 100, 50000, 5000, 100)
T = st.sidebar.slider("Time horizon $T$", 10, 500, 100, 10)
seed = st.sidebar.number_input("Random seed (optional)", value=42, min_value=0, step=1)

run_clicked = st.sidebar.button("▶ Run simulation", type="primary", use_container_width=True)

# Net profit condition check
if c > lam * claim_mean:
    st.sidebar.success("✅ Net profit condition $c > \\lambda\\mu$ satisfied.")
else:
    st.sidebar.warning("⚠️ Net profit condition violated ($c \\leq \\lambda\\mu$). Ruin probability will be high.")

# ----- Main (after run or default) -----
if run_clicked:
    with st.spinner("Running simulation…"):
        st.session_state["sim_results"] = run_simulation(u, c, lam, claim_dist, _hashable_params(claim_params), n_paths, T, seed)
        st.session_state["sim_params"] = (u, c, lam, claim_dist, claim_params, n_paths, T, seed, theta_input, claim_mean)
if "sim_results" not in st.session_state:
    st.info("Adjust parameters and click **▶ Run simulation** in the sidebar.")
    st.stop()

sim_results = st.session_state["sim_results"]
if "sim_params" in st.session_state:
    u, c, lam, claim_dist, claim_params, n_paths, T, seed, theta_input, claim_mean = st.session_state["sim_params"]

est = estimate_ruin_probability(sim_results["ruin_occurred"])
time_stats = estimate_ruin_time_stats(sim_results["tau"], sim_results["ruin_occurred"])
deficit_stats = estimate_deficit_stats(sim_results["deficit_at_ruin"], sim_results["ruin_occurred"])

# Metrics row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Ruin probability ψ̂(u)", f"{est['psi_hat']:.4f}", delta=format_ci(est["wilson_ci"][0], est["wilson_ci"][1]))
with m2:
    E_tau = time_stats["E_tau"]
    delta_tau = format_ci(time_stats["ci_tau"][0], time_stats["ci_tau"][1]) if E_tau is not None else "—"
    st.metric("Mean ruin time E[τ]", f"{E_tau:.2f}" if E_tau is not None else "—", delta=delta_tau)
with m3:
    median_tau = time_stats["median_tau"]
    st.metric("Median ruin time", f"{median_tau:.2f}" if median_tau is not None else "—", delta=None)
with m4:
    mean_def = deficit_stats["mean_deficit"]
    delta_def = format_ci(deficit_stats["ci_deficit"][0], deficit_stats["ci_deficit"][1]) if mean_def is not None else "—"
    st.metric("Mean deficit |U(τ)|", f"{mean_def:.2f}" if mean_def is not None else "—", delta=delta_def)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Sample paths", "🎯 Ruin probability", "📉 Convergence", "⏱️ Ruin times", "📊 Claims", "🔬 Sensitivity", "🔍 Min surplus",
])

with tab1:
    n_display = st.slider("Paths to display", 10, 200, 50, 10, key="n_display")
    fig = plots.plot_sample_paths(sim_results, n_display=n_display, T=T)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📐 Mathematical background"):
        st.latex(r"U(t) = u + ct - \sum_{i=1}^{N(t)} X_i")
        st.markdown("Ruin occurs when $U(t) < 0$ for some $t$. Between claims, surplus is linear in $t$.")

with tab2:
    u_sweep_max = 2 * u
    u_vals = np.linspace(0, max(1, u_sweep_max), 20)
    with st.spinner("Sweeping over u…"):
        psi_hats, ci_lo, ci_hi = run_u_sweep(u_vals, c, lam, claim_dist, _hashable_params(claim_params), n_paths, T, seed)
    fig = plots.plot_ruin_probability_vs_surplus(u_vals, psi_hats, ci_lo, ci_hi)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Wald vs Wilson CI** (Wilson preferred near 0/1)")
    fig_ci = plots.plot_ci_comparison(est["wald_ci"], est["wilson_ci"], est["psi_hat"])
    st.plotly_chart(fig_ci, use_container_width=True)
    with st.expander("📐 Mathematical background"):
        st.markdown("The ruin probability $\\psi(u)$ is the probability that the surplus process ever becomes negative, starting from initial surplus $u$.")
        st.latex(r"\psi(u) = P(U(t) < 0 \text{ for some } t \geq 0 \mid U(0) = u)")
        st.markdown("Monte Carlo estimation: $\\hat{\\psi}(u) = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{1}_{\\{\\text{ruin in path } i\\}}$.")

with tab3:
    conv = compute_convergence(sim_results["ruin_occurred"], n_points=50)
    fig = plots.plot_convergence(conv)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📐 Mathematical background"):
        st.markdown("As the number of simulated paths $N \\to \\infty$, the Monte Carlo estimate $\\hat{\\psi}(u) \\to \\psi(u)$ by the law of large numbers.")
        st.markdown("The plot shows the running estimate with 95% confidence intervals (Wilson) as $N$ increases.")
        st.markdown("Wilson CI has better coverage than Wald CI near boundaries (0 and 1).")

with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_t = plots.plot_ruin_time_distribution(sim_results["tau"], sim_results["ruin_occurred"])
        st.plotly_chart(fig_t, use_container_width=True)
    with col_b:
        tau_cond = sim_results["tau"][sim_results["ruin_occurred"]]
        if tau_cond.size > 0:
            def_cond = sim_results["deficit_at_ruin"][sim_results["ruin_occurred"]]
            fig_d = go.Figure()
            fig_d.add_trace(go.Histogram(x=def_cond, nbinsx=min(30, max(10, len(def_cond) // 5)), name="Deficit at ruin"))
            fig_d.add_vline(x=np.mean(def_cond), line_dash="dash", line_color=plots.COLOR_BOUND, annotation_text="Mean")
            plots._default_layout(fig_d, x_title="Deficit |U(τ)|", y_title="Count")
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No ruin events — no deficit distribution.")
    with st.expander("📐 Mathematical background"):
        st.markdown("Distribution of ruin time $\\tau$ and deficit $|U(\\tau)|$ conditional on ruin.")

with tab5:
    fig_c = plots.plot_claim_size_distribution(claim_dist, claim_params, n_samples=10000)
    st.plotly_chart(fig_c, use_container_width=True)
    with st.expander("📐 Mathematical background"):
        st.markdown("Claim sizes $X_i$ are i.i.d. from the chosen distribution. Mean $\\mu$ drives net profit condition.")

with tab6:
    st.caption("Heatmap over (θ, u). Uses smaller N per cell; progress below.")
    theta_min, theta_max = st.slider("θ range", 0.05, 1.0, (0.1, 0.5), 0.05, key="theta_range")
    u_min, u_max = st.slider("u range", 0, 200, (10, 100), 5, key="u_range")
    n_theta = st.slider("θ grid size", 3, 15, 6, key="n_theta")
    n_u = st.slider("u grid size", 3, 15, 6, key="n_u")
    n_paths_sens = st.slider("Paths per cell (sensitivity)", 500, 5000, 2000, 500, key="n_sens")
    theta_vals = np.linspace(theta_min, theta_max, n_theta)
    u_vals_sens = np.linspace(u_min, u_max, n_u)
    progress = st.progress(0)
    try:
        psi_mat = run_sensitivity_sweep(theta_vals, u_vals_sens, lam, claim_dist, _hashable_params(claim_params), n_paths_sens, T, seed)
        progress.progress(1.0)
        fig_h = plots.plot_sensitivity_heatmap(theta_vals, u_vals_sens, psi_mat)
        st.plotly_chart(fig_h, use_container_width=True)
    except Exception as e:
        progress.progress(1.0)
        st.error(str(e))
    with st.expander("📐 Mathematical background"):
        st.markdown("Ruin probability $\\hat{\\psi}(u)$ over safety loading $\\theta$ and initial surplus $u$.")
        st.latex(r"c = (1+\\theta)\\lambda\\mu")

with tab7:
    st.subheader("Find minimum initial surplus for ψ(u) < 0.005")
    st.markdown("Sweeps over $u$ values and computes **99.5% Wilson CI** using `intervals.py`. "
                "The minimum $u$ is where the **upper** CI bound drops below 0.005.")
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        u_search_min = st.number_input("u min", value=0, min_value=0, step=5, key="u_search_min")
    with col_u2:
        u_search_max = st.number_input("u max", value=max(200, 4 * u), min_value=1, step=10, key="u_search_max")
    with col_u3:
        u_search_n = st.slider("Number of u points", 10, 100, 40, key="u_search_n")
    u_search_vals = np.linspace(u_search_min, u_search_max, u_search_n)
    with st.spinner("Running minimum surplus sweep (99.5% Wilson CI)…"):
        psi_s, ci_lo_s, ci_hi_s = run_min_surplus_sweep(
            u_search_vals, c, lam, claim_dist,
            _hashable_params(claim_params), n_paths, T, seed,
        )
    fig_min = plots.plot_min_surplus_search(u_search_vals, psi_s, ci_lo_s, ci_hi_s, threshold=0.005)
    st.plotly_chart(fig_min, use_container_width=True)
    # Report the result
    below_threshold = np.where(ci_hi_s <= 0.005)[0]
    if len(below_threshold) > 0:
        min_u_found = u_search_vals[below_threshold[0]]
        st.success(f"**Minimum u ≈ {min_u_found:.1f}** — the upper 99.5% Wilson CI drops below 0.005 here.")
    else:
        st.warning("Upper CI bound did not drop below 0.005 in the searched range. Try increasing u max.")
    with st.expander("📐 Mathematical background"):
        st.markdown("We use the Wilson score interval at 99.5% confidence from `intervals.py`:")
        st.latex(r"\tilde{p} = \frac{\hat{\psi} + z^2/(2N)}{1 + z^2/N}, \quad "
                 r"CI = \tilde{p} \pm \frac{z\sqrt{\tilde{p}(1-\tilde{p})/(1+z^2/N)}}{1+z^2/N}")
        st.markdown("We find the smallest $u$ such that the **upper bound** of this CI is $\\leq 0.005$, "
                     "giving us statistical confidence that the ruin probability is below 0.005.")
