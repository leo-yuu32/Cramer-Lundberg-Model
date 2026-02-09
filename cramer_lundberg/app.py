"""
Cramér-Lundberg Ruin Simulator — Streamlit app.
Single-page interactive simulation with animated Monte Carlo visualization.
"""

import streamlit as st
import numpy as np
from simulation import CramerLundbergModel
from utils import get_claim_mean, format_ci
from estimation import (
    estimate_ruin_probability,
    estimate_ruin_time_stats,
    estimate_deficit_stats,
    compute_convergence,
    bootstrap_ci,
)
from estimation import analyze_ruin_decay
import plots

st.set_page_config(
    page_title="Cramér-Lundberg Simulator",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Tighten main content padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Style metric cards - dark mode compatible */
    [data-testid="stMetric"] {
        background-color: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Metric container background */
    div[data-testid="metric-container"] {
        background-color: transparent !important;
    }
    
    /* Slider styling - dark mode compatible */
    .stSlider {
        background-color: transparent !important;
    }
    
    .stSlider > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    
    /* Divider spacing */
    hr {
        margin: 0.75rem 0;
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Make plotly chart container cleaner */
    .stPlotlyChart {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
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




# ----- Sidebar -----
st.sidebar.markdown("## ⚙️ Simulation Parameters")

# Preset selector (placeholder for now)
preset = st.sidebar.selectbox("📋 Preset", ["Custom"], key="preset")

st.sidebar.divider()

st.sidebar.markdown("**Model Parameters**")

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

claim_params = get_claim_params_from_sidebar(claim_dist)
claim_mean = get_claim_mean(claim_dist, claim_params)

if premium_mode == "Specify safety loading θ":
    theta_input = st.sidebar.number_input("Safety loading θ", min_value=-0.99, value=0.2, step=0.05, format="%.2f")
    c = CramerLundbergModel.compute_premium_rate(lam, claim_mean, theta_input)
else:
    c = st.sidebar.number_input("Premium rate $c$", min_value=0.01, value=float((1 + 0.2) * lam * claim_mean), step=0.1, format="%.2f")
    theta_input = (c / (lam * claim_mean)) - 1.0 if lam * claim_mean > 0 else 0.0

st.sidebar.divider()

st.sidebar.markdown("**Simulation Settings**")
n_paths = st.sidebar.slider("Number of paths $N$", 100, 50000, 5000, 100)
T = st.sidebar.slider("Time horizon $T$", 10, 500, 100, 10)
seed = st.sidebar.number_input("Random seed (optional)", value=42, min_value=0, step=1)

st.sidebar.divider()

st.sidebar.markdown("**Animation Settings**")
anim_speed = st.sidebar.slider("Animation speed", 1, 10, 5, 1, help="Speed 1 = slow, 10 = fast")
n_display = st.sidebar.slider("Paths to display", 10, 200, 50, 10)

st.sidebar.divider()

run_button = st.sidebar.button("▶ Run Simulation", type="primary", use_container_width=True)

# Net profit condition check
if c > lam * claim_mean:
    st.sidebar.success("✅ Net profit condition $c > \\lambda\\mu$ satisfied.")
else:
    st.sidebar.warning("⚠️ Net profit condition violated ($c \\leq \\lambda\\mu$). Ruin probability will be high.")


# ----- Title -----
st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <h1 style="font-size: 2.4rem; font-weight: 700; margin-bottom: 0.3rem;">
        Cramér-Lundberg Ruin Simulator
    </h1>
    <p style="font-size: 1.1rem; color: #6b7280; margin-top: 0;">
        Monte Carlo simulation of the classical insurance surplus process
    </p>
</div>
""", unsafe_allow_html=True)


# ----- Run simulation -----
if run_button:
    with st.spinner("Running simulation…"):
        st.session_state["sim_results"] = run_simulation(u, c, lam, claim_dist, _hashable_params(claim_params), n_paths, T, seed)
        st.session_state["sim_params"] = (u, c, lam, claim_dist, claim_params, n_paths, T, seed, theta_input, claim_mean)

# ----- Display results or placeholder -----
if "sim_results" not in st.session_state:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 0; color: #94a3b8;">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">📉</p>
        <p style="font-size: 1.1rem;">Configure parameters in the sidebar and click <strong>Run Simulation</strong> to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

sim_results = st.session_state["sim_results"]
if "sim_params" in st.session_state:
    u, c, lam, claim_dist, claim_params, n_paths, T, seed, theta_input, claim_mean = st.session_state["sim_params"]

# ----- Main chart -----
col_chart, col_toggle = st.columns([5, 1])
with col_toggle:
    show_animation = st.toggle("🎬 Animate", value=True)

if show_animation:
    fig = plots.build_animated_paths(sim_results, n_display=n_display, n_frames=60, T=T, speed=anim_speed, u=u)
else:
    fig = plots.plot_sample_paths(sim_results, n_display=n_display, T=T)

st.plotly_chart(fig, use_container_width=True, key="main_chart")

# ----- Metrics row -----
est = estimate_ruin_probability(sim_results["ruin_occurred"])
time_stats = estimate_ruin_time_stats(sim_results["tau"], sim_results["ruin_occurred"])
deficit_stats = estimate_deficit_stats(sim_results["deficit_at_ruin"], sim_results["ruin_occurred"])

n_ruined = int(np.sum(sim_results["ruin_occurred"]))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Ruin Probability ψ̂(u)", f"{est['psi_hat']:.4f}",
              delta=f"95% CI: {format_ci(est['wilson_ci'][0], est['wilson_ci'][1])}")
with col2:
    st.metric("Paths Ruined", f"{n_ruined:,} / {n_paths:,}")
with col3:
    E_tau = time_stats["E_tau"]
    if E_tau is not None:
        st.metric("Mean Ruin Time E[τ]", f"{E_tau:.1f}",
                  delta=f"95% CI: {format_ci(time_stats['ci_tau'][0], time_stats['ci_tau'][1])}")
    else:
        st.metric("Mean Ruin Time E[τ]", "—")
with col4:
    mean_def = deficit_stats.get("mean_deficit")
    if mean_def is not None:
        st.metric("Mean Deficit |U(τ)|", f"{mean_def:.2f}",
                  delta=f"95% CI: {format_ci(deficit_stats['ci_deficit'][0], deficit_stats['ci_deficit'][1])}")
    else:
        st.metric("Mean Deficit |U(τ)|", "—")

# ----- Detailed Statistics (collapsible) -----
with st.expander("📊 Detailed Statistics"):
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Ruin Probability Confidence Intervals**")
        st.code(f"Wald 95% CI:   {format_ci(est['wald_ci'][0], est['wald_ci'][1])}\n"
                f"Wilson 95% CI: {format_ci(est['wilson_ci'][0], est['wilson_ci'][1])}")
    with col_right:
        st.markdown("**Ruin Time Confidence Intervals**")
        if E_tau is not None:
            # Bootstrap CI for ruin time
            tau_cond = sim_results["tau"][sim_results["ruin_occurred"]]
            try:
                boot_ci = bootstrap_ci(tau_cond, np.mean, n_bootstrap=2000)
                st.code(f"CLT 95% CI:       {format_ci(time_stats['ci_tau'][0], time_stats['ci_tau'][1])}\n"
                        f"Bootstrap 95% CI: {format_ci(boot_ci[0], boot_ci[1])}")
            except Exception:
                st.code(f"CLT 95% CI: {format_ci(time_stats['ci_tau'][0], time_stats['ci_tau'][1])}")
        else:
            st.code("No ruin events — no ruin time statistics.")
    
    # Convergence mini-plot
    st.markdown("**Convergence of ψ̂(u)**")
    convergence_data = compute_convergence(sim_results["ruin_occurred"], n_points=50)
    fig_conv = plots.plot_convergence(convergence_data)
    fig_conv.update_layout(height=300)
    st.plotly_chart(fig_conv, use_container_width=True)

# ----- Advanced Decay Analysis -----
with st.expander("📊 Advanced Decay Analysis"):
    st.markdown("**Empirical Ruin Decay Analysis**")
    st.markdown("Demonstrate that ruin probability decays exponentially with initial surplus $u$.")

    col_decay1, col_decay2 = st.columns(2)
    with col_decay1:
        u_max_decay = st.slider(
            "Max $u$ for analysis",
            min_value=10,
            max_value=500,
            value=int(u * 2) if u > 0 else 100,
            step=10,
            key="u_max_decay"
        )
    with col_decay2:
        u_steps_decay = st.slider(
            "Number of points",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            key="u_steps_decay"
        )

    run_decay_btn = st.button(
        "Run Decay Analysis & Save Figure",
        type="secondary",
        use_container_width=True
    )

    if run_decay_btn:
        with st.spinner(
            "Computing ruin probabilities across u range (this may take a moment)..."
        ):
            try:
                # Use smaller N=2000 for faster computation
                n_paths_decay = 2000
                u_range = np.linspace(0, u_max_decay, u_steps_decay)
                u_decay, psi_decay = analyze_ruin_decay(
                    u_range, c, lam, claim_dist, claim_params,
                    n_paths_decay, T, seed
                )

                fig_decay = plots.plot_and_save_decay(u_decay, psi_decay)
                st.plotly_chart(fig_decay, use_container_width=True)

                # Check if save was successful
                if hasattr(fig_decay, "_saved_successfully") and fig_decay._saved_successfully:
                    st.success(
                        "Figure saved to `figures/ruin_decay_analysis.png`"
                    )
                else:
                    st.warning(
                        "Figure displayed but could not be saved. "
                        "Install `kaleido` package for image export: `pip install kaleido`"
                    )

                st.markdown("""
                **Interpretation:** On a semi-logarithmic scale, an exponential decay appears as a straight line.
                The empirical Monte Carlo data demonstrates that $\\psi(u)$ decays approximately exponentially with $u$,
                consistent with theoretical predictions (e.g., Lundberg's inequality $\\psi(u) \\leq e^{-Ru}$).
                """)
            except Exception as e:
                st.error(f"Error generating decay analysis: {str(e)}")
    else:
        st.info(
            "Click **Run Decay Analysis & Save Figure** to compute ruin "
            "probabilities across a range of initial surplus values. "
            "The figure will be automatically saved to `figures/ruin_decay_analysis.png`."
        )

# ----- Mathematical Background -----
with st.expander("📐 Mathematical Background"):
    st.markdown(r"""
    **The Surplus Process**
    
    $$U(t) = u + ct - S(t)$$
    
    where $S(t) = \sum_{i=1}^{N(t)} X_i$ is the aggregate claims process, $N(t)$ is a Poisson process with rate $\lambda$, and $X_i$ are i.i.d. claim sizes. The premium rate is $c$, and $u$ is the initial surplus.
    
    **Ruin Probability**
    
    $$\psi(u) = \mathbb{P}(\inf_{t \geq 0} U(t) < 0 \mid U(0) = u)$$
    
    Ruin occurs when the surplus process first becomes negative.
    
    **Monte Carlo Estimation**
    
    $$\hat{\psi}(u) = \frac{1}{n} \sum_{k=1}^{n} \mathbf{1}\{U^{(k)}(t) < 0 \text{ for some } t \leq T\}$$
    
    where $U^{(k)}$ are $n$ independent simulated paths.
    
    **Wilson Confidence Interval** (preferred for proportions near 0 or 1):
    
    $$\tilde{p} = \frac{n\hat{\psi} + z^2/2}{n + z^2}, \quad \text{CI} = \tilde{p} \pm z\sqrt{\frac{\tilde{p}(1-\tilde{p})}{n + z^2}}$$
    
    where $z = 1.96$ for 95% confidence.
    
    **Net Profit Condition**
    
    For the insurer to be viable, we require $c > \lambda\mu$ where $\mu = \mathbb{E}[X_i]$ is the mean claim size. If this condition is violated, ruin is certain in the long run.
    
    **Exponential Decay of Ruin Probability**
    
    Under the net profit condition, ruin probability decays exponentially with initial surplus:
    
    $$\psi(u) \sim Ce^{-Ru} \quad \text{as } u \to \infty$$
    
    where $R$ is the adjustment coefficient. On a semi-logarithmic plot ($\log \psi(u)$ vs $u$), this appears as a straight line with slope $-R$.
    """)
