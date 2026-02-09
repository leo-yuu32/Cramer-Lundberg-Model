"""
All visualisations using Plotly only. Consistent colour palette.
"""

import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Colour palette
COLOR_SURVIVING = "rgba(99, 110, 250, 0.15)"
COLOR_RUINED = "rgba(239, 85, 59, 0.25)"
COLOR_BOUND = "#00CC96"
COLOR_EXACT = "#AB63FA"
COLOR_CI_BAND = "rgba(99, 110, 250, 0.2)"
COLOR_RUIN_LINE = "darkgrey"


def _default_layout(fig, title=None, x_title=None, y_title=None, height=420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=40, t=50, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def plot_sample_paths(sim_results, n_display=50, T=100):
    """Plot up to n_display paths, coloured by ruin status; horizontal line at U=0."""
    t_list = sim_results["t"]
    U_t_list = sim_results["U_t"]
    ruin_occurred = sim_results["ruin_occurred"]
    n_show = min(n_display, len(t_list))
    fig = go.Figure()
    # Ruined paths (no legend)
    for i in range(n_show):
        if ruin_occurred[i]:
            fig.add_trace(
                go.Scatter(
                    x=t_list[i], y=U_t_list[i], mode="lines",
                    line=dict(color=COLOR_RUINED, width=0.8), showlegend=False,
                )
            )
    # Surviving paths (no legend)
    for i in range(n_show):
        if not ruin_occurred[i]:
            fig.add_trace(
                go.Scatter(
                    x=t_list[i], y=U_t_list[i], mode="lines",
                    line=dict(color=COLOR_SURVIVING, width=0.8), showlegend=False,
                )
            )
    # Legend placeholders
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=COLOR_RUINED, width=2), name="Ruin")
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=COLOR_SURVIVING, width=2), name="Surviving")
    )
    # U=0 line
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_RUIN_LINE)
    n_ruin = np.sum(ruin_occurred[:n_show])
    fig.add_annotation(
        text=f"Ruin: {n_ruin}/{n_show}",
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
        font=dict(size=12),
    )
    return _default_layout(fig, x_title="Time t", y_title="Surplus U(t)")


def plot_ruin_probability_vs_surplus(u_values, psi_estimates, ci_lowers, ci_uppers):
    """Line plot with Monte Carlo estimate and confidence interval band."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=u_values, y=psi_estimates, mode="lines+markers", name="ψ̂(u)",
            line=dict(color="rgb(99, 110, 250)", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([u_values, u_values[::-1]]),
            y=np.concatenate([ci_uppers, ci_lowers[::-1]]),
            fill="toself", fillcolor=COLOR_CI_BAND, line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        )
    )
    fig.update_yaxes(type="log")
    return _default_layout(fig, x_title="Initial surplus u", y_title="Ruin probability ψ(u)")


def plot_convergence(convergence_data):
    """Running estimate with CI band showing convergence as sample size increases."""
    fig = go.Figure()
    s = convergence_data["sample_sizes"]
    e = convergence_data["estimates"]
    cl = convergence_data["ci_lower"]
    cu = convergence_data["ci_upper"]
    fig.add_trace(
        go.Scatter(x=s, y=e, mode="lines+markers", name="ψ̂(u)", line=dict(color="rgb(99, 110, 250)", width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([s, s[::-1]]), y=np.concatenate([cu, cl[::-1]]),
            fill="toself", fillcolor=COLOR_CI_BAND, line=dict(color="rgba(255,255,255,0)"), name="95% CI"
        )
    )
    return _default_layout(fig, x_title="Number of simulations", y_title="ψ̂(u)")


def plot_ruin_time_distribution(tau_values, ruin_occurred):
    """Histogram of ruin times τ (conditional on ruin) with KDE and mean line."""
    tau_cond = np.asarray(tau_values)[np.asarray(ruin_occurred)]
    if tau_cond.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No ruin events.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _default_layout(fig, x_title="Ruin time τ", y_title="Count")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=tau_cond, nbinsx=min(40, max(10, tau_cond.size // 5)), name="Ruin times τ"))
    mean_tau = float(np.mean(tau_cond))
    fig.add_vline(x=mean_tau, line_dash="dash", line_color=COLOR_BOUND, annotation_text=f"Mean = {mean_tau:.2f}")
    # KDE
    try:
        kde = stats.gaussian_kde(tau_cond)
        x_kde = np.linspace(tau_cond.min(), tau_cond.max(), 200)
        scale = tau_cond.size * (tau_cond.max() - tau_cond.min()) / 40
        fig.add_trace(
            go.Scatter(
                x=x_kde, y=kde(x_kde) * scale, mode="lines", name="KDE",
                line=dict(color=COLOR_EXACT, width=2),
            )
        )
    except Exception:
        pass
    return _default_layout(fig, x_title="Ruin time τ", y_title="Count")


def plot_claim_size_distribution(claim_dist, claim_params, n_samples=10000):
    """Histogram of sampled claims with theoretical PDF overlay."""
    from utils import get_distribution
    dist = get_distribution(claim_dist, claim_params)
    samples = dist.rvs(size=n_samples, random_state=42)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=samples, nbinsx=50, name="Samples", opacity=0.7,
            histnorm="probability density",
        )
    )
    x_pdf = np.linspace(max(1e-6, samples.min()), np.percentile(samples, 99.5), 300)
    pdf_vals = dist.pdf(x_pdf)
    fig.add_trace(
        go.Scatter(
            x=x_pdf, y=pdf_vals, mode="lines", name="PDF",
            line=dict(color=COLOR_BOUND, width=2),
        )
    )
    mu, var = float(dist.mean()), float(dist.var())
    fig.add_annotation(
        text=f"μ = {mu:.3f}, σ² = {var:.3f}",
        xref="paper", yref="paper", x=0.98, y=0.98, showarrow=False, xanchor="right", yanchor="top",
    )
    return _default_layout(fig, x_title="Claim size", y_title="Density")


def plot_sensitivity_heatmap(theta_values, u_values, psi_matrix, ci_matrix=None):
    """Heatmap: x=θ, y=u, colour=ψ̂. Optional CI matrix for annotations."""
    fig = go.Figure(
        data=go.Heatmap(
            x=theta_values, y=u_values, z=psi_matrix,
            colorscale="Blues", colorbar=dict(title="ψ̂(u)"),
        )
    )
    if psi_matrix.size <= 20 * 20 and ci_matrix is None:
        # Annotate cells if grid small
        fig.update_traces(text=np.round(psi_matrix, 3), texttemplate="%{text}", textfont={"size": 10})
    return _default_layout(fig, x_title="θ (safety loading)", y_title="Initial surplus u", height=450)


def plot_ci_comparison(wald_ci, wilson_ci, psi_hat):
    """Visualise Wald vs Wilson CIs (horizontal intervals)."""
    fig = go.Figure()
    y_wald, y_wilson = 1, 0
    fig.add_trace(
        go.Scatter(
            x=[wald_ci[0], wald_ci[1]], y=[y_wald, y_wald], mode="lines+markers+text",
            line=dict(color="rgb(99, 110, 250)", width=4), name="Wald CI",
            text=["", ""], textposition="top center",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[wilson_ci[0], wilson_ci[1]], y=[y_wilson, y_wilson], mode="lines+markers+text",
            line=dict(color=COLOR_BOUND, width=4), name="Wilson CI",
            text=["", ""], textposition="bottom center",
        )
    )
    fig.add_trace(
        go.Scatter(x=[psi_hat], y=[0.5], mode="markers", marker=dict(size=14, color=COLOR_EXACT, symbol="diamond"),
                   name="ψ̂")
    )
    fig.update_yaxes(tickvals=[0, 1], ticktext=["Wilson", "Wald"], range=[-0.2, 1.2])
    return _default_layout(fig, x_title="Probability", height=280)


def plot_min_surplus_search(u_values, psi_estimates, ci_lowers, ci_uppers, threshold=0.005):
    """Plot ψ̂(u) with 99.5% Wilson CI band and a horizontal threshold line.
    Annotates the minimum u where the upper CI bound drops below threshold."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=u_values, y=psi_estimates, mode="lines+markers", name="ψ̂(u)",
            line=dict(color="rgb(99, 110, 250)", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=u_values, y=ci_uppers, mode="lines", name="Upper 99.5% CI",
            line=dict(color=COLOR_BOUND, width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=u_values, y=ci_lowers, mode="lines", name="Lower 99.5% CI",
            line=dict(color=COLOR_EXACT, width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([u_values, u_values[::-1]]),
            y=np.concatenate([ci_uppers, ci_lowers[::-1]]),
            fill="toself", fillcolor=COLOR_CI_BAND,
            line=dict(color="rgba(255,255,255,0)"),
            name="99.5% CI band", showlegend=False,
        )
    )
    fig.add_hline(
        y=threshold, line_dash="dot", line_color="red", line_width=2,
        annotation_text=f"Threshold = {threshold}",
        annotation_position="top right",
        annotation_font_color="red",
    )
    # Find minimum u where upper CI ≤ threshold
    below = np.where(np.array(ci_uppers) <= threshold)[0]
    if len(below) > 0:
        min_idx = below[0]
        min_u = u_values[min_idx]
        fig.add_vline(
            x=min_u, line_dash="dash", line_color="green", line_width=2,
            annotation_text=f"min u = {min_u:.1f}",
            annotation_position="top left",
            annotation_font_color="green",
        )
        fig.add_trace(
            go.Scatter(
                x=[min_u], y=[ci_uppers[min_idx]], mode="markers",
                marker=dict(size=12, color="green", symbol="star"),
                name=f"min u = {min_u:.1f}",
            )
        )
    return _default_layout(
        fig, title="Minimum Surplus for ψ(u) < 0.005 (99.5% CI)",
        x_title="Initial surplus u", y_title="Ruin probability ψ(u)",
    )


def plot_min_surplus_zoomed(u_values, psi_estimates, ci_lowers, ci_uppers, threshold=0.005):
    """Zoomed view around the intersection where the upper CI crosses the threshold."""
    u_values = np.asarray(u_values)
    ci_uppers = np.asarray(ci_uppers)
    ci_lowers = np.asarray(ci_lowers)
    psi_estimates = np.asarray(psi_estimates)

    below = np.where(ci_uppers <= threshold)[0]
    if len(below) == 0:
        # No crossing found — fall back to full plot
        return plot_min_surplus_search(u_values, psi_estimates, ci_lowers, ci_uppers, threshold)

    cross_idx = below[0]
    # Window: a few points either side of the crossing
    pad = max(3, len(u_values) // 8)
    lo = max(0, cross_idx - pad)
    hi = min(len(u_values), cross_idx + pad + 1)
    sl = slice(lo, hi)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=u_values[sl], y=psi_estimates[sl], mode="lines+markers", name="ψ̂(u)",
            line=dict(color="rgb(99, 110, 250)", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=u_values[sl], y=ci_uppers[sl], mode="lines+markers", name="Upper 99.5% CI",
            line=dict(color=COLOR_BOUND, width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=u_values[sl], y=ci_lowers[sl], mode="lines+markers", name="Lower 99.5% CI",
            line=dict(color=COLOR_EXACT, width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([u_values[sl], u_values[sl][::-1]]),
            y=np.concatenate([ci_uppers[sl], ci_lowers[sl][::-1]]),
            fill="toself", fillcolor=COLOR_CI_BAND,
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
        )
    )
    fig.add_hline(
        y=threshold, line_dash="dot", line_color="red", line_width=2,
        annotation_text=f"Threshold = {threshold}",
        annotation_position="top right",
        annotation_font_color="red",
    )
    min_u = u_values[cross_idx]
    fig.add_vline(
        x=min_u, line_dash="dash", line_color="green", line_width=2,
        annotation_text=f"min u = {min_u:.1f}",
        annotation_position="top left",
        annotation_font_color="green",
    )
    fig.add_trace(
        go.Scatter(
            x=[min_u], y=[ci_uppers[cross_idx]], mode="markers",
            marker=dict(size=14, color="green", symbol="star"),
            name=f"min u = {min_u:.1f}",
        )
    )
    return _default_layout(
        fig, title="Zoomed: Upper CI crossing 0.005 threshold",
        x_title="Initial surplus u", y_title="Ruin probability ψ(u)",
    )
