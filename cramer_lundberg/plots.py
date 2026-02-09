"""
All visualisations using Plotly only. Consistent colour palette.
"""

import os
import numpy as np
import plotly.graph_objects as go

# Modern colour palette
COLOR_SURVIVING = "rgba(59, 130, 246, 0.15)"    # Soft blue
COLOR_SURVIVING_LINE = "rgba(59, 130, 246, 0.3)" # Blue for path lines
COLOR_RUINED = "rgba(239, 68, 68, 0.2)"          # Soft red
COLOR_RUINED_LINE = "rgba(239, 68, 68, 0.4)"     # Red for ruined path lines
COLOR_CI_BAND = "rgba(59, 130, 246, 0.15)"        # Light blue fill
COLOR_CI_LINE = "rgba(59, 130, 246, 0.6)"         # Blue CI border
COLOR_RUIN_LINE = "#94a3b8"                        # Slate grey for U=0
COLOR_ESTIMATE = "#2563eb"                         # Blue for point estimates
COLOR_ACCENT = "#8b5cf6"                           # Purple accent


def _default_layout(fig, title=None, x_title=None, y_title=None, height=450):
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, font=dict(size=16, color="#1e293b")) if title else None,
        xaxis=dict(
            title=x_title,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#e2e8f0"
        ),
        yaxis=dict(
            title=y_title,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#e2e8f0"
        ),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
        margin=dict(l=60, r=30, t=50, b=60),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def _interpolate_path(t_path, U_path, t_query):
    """Interpolate surplus U(t_query) for a path given its claim times and surpluses."""
    t_path = np.asarray(t_path)
    U_path = np.asarray(U_path)
    t_query = np.asarray(t_query)
    
    # For each query time, find which interval it falls in
    U_query = np.zeros_like(t_query)
    for i, t_q in enumerate(t_query):
        # Find the rightmost time <= t_q
        idx = np.searchsorted(t_path, t_q, side='right') - 1
        idx = max(0, min(idx, len(t_path) - 2))
        
        # Linear interpolation: U(t) = U(t_k) + (U(t_{k+1}) - U(t_k)) * (t - t_k) / (t_{k+1} - t_k)
        t_k = t_path[idx]
        t_kp1 = t_path[idx + 1]
        U_k = U_path[idx]
        U_kp1 = U_path[idx + 1]
        
        if t_kp1 > t_k:
            U_query[i] = U_k + (U_kp1 - U_k) * (t_q - t_k) / (t_kp1 - t_k)
        else:
            U_query[i] = U_k
    
    return U_query


def build_animated_paths(sim_results, n_display=50, n_frames=60, T=100, speed=5, u=0):
    """
    Build a Plotly figure with animation frames that progressively draw surplus paths.
    
    Args:
        sim_results: dict from CramerLundbergModel.simulate_paths()
        n_display: number of paths to show
        n_frames: number of animation frames (more = smoother but slower)
        T: time horizon
        speed: animation speed (1-10), controls frame duration
        u: initial surplus (for annotation)
    
    Returns:
        go.Figure with animation frames
    """
    t_list = sim_results["t"]
    U_t_list = sim_results["U_t"]
    ruin_occurred = sim_results["ruin_occurred"]
    tau = sim_results["tau"]
    
    n_show = min(n_display, len(t_list))
    
    # Calculate global Y-axis range from ALL paths (not just displayed)
    all_surpluses = np.concatenate(U_t_list)
    global_min = float(np.min(all_surpluses))
    global_max = float(np.max(all_surpluses))
    # Add 10% margin on each side
    margin = (global_max - global_min) * 0.1
    y_min = global_min - margin
    y_max = global_max + margin
    # Ensure initial surplus u is visible
    y_min = min(y_min, u - margin)
    y_max = max(y_max, u + margin)
    
    # Uniform time grid for frames
    frame_times = np.linspace(0, T, n_frames)
    
    # Frame duration based on speed: speed 1 = 200ms, speed 5 = 80ms, speed 10 = 20ms
    frame_duration = int(200 / speed)
    
    # Build frames
    frames = []
    for k in range(n_frames):
        t_max = frame_times[k]
        
        # For each path, get points up to t_max
        frame_data = []
        n_ruined_by_frame = 0
        
        for i in range(n_show):
            t_path = np.asarray(t_list[i])
            U_path = np.asarray(U_t_list[i])
            is_ruined = ruin_occurred[i]
            ruin_time = tau[i] if is_ruined else np.inf
            
            # Count if ruined by this frame time
            if is_ruined and ruin_time <= t_max:
                n_ruined_by_frame += 1
            
            # Only show path up to min(t_max, ruin_time)
            t_display = min(t_max, ruin_time) if is_ruined else t_max
            
            # Get points on uniform grid up to t_display
            t_query = frame_times[frame_times <= t_display]
            if len(t_query) == 0:
                t_query = np.array([0.0])
            
            U_query = _interpolate_path(t_path, U_path, t_query)
            
            # Color by ruin status (if ruined and shown past ruin time)
            if is_ruined and t_max >= ruin_time:
                color = COLOR_RUINED_LINE
            else:
                color = COLOR_SURVIVING_LINE
            
            frame_data.append(
                go.Scatter(
                    x=t_query,
                    y=U_query,
                    mode="lines",
                    line=dict(color=color, width=0.8),
                    showlegend=False,
                    name=f"Path {i+1}",
                )
            )
        
        # Add U=0 line (present in every frame)
        frame_data.append(
            go.Scatter(
                x=[0, T],
                y=[0, 0],
                mode="lines",
                line=dict(color=COLOR_RUIN_LINE, dash="dash", width=1.5),
                showlegend=False,
                name="U=0",
            )
        )
        
        # Add initial surplus marker
        frame_data.append(
            go.Scatter(
                x=[0],
                y=[u],
                mode="markers",
                marker=dict(size=8, color="#2563eb", symbol="circle"),
                showlegend=False,
                name="u",
            )
        )
        
        # Add annotation with time and ruin count
        frame_layout = go.Layout(
            annotations=[
                dict(
                    text=f"t = {t_max:.1f} | Paths ruined: {n_ruined_by_frame}/{n_show}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=13, color="#374151"),
                    bgcolor="rgba(255,255,255,0.8)",
                    borderpad=4
                )
            ]
        )
        
        frames.append(go.Frame(data=frame_data, layout=frame_layout, name=f"frame_{k}"))
    
    # Initial data (first frame)
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
    )
    
    # Legend placeholders
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=COLOR_RUINED_LINE, width=2), name="Ruin")
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=COLOR_SURVIVING_LINE, width=2), name="Surviving")
    )
    
    # Layout with animation controls
    fig.update_layout(
        template="plotly_white",
        height=550,
        title="Surplus Process U(t)",
        xaxis_title="Time t",
        yaxis=dict(
            title="Surplus U(t)",
            range=[y_min, y_max],
            fixedrange=True
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.12,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, {
                         "frame": {"duration": frame_duration, "redraw": True},
                         "fromcurrent": True,
                         "transition": {"duration": 0}
                     }]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], {
                         "frame": {"duration": 0, "redraw": False},
                         "mode": "immediate",
                         "transition": {"duration": 0}
                     }])
            ]
        )],
        sliders=[dict(
            active=0,
            steps=[dict(
                method="animate",
                args=[[f"frame_{k}"], {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }],
                label=f"t={frame_times[k]:.0f}"
            ) for k in range(0, n_frames, max(1, n_frames // 20))],
            x=0.05,
            len=0.9,
            xanchor="left",
            y=-0.05,
            currentvalue=dict(
                prefix="Time: ",
                visible=True,
                xanchor="center"
            ),
            transition=dict(duration=0)
        )]
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
                    line=dict(color=COLOR_RUINED_LINE, width=0.8), showlegend=False,
                )
            )
    # Surviving paths (no legend)
    for i in range(n_show):
        if not ruin_occurred[i]:
            fig.add_trace(
                go.Scatter(
                    x=t_list[i], y=U_t_list[i], mode="lines",
                    line=dict(color=COLOR_SURVIVING_LINE, width=0.8), showlegend=False,
                )
            )
    # Legend placeholders
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=COLOR_RUINED_LINE, width=2), name="Ruin")
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=COLOR_SURVIVING_LINE, width=2), name="Surviving")
    )
    # U=0 line
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_RUIN_LINE)
    n_ruin = np.sum(ruin_occurred[:n_show])
    fig.add_annotation(
        text=f"Ruin: {n_ruin}/{n_show}",
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
        font=dict(size=12),
    )
    return _default_layout(fig, title="Surplus Process U(t)", x_title="Time t", y_title="Surplus U(t)", height=550)


def plot_convergence(convergence_data, true_value=None):
    """Running estimate with CI band showing convergence as sample size increases."""
    fig = go.Figure()
    s = convergence_data["sample_sizes"]
    e = convergence_data["estimates"]
    cl = convergence_data["ci_lower"]
    cu = convergence_data["ci_upper"]
    fig.add_trace(
        go.Scatter(x=s, y=e, mode="lines+markers", name="ψ̂(u)", line=dict(color=COLOR_ESTIMATE, width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([s, s[::-1]]), y=np.concatenate([cu, cl[::-1]]),
            fill="toself", fillcolor=COLOR_CI_BAND, line=dict(color="rgba(255,255,255,0)"), name="95% CI"
        )
    )
    if true_value is not None and np.isfinite(true_value):
        fig.add_hline(y=true_value, line_dash="dash", line_color=COLOR_ACCENT, annotation_text="True ψ(u)")
    return _default_layout(fig, title="Convergence of Ruin Probability Estimate", x_title="Number of simulations", y_title="ψ̂(u)", height=300)


def plot_and_save_decay(u_values, psi_values):
    """
    Plot empirical ruin probability decay on semi-log scale and auto-save figure.

    Args:
        u_values: Array of initial surplus values
        psi_values: Array of empirical ruin probabilities

    Returns:
        go.Figure with log-scale Y-axis
    """
    u_values = np.asarray(u_values)
    psi_values = np.asarray(psi_values)

    # Filter out zero probabilities to avoid log(0) issues
    # Use floor of 1e-5 for zero values to allow log plotting
    psi_plot = np.where(psi_values > 0, psi_values, 1e-5)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=u_values,
            y=psi_plot,
            mode="lines+markers",
            name="Empirical ψ̂(u)",
            line=dict(color=COLOR_ESTIMATE, width=2),
            marker=dict(size=6, color=COLOR_ESTIMATE),
        )
    )

    fig.update_yaxes(type="log")

    # Set Y-axis range to avoid log(0) - start slightly above minimum non-zero value
    non_zero_psi = psi_values[psi_values > 0]
    if len(non_zero_psi) > 0:
        y_min = max(1e-5, np.min(non_zero_psi) * 0.5)
        y_max = np.max(psi_values) * 1.5 if np.max(psi_values) > 0 else 1.0
        fig.update_yaxes(range=[y_min, y_max])
    else:
        # Edge case: no ruin at any u - set a default range
        fig.update_yaxes(range=[1e-5, 1.0])

    fig = _default_layout(
        fig,
        title="Empirical Ruin Probability Decay (Semi-Log Scale)",
        x_title="Initial Surplus u",
        y_title="P(Ruin) - Log Scale",
        height=450
    )

    # Auto-save logic: create figures/ directory if needed and save
    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    save_path = os.path.join(figures_dir, "ruin_decay_analysis.png")
    try:
        fig.write_image(save_path, scale=2)
        # Store save status in figure's customdata for UI to check
        fig._saved_successfully = True
    except Exception as e:
        # If kaleido is not installed, skip saving but don't fail
        # Note: Install kaleido with: pip install kaleido
        fig._saved_successfully = False
        fig._save_error = str(e)

    return fig
