"""
dashboard/app.py
=================
Streamlit dashboard for the Wind Turbine Digital Twin.

Design decision: Industrial dark-theme aesthetic — this is SCADA-style
monitoring software used by engineers in control rooms. The UI should
feel precise, data-dense, and serious. Think: offshore wind farm ops centre.

Architecture: The dashboard runs the DigitalTwin in session state,
adding N new steps each refresh cycle. Streamlit's auto-rerun + st.empty()
gives a live-updating feel without a separate backend process.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from datetime import datetime

from twin_engine.twin import DigitalTwin
from data_pipeline.schemas import TwinSnapshot
from rag_assistant.anomaly_context import build_anomaly_event, format_anomaly_summary

# ─── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wind Turbine Digital Twin",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
  }

  .stApp {
    background-color: #080d13;
    color: #c8d8e8;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #0d1520;
    border-right: 1px solid #1e3048;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1a27 0%, #111f30 100%);
    border: 1px solid #1e3a54;
    border-radius: 4px;
    padding: 12px 16px;
  }

  [data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #4a7fa0 !important;
    text-transform: uppercase;
  }

  [data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem !important;
    color: #7ecfff !important;
  }

  [data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem !important;
  }

  /* Section headers */
  .section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #2a6080;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3048;
    padding-bottom: 6px;
    margin: 16px 0 12px 0;
  }

  /* Anomaly alert */
  .anomaly-alert {
    background: linear-gradient(90deg, #2d0a0a 0%, #1a0808 100%);
    border-left: 3px solid #ff3333;
    border-radius: 2px;
    padding: 10px 14px;
    margin: 4px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #ff8888;
  }

  .normal-status {
    background: linear-gradient(90deg, #0a1f10 0%, #081510 100%);
    border-left: 3px solid #22cc66;
    border-radius: 2px;
    padding: 10px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #44ee88;
  }

  /* Title */
  .dashboard-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem;
    color: #7ecfff;
    letter-spacing: 0.1em;
  }

  .dashboard-subtitle {
    font-family: 'Barlow', sans-serif;
    font-size: 0.8rem;
    color: #3a6a8a;
    letter-spacing: 0.05em;
    margin-top: -8px;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ────────────────────────────────────────────────────────────────

PLOT_BG = "#080d13"
GRID_COLOR = "#1a2a3a"
TEXT_COLOR = "#7090a0"
ACCENT_BLUE = "#7ecfff"
ACCENT_GREEN = "#22dd66"
ACCENT_RED = "#ff4444"
ACCENT_AMBER = "#ffaa33"

PLOT_LAYOUT = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="Share Tech Mono", color=TEXT_COLOR, size=11),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
    margin=dict(l=48, r=16, t=32, b=32),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR, borderwidth=1),
)

# ─── Session state ───────────────────────────────────────────────────────────────

def init_state():
    if "twin" not in st.session_state:
        st.session_state.twin = DigitalTwin(
            config_path=os.path.join(os.path.dirname(__file__), "..", "config", "turbine_config.yaml")
        )
        st.session_state.snapshots = []
        st.session_state.anomaly_log = []
        st.session_state.running = False
        st.session_state.step_count = 0

init_state()

# ─── Sidebar controls ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="dashboard-title">⚡ TWIN CTRL</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">NREL 5MW Reference Turbine</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Simulation Control</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ RUN", use_container_width=True, type="primary")
    with col2:
        stop_btn = st.button("⏹ STOP", use_container_width=True)

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    batch_size = st.slider("Steps per refresh", 5, 50, 10)
    max_history = st.slider("Chart history (steps)", 100, 1200, 400)
    refresh_ms = st.slider("Refresh interval (ms)", 200, 2000, 500)

    st.markdown("---")
    st.markdown('<div class="section-header">Turbine Config</div>', unsafe_allow_html=True)
    twin = st.session_state.twin
    cfg = twin.physics.config
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#4a7fa0; line-height:1.9;">
    RATED POWER &nbsp;&nbsp; {cfg.rated_power_kw:.0f} kW<br>
    CUT-IN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {cfg.cut_in_speed_ms} m/s<br>
    RATED WIND &nbsp;&nbsp; {cfg.rated_speed_ms} m/s<br>
    CUT-OUT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {cfg.cut_out_speed_ms} m/s<br>
    ROTOR DIA &nbsp;&nbsp;&nbsp; {cfg.rotor_diameter_m} m<br>
    Cp &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {cfg.power_coefficient_cp}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">ML Detector</div>', unsafe_allow_html=True)
    detector = twin.anomaly_detector
    trained_pct = int(detector.training_progress * 100)
    st.progress(trained_pct / 100, text=f"Training: {trained_pct}%")
    if detector.is_trained:
        st.markdown(
            '<span style="color:#22dd66; font-family:Share Tech Mono; font-size:0.75rem;">● ONLINE — Isolation Forest</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#ffaa33; font-family:Share Tech Mono; font-size:0.75rem;">◌ WARMING UP — Z-score fallback</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("🔄 Reset Twin", use_container_width=True):
        for key in ["twin", "snapshots", "anomaly_log", "running", "step_count"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ─── Main dashboard ──────────────────────────────────────────────────────────────

# Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<div class="dashboard-title">🌬️ WIND TURBINE DIGITAL TWIN</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Real-time physics model vs sensor comparison · Anomaly Detection · Fault Analysis</div>', unsafe_allow_html=True)

with col_status:
    ts_str = datetime.utcnow().strftime("%H:%M:%S UTC")
    if st.session_state.running:
        st.markdown(f'<div style="text-align:right; font-family:Share Tech Mono; font-size:0.75rem; color:#22dd66; margin-top:8px;">● LIVE &nbsp;&nbsp; {ts_str}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align:right; font-family:Share Tech Mono; font-size:0.75rem; color:#ffaa33; margin-top:8px;">⏸ PAUSED &nbsp; {ts_str}</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── Advance simulation ──────────────────────────────────────────────────────────

if st.session_state.running:
    new_snaps = st.session_state.twin.run_batch(batch_size)
    st.session_state.snapshots.extend(new_snaps)
    st.session_state.step_count += batch_size

    # Log anomaly events
    for snap in new_snaps:
        if snap.is_anomaly:
            event = build_anomaly_event(snap)
            summary = format_anomaly_summary(event)
            st.session_state.anomaly_log.insert(0, summary)
            if len(st.session_state.anomaly_log) > 30:
                st.session_state.anomaly_log = st.session_state.anomaly_log[:30]

# Trim history
snaps = st.session_state.snapshots[-max_history:]

# ─── KPI Row ────────────────────────────────────────────────────────────────────

if snaps:
    latest = snaps[-1]
    prev = snaps[-2] if len(snaps) > 1 else latest

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("WIND SPEED", f"{latest.wind_speed_ms:.1f} m/s",
                  delta=f"{latest.wind_speed_ms - prev.wind_speed_ms:+.2f}")
    with k2:
        st.metric("EXPECTED PWR", f"{latest.expected_power_kw:.0f} kW")
    with k3:
        color_delta = latest.actual_power_kw - latest.expected_power_kw
        st.metric("ACTUAL PWR", f"{latest.actual_power_kw:.0f} kW",
                  delta=f"{color_delta:+.0f} kW")
    with k4:
        st.metric("DEVIATION", f"{latest.deviation_pct:+.1f}%",
                  delta=f"{latest.deviation_pct - prev.deviation_pct:+.1f}pp")
    with k5:
        eff_pct = latest.efficiency_ratio * 100
        st.metric("EFFICIENCY", f"{eff_pct:.1f}%",
                  delta=f"{(latest.efficiency_ratio - prev.efficiency_ratio)*100:+.1f}pp")
    with k6:
        total_steps = st.session_state.step_count
        anomaly_count = sum(1 for s in st.session_state.snapshots if s.is_anomaly)
        anom_rate = (anomaly_count / total_steps * 100) if total_steps > 0 else 0.0
        st.metric("ANOMALY RATE", f"{anom_rate:.1f}%", delta=f"{total_steps} steps")
else:
    st.info("▶ Press RUN to start the simulation")
    st.stop()

# ─── Anomaly status banner ───────────────────────────────────────────────────────

if latest.is_anomaly:
    fault_label = latest.fault_type.replace("_", " ").upper() if latest.fault_type else "UNKNOWN"
    st.markdown(
        f'<div class="anomaly-alert">🔴 ANOMALY DETECTED · {fault_label} · '
        f'Score: {latest.anomaly_score:.3f} · '
        f'Deviation: {latest.deviation_pct:+.1f}%</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="normal-status">✔ NOMINAL OPERATION — All parameters within expected bounds</div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ─── Main charts ────────────────────────────────────────────────────────────────

df = pd.DataFrame([{
    "time": s.timestamp,
    "wind_speed": s.wind_speed_ms,
    "expected_power": s.expected_power_kw,
    "actual_power": s.actual_power_kw,
    "deviation_pct": s.deviation_pct,
    "efficiency": s.efficiency_ratio * 100,
    "is_anomaly": s.is_anomaly,
    "anomaly_score": s.anomaly_score,
    "fault_type": s.fault_type or "Normal",
} for s in snaps])

chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    st.markdown('<div class="section-header">Expected vs Actual Power Output</div>', unsafe_allow_html=True)

    fig = go.Figure()

    # Expected power (physics model)
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["expected_power"],
        name="Expected (Physics)",
        line=dict(color=ACCENT_BLUE, width=1.5, dash="dot"),
        hovertemplate="%{y:.0f} kW<extra>Expected</extra>",
    ))

    # Actual power
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["actual_power"],
        name="Actual (Sensor)",
        line=dict(color=ACCENT_GREEN, width=1.5),
        hovertemplate="%{y:.0f} kW<extra>Actual</extra>",
    ))

    # Fill between — red where actual < expected
    fig.add_trace(go.Scatter(
        x=pd.concat([df["time"], df["time"][::-1]]),
        y=pd.concat([df["expected_power"], df["actual_power"][::-1]]),
        fill="toself",
        fillcolor="rgba(255, 80, 80, 0.07)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Deviation zone",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Anomaly markers
    anom_df = df[df["is_anomaly"]]
    if not anom_df.empty:
        fig.add_trace(go.Scatter(
            x=anom_df["time"], y=anom_df["actual_power"],
            mode="markers",
            marker=dict(color=ACCENT_RED, size=6, symbol="x"),
            name="Anomaly",
            hovertemplate="ANOMALY · %{y:.0f} kW<extra></extra>",
        ))

    fig.update_layout(
        **PLOT_LAYOUT,
        height=280,
        yaxis_title="Power (kW)",
        yaxis_range=[0, twin.physics.config.rated_power_kw * 1.05],
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.markdown('<div class="section-header">Power Curve · Expected</div>', unsafe_allow_html=True)

    curve = twin.physics.power_curve_table(v_min=0, v_max=28, steps=140)
    cv_df = pd.DataFrame(curve, columns=["wind_ms", "power_kw"])

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=cv_df["wind_ms"], y=cv_df["power_kw"],
        fill="tozeroy",
        fillcolor="rgba(126,207,255,0.08)",
        line=dict(color=ACCENT_BLUE, width=2),
        name="Power curve",
        hovertemplate="%{x:.1f} m/s → %{y:.0f} kW<extra></extra>",
    ))

    # Current operating point
    if snaps:
        fig_curve.add_trace(go.Scatter(
            x=[latest.wind_speed_ms], y=[latest.actual_power_kw],
            mode="markers",
            marker=dict(
                color=ACCENT_RED if latest.is_anomaly else ACCENT_GREEN,
                size=12, symbol="circle",
                line=dict(color="white", width=1.5)
            ),
            name="Current",
            hovertemplate=f"{latest.wind_speed_ms:.1f} m/s · {latest.actual_power_kw:.0f} kW<extra>Now</extra>",
        ))

    fig_curve.update_layout(
        **PLOT_LAYOUT,
        height=280,
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Power (kW)",
    )
    st.plotly_chart(fig_curve, use_container_width=True)

# ─── Row 2: Deviation + Anomaly Score ───────────────────────────────────────────

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown('<div class="section-header">Power Deviation %</div>', unsafe_allow_html=True)

    fig_dev = go.Figure()
    fig_dev.add_hline(y=0, line_dash="dot", line_color=ACCENT_BLUE, opacity=0.4)
    fig_dev.add_hline(y=-10, line_dash="dash", line_color=ACCENT_AMBER, opacity=0.4,
                      annotation_text="−10% threshold", annotation_position="bottom right",
                      annotation_font_color=ACCENT_AMBER, annotation_font_size=9)

    colors = [ACCENT_RED if v < -10 else (ACCENT_AMBER if v < 0 else ACCENT_GREEN)
              for v in df["deviation_pct"]]
    fig_dev.add_trace(go.Bar(
        x=df["time"], y=df["deviation_pct"],
        marker_color=colors, name="Deviation %",
        hovertemplate="%{y:+.1f}%<extra></extra>",
    ))
    fig_dev.update_layout(**PLOT_LAYOUT, height=220, yaxis_title="Deviation (%)")
    st.plotly_chart(fig_dev, use_container_width=True)

with row2_col2:
    st.markdown('<div class="section-header">ML Anomaly Score</div>', unsafe_allow_html=True)

    fig_score = go.Figure()
    fig_score.add_hline(y=0.5, line_dash="dash", line_color=ACCENT_RED, opacity=0.5,
                        annotation_text="Decision boundary", annotation_position="top right",
                        annotation_font_color=ACCENT_RED, annotation_font_size=9)

    score_colors = [ACCENT_RED if s > 0.5 else ACCENT_BLUE for s in df["anomaly_score"]]
    fig_score.add_trace(go.Scatter(
        x=df["time"], y=df["anomaly_score"],
        fill="tozeroy",
        fillcolor="rgba(126,207,255,0.06)",
        line=dict(color=ACCENT_BLUE, width=1.5),
        name="Anomaly Score",
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    fig_score.update_layout(**PLOT_LAYOUT, height=220, yaxis_title="Score", yaxis_range=[0, 1])
    st.plotly_chart(fig_score, use_container_width=True)

# ─── Row 3: Fault distribution + event log ───────────────────────────────────────

row3_col1, row3_col2 = st.columns([1, 2])

with row3_col1:
    st.markdown('<div class="section-header">Fault Type Distribution</div>', unsafe_allow_html=True)
    fault_counts = df["fault_type"].value_counts().reset_index()
    fault_counts.columns = ["fault", "count"]
    palette = {
        "Normal": ACCENT_GREEN,
        "efficiency_drop": ACCENT_RED,
        "vibration_spike": ACCENT_AMBER,
        "overheating": "#ff6688",
    }
    fig_pie = go.Figure(go.Pie(
        labels=fault_counts["fault"],
        values=fault_counts["count"],
        marker_colors=[palette.get(f, "#555") for f in fault_counts["fault"]],
        textfont_size=10,
        hole=0.5,
        hovertemplate="%{label}: %{value} steps (%{percent})<extra></extra>",
    ))
    fig_pie.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Share Tech Mono", color=TEXT_COLOR, size=11),
        height=220,
        showlegend=True,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with row3_col2:
    st.markdown('<div class="section-header">Anomaly Event Log</div>', unsafe_allow_html=True)
    if st.session_state.anomaly_log:
        for entry in st.session_state.anomaly_log[:8]:
            st.markdown(f'<div class="anomaly-alert">{entry}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="normal-status">No anomalies detected yet.</div>', unsafe_allow_html=True)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────────

if st.session_state.running:
    time.sleep(refresh_ms / 1000.0)
    st.rerun()
