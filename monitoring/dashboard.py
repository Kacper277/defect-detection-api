"""
Simple data drift monitoring dashboard for Defect Detection API.
Usage: streamlit run monitoring/dashboard.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.config import CLASSES, CHECKPOINT_DIR

LOG_DIR = Path(__file__).parent / "logs"
BASELINE_FILE = CHECKPOINT_DIR / "baseline_distribution.json"

st.set_page_config(
    page_title="Defect Detection - Drift Monitoring",
    page_icon="",
    layout="wide",
)

st.title("Defect Detection API - Data Drift Monitoring")

# --- Sidebar filters ---
st.sidebar.header("Filters")

last_n = st.sidebar.slider("Last N predictions", 50, 2000, 500, step=50)
refresh = st.sidebar.button("Refresh data")

if refresh:
    st.cache_data.clear()


# --- Data loading ---
@st.cache_data(ttl=30)
def load_predictions(n: int = 500):
    """Load the last N predictions from JSONL files."""
    log_files = sorted(LOG_DIR.glob("predictions_*.jsonl"))
    if not log_files:
        return pd.DataFrame()

    records = []
    for log_file in reversed(log_files):
        with open(log_file) as f:
            for line in f:
                records.append(json.loads(line.strip()))
                if len(records) >= n:
                    break
        if len(records) >= n:
            break

    return pd.DataFrame(records)


def load_baseline():
    """Load the baseline distribution."""
    if not BASELINE_FILE.exists():
        return None
    with open(BASELINE_FILE) as f:
        return json.load(f)


def load_alerts():
    """Load drift alerts."""
    alerts_file = LOG_DIR / "drift_alerts.jsonl"
    if not alerts_file.exists():
        return []
    alerts = []
    with open(alerts_file) as f:
        for line in f:
            alerts.append(json.loads(line.strip()))
    return alerts


# --- Data ---
df = load_predictions(last_n)
baseline = load_baseline()
alerts = load_alerts()

if df.empty:
    st.warning("No monitoring data available. Run the API and send some /predict requests.")
    st.stop()

# --- Top metrics ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Prediction count", len(df))

with col2:
    avg_conf = df["confidence"].mean() if "confidence" in df.columns else 0
    st.metric("Average confidence", f"{avg_conf:.2%}")

with col3:
    drift_count = len(alerts)
    st.metric("Drift alerts", drift_count,
              delta=None if drift_count == 0 else f"{drift_count}")

with col4:
    if baseline:
        st.metric("Baseline (mean pixel)", f"{baseline.get('mean_mean', 0):.1f}")
    else:
        st.metric("Baseline", "missing")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Class distribution", "Confidence", "Pixel drift", "Alerts"])

with tab1:
    st.subheader("Predicted class distribution")
    if "predicted_class" in df.columns:
        class_counts = df["predicted_class"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]

        fig = px.bar(
            class_counts,
            x="Class",
            y="Count",
            color="Class",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"Class distribution (last {last_n} predictions)",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

        # Percentage shares
        st.write("**Percentage shares:**")
        total = class_counts["Count"].sum()
        for _, row in class_counts.iterrows():
            st.write(f"- {row['Class']}: {row['Count']} ({(row['Count'] / total * 100):.1f}%)")
    else:
        st.info("No class data available.")

with tab2:
    st.subheader("Prediction confidence over time")
    if "timestamp" in df.columns and "confidence" in df.columns:
        df_time = df.copy()
        df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])
        df_time = df_time.sort_values("timestamp")

        fig = px.line(
            df_time,
            x="timestamp",
            y="confidence",
            color="predicted_class",
            title="Prediction confidence",
            labels={"confidence": "Confidence", "timestamp": "Time"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold 50%")
        st.plotly_chart(fig, width='stretch')

        # Histogram
        fig2 = px.histogram(
            df_time,
            x="confidence",
            nbins=30,
            title="Confidence histogram",
            color="predicted_class",
        )
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("No confidence/timestamp data.")

with tab3:
    st.subheader("Mean pixel - comparison with baseline")
    if "mean_pixel" in df.columns and baseline:
        df_time = df.copy()
        df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])
        df_time = df_time.sort_values("timestamp")

        baseline_mean = baseline["mean_mean"]
        baseline_std = baseline["mean_std"]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_time["timestamp"],
            y=df_time["mean_pixel"],
            mode="markers",
            name="Predictions",
            marker=dict(size=4, opacity=0.6),
        ))

        # Baseline +/- 3 sigma zone
        fig.add_hline(
            y=baseline_mean,
            line_color="green",
            annotation_text=f"Baseline mean: {baseline_mean:.1f}",
        )
        fig.add_hrect(
            y0=baseline_mean - 3 * baseline_std,
            y1=baseline_mean + 3 * baseline_std,
            fillcolor="green",
            opacity=0.1,
            annotation_text="+/-3 sigma (normal range)",
        )

        fig.update_layout(
            title="Average pixel of input images",
            xaxis_title="Time",
            yaxis_title="Mean pixel",
        )
        st.plotly_chart(fig, width='stretch')

        # Statistics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Baseline mean", f"{baseline_mean:.2f} +/- {baseline_std:.2f}")
        with col_b:
            current_mean = df["mean_pixel"].mean()
            st.metric("Current mean", f"{current_mean:.2f}",
                      delta=f"{current_mean - baseline_mean:+.2f}")
    else:
        st.info("No mean pixel data or baseline missing.")

with tab4:
    st.subheader("Drift alerts")
    if alerts:
        st.warning(f"{len(alerts)} drift alerts detected!")

        alerts_df = pd.DataFrame(alerts)
        if "timestamp" in alerts_df.columns:
            alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"])
            alerts_df = alerts_df.sort_values("timestamp", ascending=False)

        st.dataframe(alerts_df, width='stretch')

        # Alerts over time
        if "timestamp" in alerts_df.columns and "z_score_mean_pixel" in alerts_df.columns:
            fig = px.scatter(
                alerts_df,
                x="timestamp",
                y="z_score_mean_pixel",
                title="Z-score of mean pixel (alerts)",
                color=alerts_df["z_score_mean_pixel"].abs() > 3,
                color_discrete_map={True: "red", False: "orange"},
            )
            fig.add_hline(y=3, line_dash="dash", line_color="red")
            fig.add_hline(y=-3, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')
    else:
        st.success("No drift alerts - data consistent with baseline.")

# --- Footer ---
st.divider()
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
           f"Logs: `{LOG_DIR}` | Baseline: `{BASELINE_FILE}`")