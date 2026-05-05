from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Page 2: Home Charging Signature Detection")
data_dir = Path("data/processed")
smart_path = data_dir / "smart_meter.csv"
signal_path = data_dir / "with_without_signal.csv"

if not smart_path.exists():
    st.warning("Run the pipeline from Home page first.")
    st.stop()

smart = pd.read_csv(smart_path, parse_dates=["timestamp"])
signal = pd.read_csv(signal_path, parse_dates=["timestamp"])

feeder_id = st.selectbox("Feeder", sorted(smart["feeder_id"].unique().tolist()))
trace = smart[smart["feeder_id"] == feeder_id].sort_values("timestamp").tail(250)
detected = trace[trace["ev_signature_detected"] == 1]

fig = px.line(trace, x="timestamp", y="voltage_v", title="Synthetic smart-meter voltage trace")
fig.add_scatter(x=detected["timestamp"], y=detected["voltage_v"], mode="markers", name="Detected EV start (voltage sag)")
st.plotly_chart(fig, use_container_width=True)

agg = (
    trace.assign(hour=trace["timestamp"].dt.hour)
    .groupby("hour", as_index=False)["ev_signature_detected"]
    .sum()
    .rename(columns={"ev_signature_detected": "inferred_home_charging_events"})
)
st.bar_chart(agg.set_index("hour"))

st.subheader("With vs Without Home-Charging Signal")
comp = signal[signal["feeder_id"] == feeder_id].copy().tail(200)
st.line_chart(comp.set_index("timestamp")[["predicted_load_p95_kw", "pred_without_signature_kw"]])
st.metric("Average p95 uplift from signal (kW)", f"{comp['delta_kw'].mean():.2f}")

