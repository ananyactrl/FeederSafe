from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


st.title("Page 1: Feeder Risk Map")
data_dir = Path("data/processed")
risk_path = data_dir / "feeder_hourly_risk.csv"
feeders_path = data_dir / "feeders.csv"
nudge_path = data_dir / "nudge_recommendations.csv"
cf_path = data_dir / "counterfactuals.csv"

if not risk_path.exists():
    st.warning("Run the pipeline from Home page first.")
    st.stop()

risk = pd.read_csv(risk_path)
feeders = pd.read_csv(feeders_path)
nudges = pd.read_csv(nudge_path) if nudge_path.exists() else pd.DataFrame()
counterfactuals = pd.read_csv(cf_path) if cf_path.exists() else pd.DataFrame()
merged = risk.merge(feeders, on="feeder_id", how="left")

hour = st.slider("Select hour", min_value=18, max_value=23, value=20)
hour_df = merged[merged["hour"] == hour].copy()

color_map = {"SAFE": "green", "HIGH": "orange", "CRITICAL": "red"}
m = folium.Map(location=[12.97, 77.61], zoom_start=11, tiles="cartodbpositron")
for row in hour_df.itertuples(index=False):
    popup = f"{row.feeder_id} | {row.zone} | {row.status} | {row.capacity_pct:.1f}%"
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=8,
        color=color_map.get(row.status, "blue"),
        fill=True,
        fill_opacity=0.75,
        popup=popup,
    ).add_to(m)
st_folium(m, width=1100, height=500)

selected = st.selectbox("Select feeder for detail view", sorted(hour_df["feeder_id"].unique().tolist()))
fdf = hour_df[hour_df["feeder_id"] == selected]
st.line_chart(fdf[["predicted_load_p95_kw", "rated_capacity_kva"]].rename(columns={"rated_capacity_kva": "rated_capacity"}))

st.subheader("Nudge Recommendation")
st.dataframe(nudges[nudges["feeder_id"] == selected][["status", "time_window", "discount_inr_per_kwh", "projected_load_shift_pct", "nudge_text"]])

st.subheader("Counterfactual: How to reduce CRITICAL to HIGH?")
st.dataframe(counterfactuals[counterfactuals["feeder_id"] == selected][["capacity_pct", "users_delay_2h", "counterfactual_text"]])

