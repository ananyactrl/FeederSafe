from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from feedersafe.config import AppConfig
from feedersafe.part_b.scoring import run_part_b

st.title("Page 3: Site Feasibility Planner")
data_dir = Path("data/processed")
required = ["feeders.csv", "candidate_sites.csv", "feeder_hourly_risk.csv"]
if not all((data_dir / x).exists() for x in required):
    st.warning("Run the pipeline from Home page first.")
    st.stop()

feeders = pd.read_csv(data_dir / "feeders.csv")
sites = pd.read_csv(data_dir / "candidate_sites.csv")
risk = pd.read_csv(data_dir / "feeder_hourly_risk.csv")
baseline = pd.read_csv(data_dir / "site_results.csv") if (data_dir / "site_results.csv").exists() else None

st.sidebar.header("Operator-adjustable veto thresholds")
cfg = AppConfig(
    dt_headroom_min_pct=st.sidebar.slider("DT headroom min %", 5.0, 40.0, 15.0),
    trench_distance_max_m=st.sidebar.slider("Trench distance max (m)", 50.0, 300.0, 150.0),
    min_width_m=st.sidebar.slider("Min footprint width (m)", 2.0, 6.0, 3.0),
    min_length_m=st.sidebar.slider("Min footprint length (m)", 4.0, 10.0, 6.0),
    min_road_width_m=st.sidebar.slider("Min road width (m)", 3.0, 8.0, 4.5),
    hydrant_clearance_min_m=st.sidebar.slider("Hydrant clearance min (m)", 5.0, 30.0, 15.0),
    phase_imbalance_max_pct=st.sidebar.slider("Phase imbalance max %", 10.0, 50.0, 30.0),
)

result = run_part_b(cfg, feeders, sites, risk).site_results
result.to_csv(data_dir / "site_results_live.csv", index=False)

m = folium.Map(location=[12.97, 77.61], zoom_start=11, tiles="cartodbpositron")
for row in result.itertuples(index=False):
    color = "green" if row.decision == "APPROVED" else "red"
    popup = f"{row.site_id} | {row.decision} | {row.demand_score}/100"
    folium.CircleMarker([row.lat, row.lon], radius=5, color=color, fill=True, popup=popup).add_to(m)
st_folium(m, width=1100, height=480)

site_id = st.selectbox("Select site", sorted(result["site_id"].unique().tolist()))
row = result[result["site_id"] == site_id].iloc[0]
st.write(
    f"**{row['site_id']} | Demand Score: {row['demand_score']}/100 | Decision: {row['decision']}**"
)
st.write(f"Veto reasons: {row['veto_reasons']}")
st.write(f"Nearest feasible alternative: {row['nearest_feasible_alternative']}")

if baseline is not None:
    st.caption("Comparison vs baseline thresholds")
    base_row = baseline[baseline["site_id"] == site_id]
    if not base_row.empty:
        st.write(
            f"Baseline decision: {base_row.iloc[0]['decision']} | Baseline score: {base_row.iloc[0]['demand_score']}"
        )

