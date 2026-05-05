from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.title("Page 4: Coupled Impact Analysis")
data_dir = Path("data/processed")
required = [
    "coupled_impact.csv",
    "coupling_iterations.csv",
    "site_portfolio.csv",
    "feeders.csv",
    "feeder_hourly_risk.csv",
    "site_results.csv",
]
if not all((data_dir / x).exists() for x in required):
    st.warning("Run the pipeline from Home page first.")
    st.stop()

coupled = pd.read_csv(data_dir / "coupled_impact.csv")
iters = pd.read_csv(data_dir / "coupling_iterations.csv")
portfolio = pd.read_csv(data_dir / "site_portfolio.csv")
feeders = pd.read_csv(data_dir / "feeders.csv")
risk = pd.read_csv(data_dir / "feeder_hourly_risk.csv")
sites = pd.read_csv(data_dir / "site_results.csv")

approved = sites[sites["decision"] == "APPROVED"]["site_id"].tolist()
if not approved:
    st.error("No approved sites available under current veto thresholds.")
    st.stop()

st.subheader("Recommended rollout portfolio")
if portfolio.empty:
    st.info("Portfolio is empty for this run. Regenerate pipeline outputs.")
else:
    top_k = st.slider("Top-K recommended sites", min_value=3, max_value=min(30, len(portfolio)), value=min(10, len(portfolio)))
    portfolio_view = portfolio.head(top_k).copy()
    st.dataframe(
        portfolio_view[
            [
                "portfolio_rank",
                "site_id",
                "zone",
                "assigned_feeder_id",
                "portfolio_score",
                "demand_score",
                "mean_delta_capacity_pct",
                "iterations_selected",
                "rollout_priority",
            ]
        ]
    )

site_id = st.selectbox("Select approved site", sorted(approved))
max_iter = int(coupled["iteration"].max()) if not coupled.empty else 1
iteration = st.slider("Iteration", min_value=1, max_value=max_iter, value=max_iter)
impact = coupled[(coupled["site_id"] == site_id) & (coupled["iteration"] == iteration)].copy()

st.subheader("Optimization convergence")
if not iters.empty:
    st.line_chart(iters.set_index("iteration")[["objective_before", "objective_after"]])
    latest = iters.sort_values("iteration").tail(1).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Stressed feeders (before)", int(latest["stressed_before"]))
    c2.metric("Stressed feeders (after)", int(latest["stressed_after"]))
    c3.metric("Converged", "Yes" if bool(latest["converged"]) else "No")

base = risk.groupby("feeder_id", as_index=False).tail(1)[["feeder_id", "status", "capacity_pct"]]
merged = feeders.merge(base, on="feeder_id", how="left").merge(
    impact[["feeder_id", "after_status", "capacity_pct_after"]], on="feeder_id", how="left"
)
merged["after_status"] = merged["after_status"].fillna(merged["status"])

color_map = {"SAFE": "green", "HIGH": "orange", "CRITICAL": "red"}
m = folium.Map(location=[12.97, 77.61], zoom_start=11, tiles="cartodbpositron")
for row in merged.itertuples(index=False):
    popup = f"{row.feeder_id}: {row.status} -> {row.after_status}"
    folium.CircleMarker(
        [row.lat, row.lon], radius=7, color=color_map.get(row.after_status, "blue"), fill=True, popup=popup
    ).add_to(m)
st_folium(m, width=1100, height=500)

st.subheader("Feeder-level impact table")
impact["effect_bucket"] = impact["delta_capacity_pct"].apply(
    lambda x: "Improves" if x < 0 else ("Worsens" if x > 0 else "Unchanged")
)
st.dataframe(
    impact[
        [
            "feeder_id",
            "before_status",
            "after_status",
            "delta_capacity_pct",
            "effect_bucket",
            "objective_before",
            "objective_after",
            "stressed_before",
            "stressed_after",
        ]
    ].sort_values("delta_capacity_pct")
)

