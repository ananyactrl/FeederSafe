from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_frame(data_key: str, default_path: Path) -> pd.DataFrame:
    state_data = st.session_state.get("data", {})
    if isinstance(state_data, dict) and data_key in state_data:
        return state_data[data_key].copy()
    if default_path.exists():
        return _read_csv_cached(default_path).copy()
    return pd.DataFrame()


st.title("FeederSafe | Feeder Risk Map")
st.caption("All Bengaluru feeders with hour-wise overload risk, capacity headroom, and operator actions.")
st.caption("Bengaluru Masked Grid Network")
st.info(
    "BESCOM operates thousands of distribution transformers in Bengaluru; this map highlights simulated feeder overload risk by hour."
)
data_dir = Path("data/processed")
risk_path = data_dir / "feeder_hourly_risk.csv"
feeders_path = data_dir / "feeders.csv"
nudge_path = data_dir / "nudge_recommendations.csv"
cf_path = data_dir / "counterfactuals.csv"

if not risk_path.exists() or not feeders_path.exists():
    st.warning("Run the pipeline from Home page first.")
    st.stop()

risk = _load_frame("feeder_hourly_risk", risk_path)
feeders = _load_frame("feeders", feeders_path)
nudges = _load_frame("nudge_recommendations", nudge_path)
counterfactuals = _load_frame("counterfactuals", cf_path)

if "rated_capacity_kva" not in risk.columns:
    feeders_cap = feeders[["feeder_id", "rated_capacity_kva"]].drop_duplicates()
    risk = risk.merge(feeders_cap, on="feeder_id", how="left")

# Fallback capacity reconstruction when source column is missing.
if "rated_capacity_kva" not in risk.columns and {"predicted_load_p95_kw", "capacity_pct"} <= set(risk.columns):
    risk["rated_capacity_kva"] = (risk["predicted_load_p95_kw"] / risk["capacity_pct"] * 100).replace([pd.NA], 0)
elif "rated_capacity_kva" in risk.columns and {"predicted_load_p95_kw", "capacity_pct"} <= set(risk.columns):
    missing_cap = risk["rated_capacity_kva"].isna()
    risk.loc[missing_cap, "rated_capacity_kva"] = (
        risk.loc[missing_cap, "predicted_load_p95_kw"] / risk.loc[missing_cap, "capacity_pct"] * 100
    )

hour = st.slider("Select hour", min_value=6, max_value=23, value=20)
risk_hour = risk[risk["hour"] == hour].copy()
hour_df = feeders.merge(
    risk_hour[["feeder_id", "status", "predicted_load_p95_kw", "capacity_pct"]],
    on="feeder_id",
    how="left",
)
hour_df["status"] = hour_df["status"].fillna("SAFE")
hour_df["predicted_load_p95_kw"] = hour_df["predicted_load_p95_kw"].fillna(0.0)
hour_df["capacity_pct"] = hour_df["capacity_pct"].fillna(0.0)
if "rated_capacity_kva" not in hour_df.columns:
    hour_df["rated_capacity_kva"] = pd.NA
if "rated_capacity_kva" in risk_hour.columns:
    risk_cap = risk_hour[["feeder_id", "rated_capacity_kva"]].drop_duplicates()
    hour_df = hour_df.merge(risk_cap, on="feeder_id", how="left", suffixes=("", "_risk"))
    hour_df["rated_capacity_kva"] = hour_df["rated_capacity_kva"].fillna(hour_df["rated_capacity_kva_risk"])
    hour_df = hour_df.drop(columns=["rated_capacity_kva_risk"], errors="ignore")
if {"predicted_load_p95_kw", "capacity_pct"} <= set(hour_df.columns):
    missing_cap = hour_df["rated_capacity_kva"].isna() & (hour_df["capacity_pct"] > 0)
    hour_df.loc[missing_cap, "rated_capacity_kva"] = (
        hour_df.loc[missing_cap, "predicted_load_p95_kw"] / hour_df.loc[missing_cap, "capacity_pct"] * 100
    )

color_map = {"SAFE": "green", "HIGH": "orange", "CRITICAL": "red"}
status_counts = hour_df["status"].value_counts()
prev_hour = max(6, hour - 1)
prev_df = feeders.merge(
    risk[risk["hour"] == prev_hour][["feeder_id", "status"]],
    on="feeder_id",
    how="left",
)
prev_counts = prev_df["status"].fillna("SAFE").value_counts()

c1, c2, c3 = st.columns(3)
c1.metric(
    "CRITICAL Feeders",
    int(status_counts.get("CRITICAL", 0)),
    int(status_counts.get("CRITICAL", 0) - prev_counts.get("CRITICAL", 0)),
)
c2.metric(
    "HIGH Feeders",
    int(status_counts.get("HIGH", 0)),
    int(status_counts.get("HIGH", 0) - prev_counts.get("HIGH", 0)),
)
c3.metric(
    "SAFE Feeders",
    int(status_counts.get("SAFE", 0)),
    int(status_counts.get("SAFE", 0) - prev_counts.get("SAFE", 0)),
)

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

selected = st.selectbox("Select feeder for detail view", sorted(feeders["feeder_id"].unique().tolist()))
fdf = risk[risk["feeder_id"] == selected].sort_values("hour").copy()
if "rated_capacity_kva" not in fdf.columns and {"predicted_load_p95_kw", "capacity_pct"} <= set(fdf.columns):
    fdf["rated_capacity_kva"] = fdf["predicted_load_p95_kw"] / fdf["capacity_pct"] * 100
if "rated_capacity_kva" in fdf.columns and fdf["rated_capacity_kva"].isna().any():
    fdf["rated_capacity_kva"] = fdf["rated_capacity_kva"].fillna(
        fdf["predicted_load_p95_kw"] / fdf["capacity_pct"] * 100
    )

if fdf.empty:
    st.info("No modeled risk points for this feeder; showing as SAFE by default on map.")
else:
    cap_value = float(fdf["rated_capacity_kva"].dropna().iloc[0]) if not fdf["rated_capacity_kva"].dropna().empty else 0.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fdf["hour"],
            y=fdf["predicted_load_p95_kw"],
            mode="lines+markers",
            name="Predicted load p95 (kW)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fdf["hour"],
            y=[cap_value] * len(fdf),
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Rated capacity (kVA)",
        )
    )
    fig.update_layout(
        title=f"Load profile for {selected}",
        xaxis_title="Hour of day",
        yaxis_title="Load / Capacity (kW/kVA)",
    )
    st.plotly_chart(fig, use_container_width=True)

selected_hour_row = hour_df[hour_df["feeder_id"] == selected]
selected_status = selected_hour_row["status"].iloc[0] if not selected_hour_row.empty else "SAFE"
if selected_status in {"HIGH", "CRITICAL"}:
    nudge_rows = nudges[nudges["feeder_id"] == selected]
    if not nudge_rows.empty and "nudge_text" in nudge_rows.columns:
        st.info(f"Nudge recommendation: {nudge_rows.iloc[0]['nudge_text']}")
    else:
        st.info("Nudge recommendation: offer off-peak charging incentives to reduce evening overload.")

if selected_status == "CRITICAL":
    cf_rows = counterfactuals[counterfactuals["feeder_id"] == selected]
    if not cf_rows.empty and "counterfactual_text" in cf_rows.columns:
        st.error(f"Counterfactual: {cf_rows.iloc[0]['counterfactual_text']}")
    else:
        st.error("Counterfactual: delay a share of charging sessions and add nearest vetted public site.")

