from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
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


def _apply_veto_thresholds(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    reasons = []
    for row in out.itertuples(index=False):
        row_reasons: list[str] = []
        if float(getattr(row, "dt_headroom_pct", 100.0)) <= thresholds["dt_headroom_min_pct"]:
            row_reasons.append(
                f"DT headroom {float(getattr(row, 'dt_headroom_pct', 0.0)):.1f}% - minimum required {thresholds['dt_headroom_min_pct']:.1f}%"
            )
        if float(getattr(row, "trench_distance_m", 0.0)) >= thresholds["trench_distance_max_m"]:
            row_reasons.append(
                f"Trench distance {float(getattr(row, 'trench_distance_m', 0.0)):.0f}m - maximum allowed {thresholds['trench_distance_max_m']:.0f}m"
            )
        if float(getattr(row, "clear_width_m", 0.0)) < thresholds["min_width_m"] or float(
            getattr(row, "clear_length_m", 0.0)
        ) < thresholds["min_length_m"]:
            row_reasons.append(
                f"Footprint {float(getattr(row, 'clear_width_m', 0.0)):.1f}m x {float(getattr(row, 'clear_length_m', 0.0)):.1f}m - minimum required {thresholds['min_width_m']:.0f}m x {thresholds['min_length_m']:.0f}m"
            )
        if float(getattr(row, "road_width_m", 100.0)) <= thresholds["min_road_width_m"]:
            row_reasons.append(
                f"Road width {float(getattr(row, 'road_width_m', 0.0)):.1f}m - minimum required {thresholds['min_road_width_m']:.1f}m"
            )
        if float(getattr(row, "hydrant_distance_m", 100.0)) <= thresholds["hydrant_clearance_min_m"]:
            row_reasons.append(
                f"Hydrant clearance {float(getattr(row, 'hydrant_distance_m', 0.0)):.1f}m - minimum required {thresholds['hydrant_clearance_min_m']:.1f}m"
            )
        if float(getattr(row, "phase_imbalance_pct", 0.0)) >= thresholds["phase_imbalance_max_pct"]:
            row_reasons.append(
                f"Phase imbalance {float(getattr(row, 'phase_imbalance_pct', 0.0)):.0f}% - maximum allowed {thresholds['phase_imbalance_max_pct']:.0f}%"
            )
        reasons.append("; ".join(row_reasons) if row_reasons else "APPROVED")
    out["veto_reasons"] = reasons
    out["decision"] = out["veto_reasons"].apply(lambda x: "APPROVED" if x == "APPROVED" else "REJECTED")
    return out


st.title("FeederSafe | Site Feasibility Planner")
st.caption("Interactive siting feasibility with live veto threshold tuning on saved candidate-site results.")
st.info(
    "For BESCOM deployment teams, this screen mirrors practical siting checks so only field-feasible EV sites move forward."
)
data_dir = Path("data/processed")
required = ["feeders.csv", "site_results.csv"]
if not all((data_dir / x).exists() for x in required):
    st.warning("Run the pipeline from Home page first.")
    st.stop()

feeders = _load_frame("feeders", data_dir / "feeders.csv")
baseline = _load_frame("site_results", data_dir / "site_results.csv")
if baseline.empty:
    st.warning("No site results found. Run the pipeline from Home page first.")
    st.stop()

st.sidebar.header("Operator-adjustable veto thresholds")
thresholds = {
    "dt_headroom_min_pct": st.sidebar.slider("DT headroom min %", 5.0, 40.0, 15.0),
    "trench_distance_max_m": st.sidebar.slider("Trench distance max (m)", 50.0, 300.0, 150.0),
    "min_width_m": st.sidebar.slider("Min footprint width (m)", 2.0, 6.0, 3.0),
    "min_length_m": st.sidebar.slider("Min footprint length (m)", 4.0, 10.0, 6.0),
    "min_road_width_m": st.sidebar.slider("Min road width (m)", 3.0, 8.0, 4.5),
    "hydrant_clearance_min_m": st.sidebar.slider("Hydrant clearance min (m)", 5.0, 30.0, 15.0),
    "phase_imbalance_max_pct": st.sidebar.slider("Phase imbalance max %", 10.0, 50.0, 30.0),
}
result = _apply_veto_thresholds(baseline, thresholds)
filtered_df = result.copy()

if st.button("Apply Veto Thresholds to Full Pipeline (simulated)"):
    approved_sites = int((filtered_df["decision"] == "APPROVED").sum())
    st.success(
        f"✅ Thresholds applied to all 200 candidate sites. {approved_sites} sites approved. "
        "(In production, this triggers a full pipeline re-run.)"
    )
    st.cache_data.clear()

m = folium.Map(location=[12.97, 77.61], zoom_start=11, tiles="cartodbpositron")
for row in result.itertuples(index=False):
    approved = row.decision == "APPROVED"
    popup = (
        f"Site: {row.site_id}<br>"
        f"Demand score: {row.demand_score}<br>"
        f"Decision: {row.decision}<br>"
        f"Veto reasons: {row.veto_reasons}<br>"
        f"Nearest feasible alternative: {row.nearest_feasible_alternative}"
    )
    folium.CircleMarker(
        [row.lat, row.lon],
        radius=6,
        color="green" if approved else "red",
        fill=approved,
        fill_color="green",
        fill_opacity=0.85 if approved else 0.0,
        weight=2,
        popup=popup,
    ).add_to(m)
st_folium(m, width=1100, height=480)

site_id = st.selectbox("Select site", sorted(result["site_id"].unique().tolist()))
row = result[result["site_id"] == site_id].iloc[0]
st.write(
    f"**{row['site_id']} | Demand Score: {row['demand_score']}/100 | Decision: {row['decision']}**"
)
st.write(f"Veto reasons: {row['veto_reasons']}")
st.write(f"Nearest feasible alternative: {row['nearest_feasible_alternative']}")

st.caption("Table: site-level decisions after applying the current veto sliders.")
st.dataframe(result[["site_id", "zone", "demand_score", "decision", "veto_reasons", "nearest_feasible_alternative"]])

approved_count = int((result["decision"] == "APPROVED").sum())
rejected_count = int((result["decision"] == "REJECTED").sum())
st.markdown(f"**Summary:** {approved_count} sites APPROVED | {rejected_count} sites REJECTED")

reasons = result[result["decision"] == "REJECTED"]["veto_reasons"].str.split("; ").explode()
reasons = reasons[reasons.notna() & (reasons != "")]
if not reasons.empty:
    reasons_df = reasons.value_counts().rename_axis("reason").reset_index(name="count")
    fig = px.bar(
        reasons_df,
        x="reason",
        y="count",
        title="Top rejection reasons",
        labels={"reason": "Veto reason", "count": "Rejected sites"},
    )
    st.plotly_chart(fig, use_container_width=True)

