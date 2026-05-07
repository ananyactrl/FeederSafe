from __future__ import annotations
from pathlib import Path
import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

# ---- NO CACHING - reads fresh CSV every time ----
def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

st.title("FeederSafe | Coupled Impact Analysis")
st.caption("Before/after feeder stress after placing approved sites and simulating coupled grid relief.")
st.info("BESCOM planners can use this coupled view to prioritize sites that reduce transformer stress across the network, not just locally.")

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

coupled = _read_csv(data_dir / "coupled_impact.csv")
iters = _read_csv(data_dir / "coupling_iterations.csv")
portfolio = _read_csv(data_dir / "site_portfolio.csv")
feeders = _read_csv(data_dir / "feeders.csv")
risk = _read_csv(data_dir / "feeder_hourly_risk.csv")
sites = _read_csv(data_dir / "site_results.csv")

approved = sites[sites["decision"] == "APPROVED"]["site_id"].tolist()
if not approved:
    st.error("No approved sites available under current veto thresholds.")
    st.stop()

# ---- Portfolio Section ----
st.subheader("Recommended rollout portfolio")
top_k = 0
if portfolio.empty:
    st.info("Portfolio is empty for this run. Regenerate pipeline outputs.")
else:
    n = len(portfolio)
    if n < 2:
        st.warning("Not enough approved sites in portfolio to display slider.")
        top_k = n
    else:
        top_k = st.slider("Top-K recommended sites", min_value=1, max_value=min(30, n), value=min(10, n))

    portfolio_view = portfolio.head(top_k).copy()
    portfolio_view["priority_phase"] = portfolio_view["rollout_priority"].apply(
        lambda x: "Phase 1" if "Phase 1" in str(x) else ("Phase 2" if "Phase 2" in str(x) else "Phase 3")
    )

    st.caption("Table: prioritized rollout portfolio from coupled optimization.")

    def _color_portfolio_row(row):
        if "Phase 1" in str(row["rollout_priority"]):
            bg = "#d9f2d9"
        elif "Phase 2" in str(row["rollout_priority"]):
            bg = "#fff6cc"
        else:
            bg = "#ebebeb"
        return [f"background-color: {bg}; color: #111111"] * len(row)

    st.dataframe(
        portfolio_view[
            ["portfolio_rank", "site_id", "zone", "assigned_feeder_id", "portfolio_score",
             "demand_score", "mean_delta_capacity_pct", "iterations_selected", "rollout_priority"]
        ].style
        .apply(_color_portfolio_row, axis=1)
        .set_properties(**{"color": "#111111"}),
        use_container_width=True,
    )

# ---- Site Selector ----
# Default to the top-ranked portfolio site so the feeder impact table shows real deltas on load.
default_site = portfolio.iloc[0]["site_id"] if not portfolio.empty else approved[0]
if default_site not in approved:
    default_site = approved[0]
site_id = st.selectbox("Select approved site", sorted(approved), index=sorted(approved).index(default_site))
max_iter = int(coupled["iteration"].max()) if not coupled.empty else 1
if max_iter <= 1:
    iteration = 1
    st.caption("Only one coupling iteration available for this run.")
else:
    iteration = st.slider("Iteration", min_value=1, max_value=max_iter, value=max_iter)

impact = coupled[(coupled["site_id"] == site_id) & (coupled["iteration"] == iteration)].copy()

# ---- Convergence ----
st.subheader("Optimization convergence")
if not iters.empty:
    st.line_chart(iters.set_index("iteration")[["objective_before", "objective_after"]])
    latest = iters.sort_values("iteration").tail(1).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Stressed feeders (before)", int(latest["stressed_before"]))
    c2.metric("Stressed feeders (after)", int(latest["stressed_after"]))
    c3.metric("Converged", "Yes" if bool(latest["converged"]) else "No")

# ---- Map ----
base = risk.groupby("feeder_id", as_index=False).tail(1)[["feeder_id", "status", "capacity_pct"]]
merged = feeders.merge(base, on="feeder_id", how="left").merge(
    impact[["feeder_id", "after_status", "capacity_pct_after"]], on="feeder_id", how="left"
)
merged["after_status"] = merged["after_status"].fillna(merged["status"])
merged["capacity_pct_after"] = merged["capacity_pct_after"].fillna(merged["capacity_pct"])
merged["delta_capacity_pct"] = merged["capacity_pct_after"] - merged["capacity_pct"]

color_map = {"SAFE": "green", "HIGH": "orange", "CRITICAL": "red"}
m = folium.Map(location=[12.97, 77.61], zoom_start=11, tiles="cartodbpositron")
for _, row in merged.iterrows():
    popup = f"{row['feeder_id']}: {row['status']} -> {row['after_status']}"
    folium.CircleMarker(
        [row["lat"], row["lon"]],
        radius=7,
        color=color_map.get(row["after_status"], "blue"),
        fill=True,
        popup=popup,
    ).add_to(m)
st_folium(m, width=1100, height=500)

before_critical = int((merged["status"] == "CRITICAL").sum())
after_critical = int((merged["after_status"] == "CRITICAL").sum())
total_delta = float(merged["delta_capacity_pct"].sum())
st.info(
    f"Placing {top_k} sites reduces CRITICAL feeders from {before_critical} to {after_critical} "
    f"and total stress by {abs(total_delta):.2f} capacity-% points."
)

# ---- Feeder Impact Table ----
st.subheader("Feeder-level impact table")
table_df = merged[
    ["feeder_id", "zone", "status", "after_status", "capacity_pct", "capacity_pct_after", "delta_capacity_pct"]
].rename(
    columns={
        "status": "before_status",
        "capacity_pct": "capacity_pct_before",
    }
)

st.caption("Table: feeder stress before and after selected portfolio impact.")

def _color_feeder_row(row):
    if row["after_status"] == "CRITICAL":
        bg = "#ffd6d6"
    elif row["after_status"] == "HIGH":
        bg = "#ffe5cc"
    else:
        bg = "#dcf5dc"
    return [f"background-color: {bg}; color: #111111"] * len(row)

st.dataframe(
    table_df.style
    .apply(_color_feeder_row, axis=1)
    .set_properties(**{"color": "#111111"}),
    use_container_width=True,
    column_config={
        "zone": st.column_config.TextColumn("zone", width=150),
    },
)

delta_fig = px.bar(
    table_df.sort_values("delta_capacity_pct"),
    x="feeder_id",
    y="delta_capacity_pct",
    color="delta_capacity_pct",
    color_continuous_scale="RdYlGn_r",
    title="Delta capacity-% per feeder (negative is improvement)",
    labels={"feeder_id": "Feeder", "delta_capacity_pct": "Delta capacity-%"},
)
st.plotly_chart(delta_fig, use_container_width=True)
