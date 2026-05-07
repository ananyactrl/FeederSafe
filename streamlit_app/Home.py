from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feedersafe.pipeline import run_pipeline

st.set_page_config(page_title="FeederSafe", layout="wide")


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_processed_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "feeders": "feeders.csv",
        "feeder_timeseries": "feeder_timeseries.csv",
        "smart_meter": "smart_meter.csv",
        "feeder_hourly_risk": "feeder_hourly_risk.csv",
        "nudge_recommendations": "nudge_recommendations.csv",
        "counterfactuals": "counterfactuals.csv",
        "with_without_signal": "with_without_signal.csv",
        "site_results": "site_results.csv",
        "coupled_impact": "coupled_impact.csv",
        "site_portfolio": "site_portfolio.csv",
        "coupling_iterations": "coupling_iterations.csv",
        "candidate_sites": "candidate_sites.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, file_name in files.items():
        file_path = data_dir / file_name
        if file_path.exists():
            out[key] = _read_csv_cached(file_path)
    return out


st.title("FeederSafe")
st.caption("AI for Bharat 2 submission demo: Bengaluru feeder stress, siting feasibility, coupling impact, and nudges.")
st.info("BESCOM context: this dashboard simulates transformer-level stress, site rollout, and demand-shift levers to support urban EV scaling decisions.")
st.caption("Calibrated to BESCOM published data: 80x growth in registered EV charging 2020-2023 (28,820 kWh in 2020-21 to ~23 lakh kWh in 2022-23).")

data_dir = Path("data/processed")
required_outputs = [
    "feeders.csv",
    "feeder_timeseries.csv",
    "smart_meter.csv",
    "candidate_sites.csv",
    "feeder_hourly_risk.csv",
    "nudge_recommendations.csv",
    "counterfactuals.csv",
    "with_without_signal.csv",
    "site_results.csv",
    "coupled_impact.csv",
    "coupling_iterations.csv",
    "site_portfolio.csv",
]
missing_outputs = [name for name in required_outputs if not (data_dir / name).exists()]
if missing_outputs:
    with st.spinner("Running pipeline to prepare required outputs..."):
        run_pipeline()
        _read_csv_cached.clear()
    st.success("Pipeline generated required CSV outputs.")

if st.button("Generate / Refresh Synthetic Pipeline Outputs", type="primary"):
    with st.spinner("Running pipeline..."):
        outputs = run_pipeline()
        _read_csv_cached.clear()
        st.session_state["data"] = _load_processed_data(data_dir)
    st.success("Pipeline complete.")
    st.json({k: str(v) for k, v in outputs.items()})

if not data_dir.exists():
    st.warning("No outputs found yet. Click the button above to generate prototype data.")
else:
    if "data" not in st.session_state:
        st.session_state["data"] = _load_processed_data(data_dir)
    data = st.session_state.get("data", {})
    st.info("Use the left sidebar to open pages: Feeder Risk, Home Charging, Site Planner, Coupled Impact, Nudges.")
    st.write(
        """
        ### Why FeederSafe is differentiated
        - Quantile stack at **q=0.95** for overload-tail safety
        - Inferred home charging via synthetic smart-meter voltage sag signatures
        - Hard mechanical veto matrix (field-deployable constraints)
        - Joint optimization: **where to build + when to charge + how load redistributes**
        - Counterfactual explainer for CRITICAL feeders
        """
    )

    feeders_df = data.get("feeders", pd.DataFrame())
    risk_df = data.get("feeder_hourly_risk", pd.DataFrame())
    sites_df = data.get("site_results", pd.DataFrame())
    nudge_df = data.get("nudge_recommendations", pd.DataFrame())

    total_feeders = int(feeders_df["feeder_id"].nunique()) if "feeder_id" in feeders_df.columns else 0
    tonight_risk = pd.DataFrame()
    if not risk_df.empty and {"hour", "feeder_id"}.issubset(risk_df.columns):
        latest_hour = int(pd.to_numeric(risk_df["hour"], errors="coerce").dropna().max())
        tonight_risk = risk_df[pd.to_numeric(risk_df["hour"], errors="coerce") == latest_hour].copy()
    critical_tonight = int((tonight_risk.get("status", pd.Series(dtype=str)) == "CRITICAL").sum())
    approved_sites = int((sites_df.get("decision", pd.Series(dtype=str)) == "APPROVED").sum())

    estimated_shiftable_kw = 0.0
    if not tonight_risk.empty and {"feeder_id", "predicted_load_p95_kw", "status"}.issubset(tonight_risk.columns):
        stressed = tonight_risk[tonight_risk["status"].isin(["HIGH", "CRITICAL"])][
            ["feeder_id", "status", "predicted_load_p95_kw"]
        ].copy()
        if not nudge_df.empty and {"feeder_id", "status", "projected_load_shift_pct"}.issubset(nudge_df.columns):
            nudges = (
                nudge_df[["feeder_id", "status", "projected_load_shift_pct"]]
                .drop_duplicates(subset=["feeder_id", "status"])
                .copy()
            )
            merged_shift = stressed.merge(nudges, on=["feeder_id", "status"], how="left")
            merged_shift["projected_load_shift_pct"] = pd.to_numeric(
                merged_shift["projected_load_shift_pct"], errors="coerce"
            ).fillna(0.0)
            estimated_shiftable_kw = float(
                (merged_shift["predicted_load_p95_kw"] * merged_shift["projected_load_shift_pct"] / 100.0).sum()
            )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total feeders monitored", total_feeders)
    c2.metric("Critical feeders tonight", critical_tonight)
    c3.metric("Approved EV charging sites", approved_sites)
    c4.metric("Estimated shiftable load (kW)", f"{estimated_shiftable_kw:.1f}")

    st.caption(
        "Prototype runs 50 feeders. Architecture scales to BESCOM's 800+ distribution transformers via `config.n_feeders` parameter."
    )

    if not tonight_risk.empty and {"status"}.issubset(tonight_risk.columns):
        risk_counts = (
            tonight_risk["status"].value_counts().reindex(["CRITICAL", "HIGH", "SAFE"], fill_value=0).reset_index()
        )
        risk_counts.columns = ["status", "count"]
        risk_fig = px.bar(
            risk_counts,
            x="status",
            y="count",
            text="count",
            color="status",
            color_discrete_map={"CRITICAL": "red", "HIGH": "orange", "SAFE": "green"},
            title="Feeder risk distribution (latest modeled hour)",
            labels={"status": "Risk status", "count": "Feeders"},
        )
        risk_fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
        risk_fig.update_yaxes(tickmode="linear", dtick=1, tickformat="d", rangemode="tozero")
        st.plotly_chart(risk_fig, use_container_width=True)

        top_at_risk = tonight_risk.copy()
        top_at_risk["risk_rank"] = top_at_risk["status"].map({"CRITICAL": 2, "HIGH": 1, "SAFE": 0}).fillna(0)
        top_at_risk = top_at_risk.sort_values(["risk_rank", "capacity_pct"], ascending=[False, False]).head(5)
        st.caption("Top 5 at-risk feeders (latest modeled hour).")
        st.dataframe(
            top_at_risk[["feeder_id", "hour", "status", "predicted_load_p95_kw", "capacity_pct"]]
            if {"feeder_id", "hour", "status", "predicted_load_p95_kw", "capacity_pct"}.issubset(top_at_risk.columns)
            else top_at_risk
        )

