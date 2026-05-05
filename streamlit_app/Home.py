from __future__ import annotations

from pathlib import Path

import streamlit as st

from feedersafe.pipeline import run_pipeline

st.set_page_config(page_title="FeederSafe", layout="wide")
st.title("FeederSafe: Grid-Constrained EV Charging Planner")
st.caption(
    "BESCOM hackathon prototype | Tail-risk forecasting (q=0.95), home-charging signatures, "
    "mechanical veto feasibility, and coupled site-vs-load optimization."
)

data_dir = Path("data/processed")
if st.button("Generate / Refresh Synthetic Pipeline Outputs", type="primary"):
    with st.spinner("Running synthetic data + Part A + Part B + coupling..."):
        outputs = run_pipeline()
    st.success("Pipeline complete.")
    st.json({k: str(v) for k, v in outputs.items()})

if not data_dir.exists():
    st.warning("No outputs found yet. Click the button above to generate prototype data.")
else:
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

