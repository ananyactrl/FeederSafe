from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.title("Page 5: Scheduling Nudges")
data_dir = Path("data/processed")
nudge_path = data_dir / "nudge_recommendations.csv"

if not nudge_path.exists():
    st.warning("Run the pipeline from Home page first.")
    st.stop()

nudges = pd.read_csv(nudge_path)
if nudges.empty:
    st.info("No HIGH/CRITICAL feeders detected in current synthetic run.")
    st.stop()

elasticity = st.slider("Short-run price elasticity", min_value=-0.8, max_value=-0.05, value=-0.3, step=0.05)
discount = st.slider("Discount (Rs/kWh)", min_value=0.5, max_value=6.0, value=3.0, step=0.5)

nudges["simulated_shift_pct"] = np.clip(abs(elasticity) * discount * 10, 3, 40).round(1)
st.dataframe(
    nudges[
        [
            "feeder_id",
            "status",
            "time_window",
            "off_peak_window",
            "discount_inr_per_kwh",
            "projected_load_shift_pct",
            "simulated_shift_pct",
            "nudge_text",
        ]
    ].sort_values(["status", "simulated_shift_pct"], ascending=[True, False])
)

st.caption(
    "Elasticity baseline used in prototype: -0.3 short-run own-price elasticity (illustrative planning assumption)."
)

