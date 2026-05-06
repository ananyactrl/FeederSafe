from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


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


st.title("FeederSafe | Scheduling Nudges")
st.caption("Feeder-targeted off-peak incentives to shift EV load away from high-stress evening windows.")
st.info(
    "These nudges help BESCOM flatten evening peaks by shifting home charging to off-peak windows using feeder-specific incentives."
)
data_dir = Path("data/processed")
nudge_path = data_dir / "nudge_recommendations.csv"

if not nudge_path.exists():
    st.warning("Run the pipeline from Home page first.")
    st.stop()

nudges = _load_frame("nudge_recommendations", nudge_path)
if nudges.empty:
    st.info("No HIGH/CRITICAL feeders detected in current synthetic run.")
    st.stop()

elasticity = st.slider("Short-run price elasticity", min_value=-0.8, max_value=-0.05, value=-0.3, step=0.05)
discount = st.slider("Discount (Rs/kWh)", min_value=0.5, max_value=6.0, value=3.0, step=0.5)

nudges["simulated_shift_pct"] = np.clip(abs(elasticity) * discount * 10, 3, 40).round(1)
critical_count = int((nudges["status"] == "CRITICAL").sum())
high_count = int((nudges["status"] == "HIGH").sum())
estimated_shift_kw = float((nudges["simulated_shift_pct"] / 100.0 * 100).sum())
st.subheader("Tonight's Risk Summary")
st.write(
    f"CRITICAL feeders: {critical_count} | HIGH feeders: {high_count} | "
    f"Estimated load that can be shifted: {estimated_shift_kw:.1f} kW"
)

for feeder_id, group in nudges.sort_values(["status", "simulated_shift_pct"], ascending=[True, False]).groupby("feeder_id"):
    with st.expander(f"{feeder_id} | {group.iloc[0]['status']}"):
        row = group.iloc[0]
        st.write(f"**Nudge text:** {row['nudge_text']}")
        before_after = pd.DataFrame(
            {
                "scenario": ["Before discount", "After discount"],
                "load_index": [100, 100 - float(row["simulated_shift_pct"])],
            }
        )
        fig = px.bar(
            before_after,
            x="scenario",
            y="load_index",
            title="Simulated load shift (index)",
            labels={"scenario": "Scenario", "load_index": "Relative load index"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Elasticity basis: short-run own-price elasticity {elasticity:.2f}, "
            f"discount {discount:.1f} Rs/kWh, simulated shift {row['simulated_shift_pct']:.1f}%."
        )

st.caption(
    "Elasticity baseline used in prototype: -0.3 short-run own-price elasticity (illustrative planning assumption)."
)

