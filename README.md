# FeederSafe

Grid-Constrained EV Charging Planner prototype for BESCOM (AI for Bharat 2 hackathon).

## Core idea

FeederSafe jointly optimizes:
1. **Where to build** EV charging stations (feasible sites),
2. **When to charge** (nudge windows and off-peak shift),
3. **How load redistributes** across feeders after site deployment.

This prototype demonstrates:
- Quantile tail-risk forecasting at **q=0.95** (Pinball-loss objective),
- Home charging inference via smart meter voltage sag signatures,
- Mechanical veto matrix for field-ready infra feasibility,
- Coupled before/after feeder impact analysis for candidate sites.

## Project structure

- `src/feedersafe/synthetic_data.py` synthetic BESCOM-like data generation
- `src/feedersafe/part_a/modeling.py` Part A stack (LightGBM + BiLSTM + quantile meta)
- `src/feedersafe/part_b/scoring.py` Part B spatial scoring + hard veto filters
- `src/feedersafe/coupling/optimizer.py` coupled redistribution impact logic
- `src/feedersafe/pipeline.py` end-to-end runner and data export
- `streamlit_app/` multipage dashboard

## Quickstart (local)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
set PYTHONPATH=src
python -m feedersafe.pipeline
streamlit run streamlit_app/Home.py
```

## Scaling to BESCOM Production
- Current prototype runs 50 feeders; `n_feeders` in `src/feedersafe/config.py` scales this to any number.
- The veto matrix uses vectorized pandas operations that scale approximately linearly with the number of candidate sites; tested mentally for 800+ feeders.
- The coupling loop uses precomputed proximity graphs; for production, replace the `iterrows` loops with a NumPy distance matrix for faster 800+ feeder rollouts.
- The BiLSTM can be replaced with a lighter quantile gradient boosting model for real-time inference at scale.
