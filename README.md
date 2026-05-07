# FeederSafe

### Grid-Constrained EV Charging Planner for BESCOM

**AI for Bharat 2 Hackathon – Theme 9**

FeederSafe is an end-to-end decision-support prototype that answers three questions BESCOM must solve:

1. **Where to build** new EV charging stations — only sites that pass physical, grid, and safety feasibility checks.
2. **When to charge** — feeder-specific off-peak nudges that reduce evening transformer overload.
3. **How the grid stress redistributes** after approved sites are deployed.

---

## Core Differentiators

- **Quantile tail-risk forecasting (q=0.95)** using a LightGBM + BiLSTM stacking ensemble with a pinball-loss meta-learner — predicts the 95th percentile feeder load, not the mean.
- **Home charging inference** from synthetic smart-meter voltage sag signatures — closes the blind spot of unregistered portable chargers.
- **Mechanical veto matrix** — every candidate charger site must pass hard, operator-adjustable thresholds:
  - DT headroom ≥ 15%
  - Trench distance ≤ 150 m to nearest 11 kV feeder
  - Footprint ≥ 3 m × 6 m
  - Emergency access (road width ≥ 4.5 m) and hydrant clearance ≥ 15 m
  - Phase imbalance ≤ 30%
- **Coupled impact analysis** — before/after feeder stress table, convergence chart, and rollout priority ranking (Phase 1/2/3).
- **Explainable, actionable outputs** — every rejection comes with a specific, auditable reason; every nudge is feeder-specific.

---

## Project Structure

```
feedersafe/
├── data/
│   └── processed/              # pipeline outputs (*.csv)
├── src/
│   └── feedersafe/
│       ├── config.py           # AppConfig (n_feeders, n_sites, veto defaults)
│       ├── synthetic_data.py   # generates masked Bengaluru-like data
│       ├── part_a/
│       │   └── modeling.py     # Part A: forecasting (LightGBM, BiLSTM, quantile)
│       ├── part_b/
│       │   └── scoring.py      # Part B: spatial scoring + hard veto filters
│       ├── coupling/
│       │   └── optimizer.py    # greedy placement optimizer + coupled impact
│       └── pipeline.py         # end-to-end runner that writes all CSVs
├── streamlit_app/
│   ├── Home.py                 # landing page (pipeline trigger, KPIs)
│   └── pages/
│       ├── 1_Feeder_Risk_Map.py
│       ├── 2_Home_Charging_Signature.py
│       ├── 3_Site_Feasibility_Planner.py
│       ├── 4_Coupled_Impact_Analysis.py
│       └── 5_Scheduling_Nudges.py
├── requirements.txt
└── README.md
```

---

## Quickstart (Local)

### 1. Set up environment

```bash
python -m venv .venv
```

**Windows:**

```powershell
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

Generates all processed CSV data under `data/processed/`.

**Windows PowerShell:**

```powershell
$env:PYTHONPATH='src'
python -m feedersafe.pipeline
```

**Linux / macOS / Git Bash:**

```bash
PYTHONPATH=src python -m feedersafe.pipeline
```

### 4. Launch the dashboard

**Windows PowerShell:**

```powershell
$env:PYTHONPATH='src'
streamlit run streamlit_app/Home.py
```

**Linux / macOS / Git Bash:**

```bash
PYTHONPATH=src streamlit run streamlit_app/Home.py
```

The dashboard opens at **http://localhost:8501**.

> If you see _"Run pipeline from Home page first"_, click the **Generate / Refresh Synthetic Pipeline Outputs** button on the Home page.

---

## Dashboard Walkthrough

| Page                         | What it shows                                                                                                                                                                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Home**                     | Total feeders, CRITICAL tonight count, approved sites, shiftable load (kW), risk bar chart, top-5 at-risk feeders                                                                                                                                 |
| **Feeder Risk Map**          | Bengaluru masked grid coloured by risk (green / orange / red). Select any feeder → 24-h load curve with 95th percentile line vs rated capacity. HIGH/CRITICAL feeders show nudge text and counterfactuals                                         |
| **Home Charging Signature**  | Voltage trace with red dots for inferred EV sessions, bar chart of inferred sessions by hour, with/without signal comparison showing the p95 uplift                                                                                               |
| **Site Feasibility Planner** | Map of candidate sites (green = approved, red = rejected). Sidebar has 7 operator-adjustable veto sliders. "Apply Veto Thresholds" button shows live count. Click a site to see veto reasons and the nearest feasible alternative                 |
| **Coupled Impact Analysis**  | Rollout portfolio table (Phase 1 in green, Phase 2 in yellow). Convergence line chart (4 iterations, converged = Yes). Feeder-level impact table: before/after status, capacity %, delta (negative = improvement). Total stress reduction summary |
| **Scheduling Nudges**        | Short-run price elasticity slider and discount slider. "Tonight's Risk Summary" cards. Per-feeder expanders with before/after load-shift bar chart and nudge text                                                                                 |

---

## Scaling to Real BESCOM Deployment

| Component              | How to scale                                                                                                                     |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Number of feeders**  | Change `n_feeders` in `config.py` to match BESCOM's 800+ distribution transformers                                               |
| **Veto matrix**        | Vectorised pandas operations scale linearly with candidate sites                                                                 |
| **Coupling optimizer** | Replace `iterrows` with a NumPy distance matrix for sub-second runs on 800+ feeders                                              |
| **Forecasting model**  | Swap the BiLSTM for a lighter quantile gradient boosting model for real-time inference — the stacking architecture is modular    |
| **Data integration**   | Replace the synthetic data generator with BESCOM's real SCADA, smart-meter, and GIS streams — the pipeline API remains identical |

---

## Key Dependencies

| Package                                   | Purpose                                    |
| ----------------------------------------- | ------------------------------------------ |
| `lightgbm`                                | Gradient boosting for quantile forecasting |
| `torch`                                   | BiLSTM sequence model                      |
| `scikit-learn`                            | Stacking meta-learner, preprocessing       |
| `pandas`, `numpy`                         | Data wrangling                             |
| `streamlit`, `streamlit-folium`, `folium` | Dashboard and interactive maps             |
| `plotly`                                  | Charts and visualisations                  |

Full list in [`requirements.txt`](requirements.txt). Requires **Python ≥ 3.10**.

---

## License

Created for the **AI for Bharat 2 Hackathon**. All intellectual property belongs to the team.
