from __future__ import annotations

import numpy as np
import pandas as pd


def _status_from_capacity(capacity_pct: float) -> str:
    if capacity_pct > 100:
        return "CRITICAL"
    if capacity_pct > 85:
        return "HIGH"
    return "SAFE"


def _site_feeder_relief(
    site_row: pd.Series,
    feeder_row: pd.Series,
    elasticity: float,
    nudge_strength: float,
) -> float:
    """
    Relief model for one site-feeder pair.
    - Base public-shift effect scales with elasticity and nudge strength.
    - Stronger effect when site belongs to same stressed feeder zone.
    - Small spillover to nearby stressed feeders.
    """
    base_relief = max(1.5, min(10.0, abs(elasticity) * 14 * nudge_strength))
    if site_row["assigned_feeder_id"] == feeder_row["feeder_id"]:
        return base_relief * 1.6
    if site_row["zone"] == feeder_row["zone"]:
        return base_relief * 0.7
    return base_relief * 0.35


def run_coupled_impact(
    feeder_hourly_risk: pd.DataFrame,
    site_results: pd.DataFrame,
    elasticity: float = -0.3,
    max_iterations: int = 8,
    convergence_eps: float = 0.2,
    max_sites_per_iteration: int = 3,
) -> pd.DataFrame:
    stressed = (
        feeder_hourly_risk.sort_values("hour")
        .groupby("feeder_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    stressed["status"] = stressed["capacity_pct"].map(_status_from_capacity)
    approved = (
        site_results[site_results["decision"].eq("APPROVED")]
        .sort_values("demand_score", ascending=False)
        .reset_index(drop=True)
    )

    if stressed.empty or approved.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "feeder_id",
                "before_status",
                "after_status",
                "capacity_pct_before",
                "capacity_pct_after",
                "delta_capacity_pct",
                "net_effect",
                "iteration",
                "selected_in_iteration",
                "objective_before",
                "objective_after",
                "stressed_before",
                "stressed_after",
                "converged",
            ]
        )

    selected_sites: set[str] = set()
    rows: list[dict] = []
    convergence_history: list[float] = []

    for iteration in range(1, max_iterations + 1):
        stressed_count_before = int((stressed["capacity_pct"] > 85).sum())
        objective_before = float(np.maximum(stressed["capacity_pct"] - 85, 0).sum())

        remaining_sites = approved[~approved["site_id"].isin(selected_sites)].copy()
        if remaining_sites.empty:
            break

        scores = []
        for _, site in remaining_sites.iterrows():
            total_relief = 0.0
            for _, feeder in stressed.iterrows():
                if feeder["capacity_pct"] <= 85:
                    continue
                total_relief += _site_feeder_relief(site, feeder, elasticity=elasticity, nudge_strength=1.0)
            # Penalize sites with lower demand score and weak headroom.
            penalty = max(0.0, (65 - float(site.get("demand_score", 65))) * 0.03)
            score = total_relief - penalty
            scores.append((site["site_id"], score))

        if not scores:
            break

        scored = pd.DataFrame(scores, columns=["site_id", "iteration_gain"]).sort_values(
            "iteration_gain", ascending=False
        )
        pick_ids = scored.head(max_sites_per_iteration)["site_id"].tolist()
        selected_sites.update(pick_ids)

        before_map = stressed.set_index("feeder_id")["capacity_pct"].to_dict()
        selected_this_iter = approved[approved["site_id"].isin(pick_ids)].copy()

        # Apply cumulative relief from selected sites this iteration.
        for feeder_idx, feeder in stressed.iterrows():
            relief = 0.0
            for _, site in selected_this_iter.iterrows():
                relief += _site_feeder_relief(site, feeder, elasticity=elasticity, nudge_strength=1.0)
            # Diminishing returns as feeder gets healthier.
            health_factor = 1.0 if feeder["capacity_pct"] > 100 else (0.8 if feeder["capacity_pct"] > 85 else 0.4)
            relief *= health_factor
            stressed.at[feeder_idx, "capacity_pct"] = max(40.0, feeder["capacity_pct"] - relief)

        stressed["status"] = stressed["capacity_pct"].map(_status_from_capacity)
        stressed_count_after = int((stressed["capacity_pct"] > 85).sum())
        objective_after = float(np.maximum(stressed["capacity_pct"] - 85, 0).sum())
        delta_obj = objective_before - objective_after
        convergence_history.append(delta_obj)
        converged = abs(delta_obj) <= convergence_eps

        for site_id in pick_ids:
            for feeder_id, after_capacity in stressed.set_index("feeder_id")["capacity_pct"].items():
                before_capacity = before_map[feeder_id]
                rows.append(
                    {
                        "site_id": site_id,
                        "feeder_id": feeder_id,
                        "before_status": _status_from_capacity(before_capacity),
                        "after_status": _status_from_capacity(after_capacity),
                        "capacity_pct_before": before_capacity,
                        "capacity_pct_after": after_capacity,
                        "delta_capacity_pct": after_capacity - before_capacity,
                        "net_effect": "improves" if after_capacity < before_capacity else "worsens",
                        "iteration": iteration,
                        "selected_in_iteration": True,
                        "objective_before": objective_before,
                        "objective_after": objective_after,
                        "stressed_before": stressed_count_before,
                        "stressed_after": stressed_count_after,
                        "converged": converged,
                    }
                )

        if converged:
            break

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    penalty = (
        out.groupby("site_id")["delta_capacity_pct"].mean().rename("coupling_penalty")
        .reset_index()
        .assign(coupling_penalty=lambda d: np.clip(d["coupling_penalty"], -20, 10))
    )
    return out.merge(penalty, on="site_id", how="left")


def build_site_portfolio(
    coupled_impact: pd.DataFrame,
    site_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a ranked rollout list from optimization traces.
    Higher rank = stronger average relief with better feasibility score.
    """
    if coupled_impact.empty:
        return pd.DataFrame(
            columns=[
                "portfolio_rank",
                "site_id",
                "zone",
                "assigned_feeder_id",
                "decision",
                "demand_score",
                "iterations_selected",
                "mean_delta_capacity_pct",
                "best_iteration_improvement",
                "feeder_improvements",
                "portfolio_score",
                "rollout_priority",
            ]
        )

    impact_site = (
        coupled_impact.groupby("site_id", as_index=False)
        .agg(
            iterations_selected=("iteration", "nunique"),
            mean_delta_capacity_pct=("delta_capacity_pct", "mean"),
            best_iteration_improvement=("delta_capacity_pct", "min"),
            feeder_improvements=("net_effect", lambda s: int((s == "improves").sum())),
        )
    )

    merged = impact_site.merge(
        site_results[
            ["site_id", "zone", "assigned_feeder_id", "decision", "demand_score", "nearest_feasible_alternative"]
        ],
        on="site_id",
        how="left",
    )
    merged = merged[merged["decision"].eq("APPROVED")].copy()
    if merged.empty:
        return merged

    relief_component = np.clip(-merged["mean_delta_capacity_pct"], 0, 30) * 2.2
    consistency_component = np.clip(merged["iterations_selected"], 0, 8) * 4.5
    demand_component = np.clip(merged["demand_score"], 0, 100) * 0.35
    merged["portfolio_score"] = (relief_component + consistency_component + demand_component).round(2)
    merged = merged.sort_values("portfolio_score", ascending=False).reset_index(drop=True)
    merged["portfolio_rank"] = np.arange(1, len(merged) + 1)
    merged["rollout_priority"] = np.select(
        [merged["portfolio_rank"] <= 5, merged["portfolio_rank"] <= 15],
        ["Immediate (Phase 1)", "Near-term (Phase 2)"],
        default="Later (Phase 3)",
    )
    return merged[
        [
            "portfolio_rank",
            "site_id",
            "zone",
            "assigned_feeder_id",
            "decision",
            "demand_score",
            "iterations_selected",
            "mean_delta_capacity_pct",
            "best_iteration_improvement",
            "feeder_improvements",
            "portfolio_score",
            "rollout_priority",
        ]
    ]

