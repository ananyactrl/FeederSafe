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

    # zone comparison — feeder_hourly_risk may not carry a zone column,
    # so use .get() with a sentinel to avoid KeyError
    site_zone = site_row.get("zone", None)
    feeder_zone = feeder_row.get("zone", None)
    if site_zone is not None and feeder_zone is not None and site_zone == feeder_zone:
        return base_relief * 0.7

    return base_relief * 0.35


def run_coupled_impact(
    feeder_hourly_risk: pd.DataFrame,
    site_results: pd.DataFrame,
    feeders: pd.DataFrame | None = None,
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

    # Enrich stressed with zone if available from feeders
    if feeders is not None and "zone" in feeders.columns and "zone" not in stressed.columns:
        stressed = stressed.merge(
            feeders[["feeder_id", "zone"]], on="feeder_id", how="left"
        )

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
                total_relief += _site_feeder_relief(
                    site, feeder, elasticity=elasticity, nudge_strength=1.0
                )
            penalty = max(0.0, (65 - float(site.get("demand_score", 65))) * 0.03)
            scores.append((site["site_id"], total_relief - penalty))

        if not scores:
            break

        scored = pd.DataFrame(scores, columns=["site_id", "iteration_gain"]).sort_values(
            "iteration_gain", ascending=False
        )
        pick_ids = scored.head(max_sites_per_iteration)["site_id"].tolist()
        selected_sites.update(pick_ids)

        before_map = stressed.set_index("feeder_id")["capacity_pct"].to_dict()
        selected_this_iter = approved[approved["site_id"].isin(pick_ids)].copy()

        for feeder_idx, feeder in stressed.iterrows():
            relief = 0.0
            for _, site in selected_this_iter.iterrows():
                relief += _site_feeder_relief(
                    site, feeder, elasticity=elasticity, nudge_strength=1.0
                )
            health_factor = (
                1.0 if feeder["capacity_pct"] > 100
                else (0.8 if feeder["capacity_pct"] > 85 else 0.4)
            )
            stressed.at[feeder_idx, "capacity_pct"] = max(
                40.0, feeder["capacity_pct"] - relief * health_factor
            )

        stressed["status"] = stressed["capacity_pct"].map(_status_from_capacity)
        stressed_count_after = int((stressed["capacity_pct"] > 85).sum())
        objective_after = float(np.maximum(stressed["capacity_pct"] - 85, 0).sum())
        delta_obj = objective_before - objective_after
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
        out.groupby("site_id")["delta_capacity_pct"]
        .mean()
        .rename("coupling_penalty")
        .reset_index()
        .assign(coupling_penalty=lambda d: np.clip(d["coupling_penalty"], -20, 10))
    )
    return out.merge(penalty, on="site_id", how="left")


def build_site_portfolio(
    coupled_impact: pd.DataFrame,
    site_results: pd.DataFrame,
) -> pd.DataFrame:
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

    # Only include columns that actually exist in site_results
    portfolio_cols = ["site_id", "assigned_feeder_id", "decision", "demand_score",
                      "nearest_feasible_alternative"]
    if "zone" in site_results.columns:
        portfolio_cols.insert(1, "zone")

    merged = impact_site.merge(
        site_results[[c for c in portfolio_cols if c in site_results.columns]],
        on="site_id",
        how="left",
    )
    merged = merged[merged["decision"].eq("APPROVED")].copy()
    if merged.empty:
        return merged

    relief_component = np.clip(-merged["mean_delta_capacity_pct"], 0, 30) * 2.2
    consistency_component = np.clip(merged["iterations_selected"], 0, 8) * 4.5
    demand_component = np.clip(merged["demand_score"], 0, 100) * 0.35
    merged["portfolio_score"] = (
        relief_component + consistency_component + demand_component
    ).round(2)
    merged = merged.sort_values("portfolio_score", ascending=False).reset_index(drop=True)
    merged["portfolio_rank"] = np.arange(1, len(merged) + 1)
    merged["rollout_priority"] = np.select(
        [merged["portfolio_rank"] <= 5, merged["portfolio_rank"] <= 15],
        ["Immediate (Phase 1)", "Near-term (Phase 2)"],
        default="Later (Phase 3)",
    )

    out_cols = [
        "portfolio_rank", "site_id", "assigned_feeder_id", "decision",
        "demand_score", "iterations_selected", "mean_delta_capacity_pct",
        "best_iteration_improvement", "feeder_improvements",
        "portfolio_score", "rollout_priority",
    ]
    if "zone" in merged.columns:
        out_cols.insert(2, "zone")

    return merged[[c for c in out_cols if c in merged.columns]]