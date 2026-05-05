from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import AppConfig


@dataclass
class PartBOutput:
    site_results: pd.DataFrame


def _score_land_use(v: str) -> float:
    return {
        "commercial": 100,
        "mixed_use": 90,
        "parking_lot": 85,
        "industrial": 75,
        "residential": 60,
    }.get(v, 50)


def run_part_b(
    config: AppConfig,
    feeders: pd.DataFrame,
    candidate_sites: pd.DataFrame,
    feeder_hourly_risk: pd.DataFrame,
) -> PartBOutput:
    latest_risk = (
        feeder_hourly_risk.sort_values("hour")
        .groupby("feeder_id", as_index=False)
        .tail(1)[["feeder_id", "capacity_pct", "status"]]
    )
    df = candidate_sites.merge(
        feeders[["feeder_id", "ev_registration_count", "distance_to_11kv_m", "phase_imbalance_pct", "rated_capacity_kva"]],
        left_on="assigned_feeder_id",
        right_on="feeder_id",
        how="left",
    ).merge(latest_risk, left_on="assigned_feeder_id", right_on="feeder_id", how="left", suffixes=("", "_risk"))

    df["dt_headroom_pct"] = (100 - df["capacity_pct"]).fillna(35)
    df["ev_density_score"] = np.clip(df["ev_registration_count"] / df["ev_registration_count"].max() * 100, 0, 100)
    df["proximity_score"] = np.clip((1 - df["distance_to_11kv_m"] / 300) * 100, 0, 100)
    df["headroom_score"] = np.clip(df["dt_headroom_pct"], 0, 100)
    df["road_score"] = np.clip(df["accessibility_score"], 0, 100)
    df["land_use_score"] = df["land_use"].map(_score_land_use)

    df["demand_score"] = (
        0.30 * df["ev_density_score"]
        + 0.25 * df["proximity_score"]
        + 0.25 * df["headroom_score"]
        + 0.10 * df["road_score"]
        + 0.10 * df["land_use_score"]
    ).round(2)

    rejections = []
    for row in df.itertuples(index=False):
        reasons = []
        if row.dt_headroom_pct <= config.dt_headroom_min_pct:
            reasons.append("DT headroom below threshold")
        if row.trench_distance_m >= config.trench_distance_max_m:
            reasons.append("Trench distance exceeds threshold")
        if row.clear_width_m < config.min_width_m or row.clear_length_m < config.min_length_m:
            reasons.append("Footprint below minimum 3m x 6m")
        if row.road_width_m <= config.min_road_width_m or row.hydrant_distance_m <= config.hydrant_clearance_min_m:
            reasons.append("Emergency access non-compliant")
        if row.phase_imbalance_pct >= config.phase_imbalance_max_pct:
            reasons.append("Phase balance exceeds threshold")
        rejections.append("; ".join(reasons) if reasons else "APPROVED")
    df["veto_reasons"] = rejections
    df["decision"] = np.where(df["veto_reasons"].eq("APPROVED"), "APPROVED", "REJECTED")

    approved = df[df["decision"].eq("APPROVED")].copy()
    if not approved.empty:
        tree = cKDTree(approved[["lat", "lon"]].to_numpy())
        nearest_alt = []
        for row in df.itertuples(index=False):
            if row.decision == "APPROVED":
                nearest_alt.append(row.site_id)
                continue
            _, idx = tree.query([row.lat, row.lon], k=1)
            nearest_alt.append(approved.iloc[int(idx)]["site_id"])
    else:
        nearest_alt = [None] * len(df)
    df["nearest_feasible_alternative"] = nearest_alt

    return PartBOutput(site_results=df.sort_values(["decision", "demand_score"], ascending=[True, False]))

