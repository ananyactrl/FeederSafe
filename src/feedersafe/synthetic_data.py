from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import AppConfig


ZONES = [
    "Koramangala",
    "Indiranagar",
    "Whitefield",
    "Electronic City",
    "Jayanagar",
    "Yelahanka",
    "Marathahalli",
    "HSR Layout",
    "BTM Layout",
    "Hebbal",
]

LAND_USE = ["commercial", "mixed_use", "residential", "industrial", "parking_lot"]


@dataclass
class SyntheticBundle:
    feeders: pd.DataFrame
    feeder_timeseries: pd.DataFrame
    smart_meter: pd.DataFrame
    candidate_sites: pd.DataFrame


def _zone_centroids() -> Dict[str, tuple[float, float]]:
    return {
        "Koramangala": (12.9352, 77.6245),
        "Indiranagar": (12.9784, 77.6408),
        "Whitefield": (12.9698, 77.7499),
        "Electronic City": (12.8456, 77.6603),
        "Jayanagar": (12.9293, 77.5828),
        "Yelahanka": (13.1005, 77.5963),
        "Marathahalli": (12.9591, 77.6974),
        "HSR Layout": (12.9116, 77.6474),
        "BTM Layout": (12.9166, 77.6101),
        "Hebbal": (13.0358, 77.5970),
    }


def _monsoon_multiplier(month: int) -> float:
    # Elevated appliance usage and damp-weather load effects in monsoon.
    return 1.08 if month in (6, 7, 8, 9) else 1.0


def _evening_shape(hour: float) -> float:
    if 18 <= hour <= 21:
        return 1.9
    if 21 < hour <= 22:
        return 1.55
    if 7 <= hour <= 9:
        return 1.12
    return 1.0


def generate_synthetic_data(config: AppConfig) -> SyntheticBundle:
    rng = np.random.default_rng(config.seed)
    centroids = _zone_centroids()

    feeder_rows: List[dict] = []
    for idx in range(config.n_feeders):
        zone = ZONES[idx % len(ZONES)]
        lat_c, lon_c = centroids[zone]
        feeder_rows.append(
            {
                "feeder_id": f"DT-{200 + idx}",
                "zone": zone,
                "rated_capacity_kva": rng.integers(50, 201),
                "base_load_pct": rng.uniform(0.55, 0.72),
                "ev_registration_count": int(rng.integers(70, 850)),
                "lat": lat_c + rng.normal(0, 0.01),
                "lon": lon_c + rng.normal(0, 0.01),
                "distance_to_11kv_m": float(rng.uniform(10, 300)),
                "phase_imbalance_pct": float(rng.uniform(8, 45)),
            }
        )
    feeders = pd.DataFrame(feeder_rows)

    end = pd.Timestamp(config.data_end_timestamp_utc).floor("15min")
    start = end - pd.Timedelta(days=config.n_days)
    timestamps = pd.date_range(start=start, end=end, freq=f"{config.interval_minutes}min")

    ts_rows: List[dict] = []
    smart_rows: List[dict] = []
    for feeder in feeders.itertuples(index=False):
        evening_peak_proneness = rng.uniform(0.0, 1.0)
        # Feeder-specific stress multiplier calibrated for Bengaluru evening load.
        if evening_peak_proneness > 0.82:
            feeder_peak_factor = rng.uniform(1.23, 1.32)  # CRITICAL-prone
        elif evening_peak_proneness > 0.48:
            feeder_peak_factor = rng.uniform(1.12, 1.2)   # HIGH-prone
        else:
            feeder_peak_factor = rng.uniform(0.98, 1.08)  # SAFE-prone
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60
            weekend = ts.weekday() >= 5
            temp = 22 + 8 * np.sin((hour - 7) / 24 * 2 * np.pi) + rng.normal(0, 1.2)
            temp += 1.0 if weekend else 0.0
            load_kw = (
                feeder.rated_capacity_kva
                * feeder.base_load_pct
                * _evening_shape(hour)
                * (feeder_peak_factor if 18 <= hour <= 21 else 1.0)
                * _monsoon_multiplier(ts.month)
                * (1.04 if weekend else 1.0)
                * (1 + 0.015 * max(temp - 28, 0))
            )
            noise = rng.normal(0, feeder.rated_capacity_kva * 0.035)
            inferred_events = int(rng.poisson(5 if 18 <= hour <= 21 else (3 if 21 < hour <= 22 else 1)))
            load_kw += inferred_events * rng.uniform(2.0, 5.0)
            # Calibration: keep only a minority of feeders in CRITICAL state
            # during the 18–21 peak window (hackathon target ~15–20% CRITICAL).
            # Use integer hour to scale the full 18:00-21:59 window.
            if 18 <= ts.hour <= 21:
                load_kw *= 0.445
            ts_rows.append(
                {
                    "timestamp": ts,
                    "feeder_id": feeder.feeder_id,
                    "zone": feeder.zone,
                    "load_kw": max(0.0, load_kw + noise),
                    "temperature_c": temp,
                    "is_holiday": int(ts.weekday() == 6 and rng.random() > 0.5),
                    "inferred_home_charging_events": inferred_events,
                }
            )

            # Synthetic meter signature: transient voltage sag + sustained current.
            base_voltage = 230 + rng.normal(0, 1.0)
            ev_start = int((18 <= ts.hour <= 23) and rng.random() < 0.14)
            voltage_v = base_voltage - (rng.uniform(4.6, 11.5) if ev_start else rng.uniform(0.0, 1.5))
            current_a = rng.uniform(3.0, 5.2) + (rng.uniform(6.0, 10.0) if ev_start else 0.0)
            smart_rows.append(
                {
                    "timestamp": ts,
                    "feeder_id": feeder.feeder_id,
                    "voltage_v": voltage_v,
                    "current_a": current_a,
                }
            )

    feeder_timeseries = pd.DataFrame(ts_rows)
    smart_meter = pd.DataFrame(smart_rows)

    sites = []
    for site_idx in range(config.n_sites):
        feeder = feeders.sample(1, random_state=int(config.seed + site_idx)).iloc[0]
        lat = feeder["lat"] + rng.normal(0, 0.008)
        lon = feeder["lon"] + rng.normal(0, 0.008)
        sites.append(
            {
                "site_id": f"SITE-{site_idx:03d}",
                "assigned_feeder_id": feeder["feeder_id"],
                "zone": feeder["zone"],
                "lat": lat,
                "lon": lon,
                "land_use": str(rng.choice(LAND_USE)),
                "trench_distance_m": float(rng.uniform(8, 320)),
                "clear_width_m": float(rng.uniform(2.4, 5.5)),
                "clear_length_m": float(rng.uniform(4.2, 9.5)),
                "road_width_m": float(rng.uniform(3.2, 8.5)),
                "hydrant_distance_m": float(rng.uniform(5, 40)),
                "accessibility_score": float(rng.uniform(40, 100)),
            }
        )
    candidate_sites = pd.DataFrame(sites)

    return SyntheticBundle(
        feeders=feeders,
        feeder_timeseries=feeder_timeseries,
        smart_meter=smart_meter,
        candidate_sites=candidate_sites,
    )

