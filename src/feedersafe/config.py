from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    seed: int = 42
    n_feeders: int = 50
    n_sites: int = 200
    n_days: int = 90
    interval_minutes: int = 15
    output_dir: Path = Path("data/processed")

    # Part B adjustable veto defaults.
    dt_headroom_min_pct: float = 15.0
    trench_distance_max_m: float = 150.0
    min_width_m: float = 3.0
    min_length_m: float = 6.0
    min_road_width_m: float = 4.5
    hydrant_clearance_min_m: float = 15.0
    phase_imbalance_max_pct: float = 30.0

