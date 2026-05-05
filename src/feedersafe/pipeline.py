from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .coupling.optimizer import build_site_portfolio, run_coupled_impact
from .part_a.modeling import run_part_a
from .part_b.scoring import run_part_b
from .synthetic_data import generate_synthetic_data


def run_pipeline(config: AppConfig | None = None) -> dict[str, Path]:
    cfg = config or AppConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = generate_synthetic_data(cfg)
    part_a = run_part_a(synthetic.feeders, synthetic.feeder_timeseries, synthetic.smart_meter)
    part_b = run_part_b(cfg, synthetic.feeders, synthetic.candidate_sites, part_a.feeder_hourly_risk)
    coupled = run_coupled_impact(part_a.feeder_hourly_risk, part_b.site_results, feeders=synthetic.feeders)
    site_portfolio = build_site_portfolio(coupled, part_b.site_results)
    coupling_iterations = (
        coupled.groupby("iteration", as_index=False)
        .agg(
            objective_before=("objective_before", "first"),
            objective_after=("objective_after", "first"),
            stressed_before=("stressed_before", "first"),
            stressed_after=("stressed_after", "first"),
            converged=("converged", "first"),
        )
        if not coupled.empty
        else coupled
    )

    outputs = {
        "feeders": cfg.output_dir / "feeders.csv",
        "feeder_timeseries": cfg.output_dir / "feeder_timeseries.csv",
        "smart_meter": cfg.output_dir / "smart_meter.csv",
        "candidate_sites": cfg.output_dir / "candidate_sites.csv",
        "feeder_hourly_risk": cfg.output_dir / "feeder_hourly_risk.csv",
        "nudge_recommendations": cfg.output_dir / "nudge_recommendations.csv",
        "counterfactuals": cfg.output_dir / "counterfactuals.csv",
        "with_without_signal": cfg.output_dir / "with_without_signal.csv",
        "site_results": cfg.output_dir / "site_results.csv",
        "coupled_impact": cfg.output_dir / "coupled_impact.csv",
        "coupling_iterations": cfg.output_dir / "coupling_iterations.csv",
        "site_portfolio": cfg.output_dir / "site_portfolio.csv",
    }

    synthetic.feeders.to_csv(outputs["feeders"], index=False)
    synthetic.feeder_timeseries.to_csv(outputs["feeder_timeseries"], index=False)
    synthetic.smart_meter.to_csv(outputs["smart_meter"], index=False)
    synthetic.candidate_sites.to_csv(outputs["candidate_sites"], index=False)
    part_a.feeder_hourly_risk.to_csv(outputs["feeder_hourly_risk"], index=False)
    part_a.nudge_recommendations.to_csv(outputs["nudge_recommendations"], index=False)
    part_a.counterfactuals.to_csv(outputs["counterfactuals"], index=False)
    part_a.with_without_signal.to_csv(outputs["with_without_signal"], index=False)
    part_b.site_results.to_csv(outputs["site_results"], index=False)
    coupled.to_csv(outputs["coupled_impact"], index=False)
    coupling_iterations.to_csv(outputs["coupling_iterations"], index=False)
    site_portfolio.to_csv(outputs["site_portfolio"], index=False)

    return outputs


if __name__ == "__main__":
    run_pipeline()

