from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


@dataclass
class PartAOutput:
    feeder_hourly_risk: pd.DataFrame
    nudge_recommendations: pd.DataFrame
    counterfactuals: pd.DataFrame
    with_without_signal: pd.DataFrame


class BiLSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 16):  # reduced from 24
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _prepare_features(df: pd.DataFrame, include_signature: bool = True) -> pd.DataFrame:
    x = df.copy()
    x["hour"] = x["timestamp"].dt.hour
    x["day_of_week"] = x["timestamp"].dt.weekday
    x["feeder_historical_load"] = (
        x.groupby("feeder_id")["load_kw"].shift(1).fillna(x["load_kw"].median())
    )
    cols = [
        "hour",
        "day_of_week",
        "temperature_c",
        "is_holiday",
        "feeder_historical_load",
        "ev_registration_count",
    ]
    if include_signature:
        cols.append("inferred_home_charging_events")
    return x[cols]


def _build_sequences_per_feeder(
    df: pd.DataFrame, seq_len: int = 24, max_seqs_per_feeder: int = 200
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build sequences per feeder to avoid cross-feeder contamination and
    cap total sequences to keep training fast on a hackathon machine.
    seq_len reduced to 24 (1 day of hourly data) from 48.
    """
    all_x: list[np.ndarray] = []
    all_y: list[float] = []
    meta_rows: list[dict] = []

    for feeder_id, grp in df.groupby("feeder_id"):
        grp = grp.sort_values("timestamp")
        values = grp["load_kw"].to_numpy(dtype=np.float32)
        if len(values) <= seq_len:
            continue

        xs: list[np.ndarray] = []
        ys: list[float] = []
        feeder_meta: list[dict] = []

        target_row_indices = grp.index.to_numpy()
        for i in range(seq_len, len(values)):
            xs.append(values[i - seq_len : i].reshape(seq_len, 1))
            ys.append(float(values[i]))
            feeder_meta.append(
                {
                    "feeder_id": feeder_id,
                    "timestamp": grp["timestamp"].iloc[i],
                    # row index in the input df; since we reset_index in run_part_a
                    # this lines up with .iloc positions for alignment.
                    "row_idx": int(target_row_indices[i]),
                }
            )

        # cap per feeder to keep total manageable
        if len(xs) > max_seqs_per_feeder:
            xs = xs[-max_seqs_per_feeder:]
            ys = ys[-max_seqs_per_feeder:]
            feeder_meta = feeder_meta[-max_seqs_per_feeder:]

        all_x.extend(xs)
        all_y.extend(ys)
        meta_rows.extend(feeder_meta)

    if not all_x:
        empty_meta = pd.DataFrame(columns=["feeder_id", "timestamp", "row_idx"])
        return (
            np.zeros((1, seq_len, 1), dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            empty_meta,
        )

    meta_df = pd.DataFrame(meta_rows)
    meta_df["hour"] = pd.to_datetime(meta_df["timestamp"], errors="coerce").dt.hour
    return np.array(all_x, dtype=np.float32), np.array(all_y, dtype=np.float32), meta_df


def infer_ev_charging_events(smart_meter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer EV charging events from smart-meter voltage sag + current rise.

    Rules (per feeder, per timestamp):
    - voltage sag: voltage_v < 0.97 * rolling_15min_mean(voltage_v)
    - current rise: current_a > 1.20 * rolling_15min_mean(current_a)
    - infer EV event when both are true simultaneously

    Also aggregates inferred events per feeder_id per hour into
    `inferred_home_charging_events` (count) and broadcasts that count back
    to each timestamp row.
    """
    if smart_meter_df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "feeder_id",
                "voltage_v",
                "current_a",
                "ev_event_inferred",
                "hour",
                "inferred_home_charging_events",
            ]
        )

    df = smart_meter_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "feeder_id", "voltage_v", "current_a"])
    df = df.sort_values(["feeder_id", "timestamp"]).reset_index(drop=True)

    df = df.set_index("timestamp", drop=True)
    # Time-based rolling mean per feeder.
    roll_v = (
        df.groupby("feeder_id")["voltage_v"]
        .rolling("15min", min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll_c = (
        df.groupby("feeder_id")["current_a"]
        .rolling("15min", min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    ev_event_inferred = (df["voltage_v"] < roll_v * 0.97) & (df["current_a"] > roll_c * 1.20)
    df = df.reset_index()
    df["ev_event_inferred"] = ev_event_inferred.astype(int).to_numpy()
    df["hour"] = df["timestamp"].dt.hour

    hourly_counts = (
        df.groupby(["feeder_id", "hour"], as_index=False)["ev_event_inferred"]
        .sum()
        .rename(columns={"ev_event_inferred": "inferred_home_charging_events"})
    )
    df = df.merge(hourly_counts, on=["feeder_id", "hour"], how="left")
    df["inferred_home_charging_events"] = (
        df["inferred_home_charging_events"].fillna(0).astype(int)
    )
    return df


def _predict_in_batches(model: nn.Module, arr: np.ndarray, batch_size: int = 256) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            batch = torch.tensor(arr[start : start + batch_size], dtype=torch.float32)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def run_part_a(
    feeders: pd.DataFrame, feeder_timeseries: pd.DataFrame, smart_meter: pd.DataFrame
) -> PartAOutput:
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df = feeder_timeseries.merge(
        feeders[["feeder_id", "ev_registration_count", "rated_capacity_kva"]], on="feeder_id"
    )
    df = df.sort_values(["feeder_id", "timestamp"]).reset_index(drop=True)

    # ── EV signature inference (smart-meter rule-based detection) ────────
    smart_inferred = infer_ev_charging_events(smart_meter)
    hourly_events = (
        smart_inferred[["feeder_id", "hour", "inferred_home_charging_events"]]
        .drop_duplicates(subset=["feeder_id", "hour"])
    )

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    # Overwrite any synthetic/placeholder feeder_timeseries EV feature.
    df = df.drop(columns=["inferred_home_charging_events"], errors="ignore").merge(
        hourly_events, on=["feeder_id", "hour"], how="left"
    )
    df["inferred_home_charging_events"] = df["inferred_home_charging_events"].fillna(0).astype(int)

    # ── per-feeder time-based train/test split ─────────────────────────────
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for feeder_id, grp in df.groupby("feeder_id"):
        grp = grp.sort_values("timestamp")
        cut = int(len(grp) * 0.8)
        train_parts.append(grp.iloc[:cut])
        test_parts.append(grp.iloc[cut:])

    train_df = (
        pd.concat(train_parts, ignore_index=True)
        .sort_values(["feeder_id", "timestamp"])
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat(test_parts, ignore_index=True)
        .sort_values(["feeder_id", "timestamp"])
        .reset_index(drop=True)
    )

    # ── LightGBM / GBM base learner ─────────────────────────────────────────
    x_train = _prepare_features(train_df, include_signature=True)
    x_test = _prepare_features(test_df, include_signature=True)
    y_train, y_test = train_df["load_kw"], test_df["load_kw"]

    if LGBMRegressor:
        gbm = LGBMRegressor(
            n_estimators=50,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        gbm = GradientBoostingRegressor(random_state=42)
    gbm.fit(x_train, y_train)
    gbm_pred_train = gbm.predict(x_train)
    gbm_pred_test = gbm.predict(x_test)

    # ── BiLSTM base learner (per-feeder sequences, capped) ──────────────────
    SEQ_LEN = 24  # 1 day of hourly readings — faster than 48
    seq_x, seq_y, seq_meta_train = _build_sequences_per_feeder(
        train_df, seq_len=SEQ_LEN, max_seqs_per_feeder=150
    )
    test_seq_x, test_seq_y, seq_meta_test = _build_sequences_per_feeder(
        test_df, seq_len=SEQ_LEN, max_seqs_per_feeder=150
    )

    scaler = StandardScaler()
    seq_x_flat = scaler.fit_transform(seq_x.reshape(seq_x.shape[0], -1)).reshape(seq_x.shape)
    test_seq_x_flat = scaler.transform(
        test_seq_x.reshape(test_seq_x.shape[0], -1)
    ).reshape(test_seq_x.shape)

    x_t = torch.tensor(seq_x_flat, dtype=torch.float32)
    y_t = torch.tensor(seq_y, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=256,  # reduced from 512
        shuffle=True,
    )

    model = BiLSTMRegressor(hidden_size=16)  # smaller hidden size
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # higher lr = fewer epochs needed
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(3):  # 3 epochs, small batches → fast
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        print(f"  BiLSTM epoch {epoch + 1}/3 done")

    lstm_train_pred = _predict_in_batches(model, seq_x_flat, batch_size=256)
    lstm_test_pred = _predict_in_batches(model, test_seq_x_flat, batch_size=256)

    # ── Meta-learner (quantile regression at q=0.95) ─────────────────────────
    train_target_idx = seq_meta_train["row_idx"].to_numpy(dtype=int)
    test_target_idx = seq_meta_test["row_idx"].to_numpy(dtype=int)

    meta_train = pd.DataFrame(
        {
            "gbm": gbm_pred_train[train_target_idx],
            "bilstm": lstm_train_pred,
        }
    )
    meta_target = y_train.iloc[train_target_idx].to_numpy()
    meta_model = QuantileRegressor(quantile=0.95, alpha=0.01, solver="highs")
    meta_model.fit(meta_train, meta_target)

    meta_test = pd.DataFrame(
        {
            "gbm": gbm_pred_test[test_target_idx],
            "bilstm": lstm_test_pred,
        }
    )
    p95_pred = meta_model.predict(meta_test)
    y_eval = y_test.iloc[test_target_idx].to_numpy()
    pinball = mean_pinball_loss(y_eval, p95_pred, alpha=0.95)
    print(f"  Pinball loss (q=0.95): {pinball:.4f}")

    # ── Risk classification ───────────────────────────────────────────────────
    eval_df = test_df.iloc[test_target_idx].copy()
    eval_df["predicted_load_p95_kw"] = p95_pred
    eval_df["capacity_pct"] = 100 * eval_df["predicted_load_p95_kw"] / eval_df["rated_capacity_kva"]
    eval_df["status"] = np.where(
        eval_df["capacity_pct"] > 100,
        "CRITICAL",
        np.where(eval_df["capacity_pct"] > 85, "HIGH", "SAFE"),
    )
    print(f"  eval_df feeder_id unique: {eval_df['feeder_id'].nunique()} (expected 50)")
    # Produce exactly one risk status per feeder-hour (map UI expects this).
    hourly = (
        eval_df.assign(hour=eval_df["timestamp"].dt.hour)
        .groupby(["feeder_id", "hour"], as_index=False)
        .agg(
            predicted_load_p95_kw=("predicted_load_p95_kw", "max"),
            capacity_pct=("capacity_pct", "max"),
            rated_capacity_kva=("rated_capacity_kva", "first"),
        )
    )
    hourly["status"] = np.where(
        hourly["capacity_pct"] > 100,
        "CRITICAL",
        np.where(hourly["capacity_pct"] > 85, "HIGH", "SAFE"),
    )

    # ── Nudge recommendations ─────────────────────────────────────────────────
    stressed = hourly[hourly["status"].isin(["HIGH", "CRITICAL"])].copy()
    stressed["discount_inr_per_kwh"] = np.where(stressed["status"].eq("CRITICAL"), 3.0, 1.5)
    stressed["projected_load_shift_pct"] = (
        stressed["discount_inr_per_kwh"] * 8
    ).round(0).astype(int)
    stressed["time_window"] = (
        stressed["hour"].astype(str) + ":00-" + (stressed["hour"] + 2).astype(str) + ":00"
    )
    stressed["off_peak_window"] = "23:00-05:00"
    stressed["nudge_text"] = (
        stressed["feeder_id"]
        + " flagged "
        + stressed["status"]
        + " "
        + stressed["time_window"]
        + " — recommend Rs."
        + stressed["discount_inr_per_kwh"].astype(str)
        + "/kWh discount for charging delayed to "
        + stressed["off_peak_window"]
        + ", projected "
        + stressed["projected_load_shift_pct"].astype(str)
        + "% load shift"
    )

    # ── Counterfactuals ───────────────────────────────────────────────────────
    critical = hourly[hourly["status"].eq("CRITICAL")].copy()
    if not critical.empty:
        critical["users_delay_2h"] = np.ceil(
            (critical["capacity_pct"] - 90) / 1.8
        ).clip(lower=10)
        critical["site_addition_effect_pct"] = 8
        critical["combined_effect_pct"] = 18
        critical["counterfactual_text"] = (
            "To move from CRITICAL to HIGH: delay "
            + critical["users_delay_2h"].astype(int).astype(str)
            + " users by 2h, or add nearest vetted public site (~8% relief),"
            + " or combine both (~18% relief)."
        )
    else:
        critical["users_delay_2h"] = pd.Series(dtype=float)
        critical["site_addition_effect_pct"] = pd.Series(dtype=float)
        critical["combined_effect_pct"] = pd.Series(dtype=float)
        critical["counterfactual_text"] = pd.Series(dtype=str)

    # ── With-vs-without home charging signal ─────────────────────────────────
    x_train_wo = _prepare_features(train_df, include_signature=False)
    x_test_wo = _prepare_features(test_df, include_signature=False)
    model_wo = GradientBoostingRegressor(random_state=42, n_estimators=50)
    model_wo.fit(x_train_wo, y_train)
    pred_wo = model_wo.predict(x_test_wo.iloc[test_target_idx])
    with_without = eval_df[["timestamp", "feeder_id", "predicted_load_p95_kw"]].copy()
    with_without["pred_without_signature_kw"] = pred_wo
    with_without["delta_kw"] = (
        with_without["predicted_load_p95_kw"] - with_without["pred_without_signature_kw"]
    )

    return PartAOutput(
        feeder_hourly_risk=hourly,
        nudge_recommendations=stressed,
        counterfactuals=critical,
        with_without_signal=with_without,
    )