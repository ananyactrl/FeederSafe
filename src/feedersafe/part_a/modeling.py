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
    def __init__(self, hidden_size: int = 32):
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
    x["feeder_historical_load"] = x.groupby("feeder_id")["load_kw"].shift(1).fillna(x["load_kw"].median())
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


def _build_sequences(load_series: pd.Series, seq_len: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    values = load_series.to_numpy(dtype=np.float32)
    if len(values) <= seq_len:
        return np.zeros((1, seq_len, 1), dtype=np.float32), np.array([values[-1]], dtype=np.float32)
    xs, ys = [], []
    for i in range(seq_len, len(values)):
        xs.append(values[i - seq_len : i].reshape(seq_len, 1))
        ys.append(values[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _predict_in_batches(model: nn.Module, arr: np.ndarray, batch_size: int = 512) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            batch = torch.tensor(arr[start : start + batch_size], dtype=torch.float32)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def run_part_a(feeders: pd.DataFrame, feeder_timeseries: pd.DataFrame, smart_meter: pd.DataFrame) -> PartAOutput:
    df = feeder_timeseries.merge(feeders[["feeder_id", "ev_registration_count", "rated_capacity_kva"]], on="feeder_id")
    df = df.sort_values(["feeder_id", "timestamp"]).reset_index(drop=True)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    x_train = _prepare_features(train_df, include_signature=True)
    x_test = _prepare_features(test_df, include_signature=True)
    y_train, y_test = train_df["load_kw"], test_df["load_kw"]

    if LGBMRegressor:
        gbm = LGBMRegressor(n_estimators=220, learning_rate=0.05, num_leaves=31, random_state=42)
    else:
        gbm = GradientBoostingRegressor(random_state=42)
    gbm.fit(x_train, y_train)
    gbm_pred_train = gbm.predict(x_train)
    gbm_pred_test = gbm.predict(x_test)

    seq_x, seq_y = _build_sequences(train_df["load_kw"], seq_len=48)
    test_seq_x, _ = _build_sequences(test_df["load_kw"], seq_len=48)
    scaler = StandardScaler()
    seq_x_flat = scaler.fit_transform(seq_x.reshape(seq_x.shape[0], -1)).reshape(seq_x.shape)
    test_seq_x_flat = scaler.transform(test_seq_x.reshape(test_seq_x.shape[0], -1)).reshape(test_seq_x.shape)
    x_t = torch.tensor(seq_x_flat, dtype=torch.float32)
    y_t = torch.tensor(seq_y, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=512,
        shuffle=True,
    )

    model = BiLSTMRegressor(hidden_size=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(5):  # Short training for hackathon prototype speed.
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    lstm_train_pred = _predict_in_batches(model, seq_x_flat, batch_size=512)
    lstm_test_pred = _predict_in_batches(model, test_seq_x_flat, batch_size=512)

    meta_train = pd.DataFrame(
        {"gbm": gbm_pred_train[-len(lstm_train_pred) :], "bilstm": lstm_train_pred}
    )
    meta_target = y_train.iloc[-len(meta_train) :].to_numpy()
    meta_model = QuantileRegressor(quantile=0.95, alpha=0.01, solver="highs")
    meta_model.fit(meta_train, meta_target)

    aligned_gbm_test = gbm_pred_test[-len(lstm_test_pred) :]
    meta_test = pd.DataFrame({"gbm": aligned_gbm_test, "bilstm": lstm_test_pred})
    p95_pred = meta_model.predict(meta_test)
    y_eval = y_test.iloc[-len(p95_pred) :].to_numpy()
    _ = mean_pinball_loss(y_eval, p95_pred, alpha=0.95)

    eval_df = test_df.iloc[-len(p95_pred) :].copy()
    eval_df["predicted_load_p95_kw"] = p95_pred
    eval_df["capacity_pct"] = 100 * eval_df["predicted_load_p95_kw"] / eval_df["rated_capacity_kva"]
    eval_df["status"] = np.where(
        eval_df["capacity_pct"] > 100,
        "CRITICAL",
        np.where(eval_df["capacity_pct"] > 85, "HIGH", "SAFE"),
    )
    hourly = (
        eval_df.assign(hour=eval_df["timestamp"].dt.hour)
        .groupby(["feeder_id", "hour", "status"], as_index=False)
        .agg(
            predicted_load_p95_kw=("predicted_load_p95_kw", "mean"),
            capacity_pct=("capacity_pct", "mean"),
            rated_capacity_kva=("rated_capacity_kva", "first"),
        )
    )

    stressed = hourly[hourly["status"].isin(["HIGH", "CRITICAL"])].copy()
    stressed["discount_inr_per_kwh"] = np.where(stressed["status"].eq("CRITICAL"), 3.0, 1.5)
    stressed["projected_load_shift_pct"] = (stressed["discount_inr_per_kwh"] * 8).round(0).astype(int)
    stressed["time_window"] = stressed["hour"].astype(str) + ":00-" + (stressed["hour"] + 2).astype(str) + ":00"
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

    critical = hourly[hourly["status"].eq("CRITICAL")].copy()
    if not critical.empty:
        critical["users_delay_2h"] = np.ceil((critical["capacity_pct"] - 90) / 1.8).clip(lower=10)
    else:
        critical["users_delay_2h"] = []
    critical["site_addition_effect_pct"] = 8
    critical["combined_effect_pct"] = 18
    critical["counterfactual_text"] = (
        "To move from CRITICAL to HIGH: delay "
        + critical["users_delay_2h"].astype(int).astype(str)
        + " users by 2h, or add nearest vetted public site (~8% relief), or combine both (~18% relief)."
    )

    # With-vs-without home charging signal for explainability.
    x_train_wo = _prepare_features(train_df, include_signature=False)
    x_test_wo = _prepare_features(test_df, include_signature=False)
    model_wo = GradientBoostingRegressor(random_state=42)
    model_wo.fit(x_train_wo, y_train)
    pred_wo = model_wo.predict(x_test_wo[-len(p95_pred) :])
    with_without = eval_df[["timestamp", "feeder_id", "predicted_load_p95_kw"]].copy()
    with_without["pred_without_signature_kw"] = pred_wo
    with_without["delta_kw"] = with_without["predicted_load_p95_kw"] - with_without["pred_without_signature_kw"]

    return PartAOutput(
        feeder_hourly_risk=hourly,
        nudge_recommendations=stressed,
        counterfactuals=critical,
        with_without_signal=with_without,
    )

