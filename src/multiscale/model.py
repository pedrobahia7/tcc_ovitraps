"""Frozen MLP training, naive baseline and regression metrics.

A single fixed architecture (from MultiscaleConfig) is trained per unit
with a StandardScaler front-end and no hyper-parameter search.  The
naive baseline is last-value persistence (the lag-1 EB rate), matching
the existing citywide MLP script.

Inputs:
  Train/test feature frames from features.build_unit_features.
Outputs:
  Fitted (scaler, mlp), clamped predictions, metric dicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .config import MultiscaleConfig


def train_mlp(
    train_df: pd.DataFrame, cfg: MultiscaleConfig
) -> tuple[StandardScaler, MLPRegressor]:
    """Fit the frozen MLP on a unit's training rows.

    Args:
        train_df: Training rows (target + cfg.feature_order columns).
        cfg:      Multiscale configuration (frozen architecture, seed).

    Returns:
        (scaler, mlp) — the fitted StandardScaler and MLPRegressor.
    """
    x = train_df[cfg.feature_order].to_numpy(dtype=float)
    y = train_df["target"].to_numpy(dtype=float)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    mlp = MLPRegressor(
        hidden_layer_sizes=cfg.hidden_tuple,
        alpha=cfg.alpha,
        activation="relu",
        solver="adam",
        early_stopping=True,
        max_iter=cfg.max_iter,
        random_state=cfg.seed,
    )
    mlp.fit(x_scaled, y)
    return scaler, mlp


def predict_mlp(
    scaler: StandardScaler,
    mlp: MLPRegressor,
    df: pd.DataFrame,
    cfg: MultiscaleConfig,
) -> np.ndarray:
    """Predict clamped (non-negative) EB rates for the given rows.

    Args:
        scaler: Fitted StandardScaler.
        mlp:    Fitted MLPRegressor.
        df:     Rows to predict (must contain cfg.feature_order columns).
        cfg:    Multiscale configuration.

    Returns:
        Non-negative predictions, shape (len(df),).
    """
    x_scaled = scaler.transform(df[cfg.feature_order].to_numpy(dtype=float))
    return np.maximum(mlp.predict(x_scaled), 0.0)


def naive_prediction(df: pd.DataFrame) -> np.ndarray:
    """Last-value persistence baseline: predict the lag-1 EB rate.

    Args:
        df: Rows containing the 'eb_rate_lag1' feature column.

    Returns:
        The lag-1 EB rate as the naive forecast, shape (len(df),).
    """
    return df["eb_rate_lag1"].to_numpy(dtype=float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, R², MAPE and Spearman r for one prediction set.

    Args:
        y_true: Observed EB rates.
        y_pred: Predicted EB rates.

    Returns:
        Dict with keys rmse, mae, r2, mape, spearman.  r2 and spearman
        are NaN when len(y_true) < 2 or either array is constant (no
        rank variation to correlate).
    """
    can_correlate = (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    )
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mape": float(
            np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        ),
        "spearman": (
            float(spearmanr(y_true, y_pred).statistic)
            if can_correlate else float("nan")
        ),
    }
