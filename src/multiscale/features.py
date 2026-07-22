"""Lag + seasonality feature construction for a spatial unit.

Turns a unit's aggregated egg and dengue series (aligned to the shared
biweek axis) into a supervised feature matrix.  Mirrors the recipe used
by the existing citywide / sector MLP scripts: past EB-rate lags, past
egg lags and cyclical seasonality predict the current-biweek EB rate.

Inputs:
  egg_series, dengue_series — unit series over all biweeks (aggregate.py)
Outputs:
  DataFrame [biweek, year, target, <feature_order>] with complete rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MultiscaleConfig


def _week_number(biweek: str) -> int:
    """Extract the numeric week index from a biweek label ('2015_16W04')."""
    return int(biweek.split("W")[1])


def build_unit_features(
    egg_series: np.ndarray,
    dengue_series: np.ndarray,
    biweeks: list[str],
    cfg: MultiscaleConfig,
) -> pd.DataFrame:
    """Build the supervised feature matrix for one spatial unit.

    Lags are created by positional shift on the chronologically-ordered
    biweek sequence (identical to the existing citywide MLP), then rows
    with any missing feature or target are dropped.

    Args:
        egg_series:    Unit IDW egg series, shape (n_biweeks,).
        dengue_series: Unit EB dengue-rate series, shape (n_biweeks,).
        biweeks:       Chronological biweek labels, length n_biweeks.
        cfg:           Multiscale configuration (lag windows, feature order).

    Returns:
        DataFrame with columns [biweek, year, target, *cfg.feature_order],
        one row per usable biweek (NaN lag rows removed).
    """
    df = pd.DataFrame(
        {
            "biweek": biweeks,
            "target": dengue_series,
            "_egg": egg_series,
        }
    )

    # ── Lag features ──────────────────────────────────────────────────
    for k in cfg.eb_lags:
        df[f"eb_rate_lag{k}"] = df["target"].shift(k)
    for k in cfg.egg_lags:
        df[f"egg_lag{k}"] = df["_egg"].shift(k)

    # ── Cyclical seasonality ──────────────────────────────────────────
    week_num = df["biweek"].map(_week_number)
    df["week_sin"] = np.sin(2 * np.pi * week_num / 52)
    df["week_cos"] = np.cos(2 * np.pi * week_num / 52)

    # ── Epidemic-year label for fold filtering ────────────────────────
    df["year"] = df["biweek"].str.rsplit("W", n=1).str[0]

    keep = ["biweek", "year", "target", *cfg.feature_order]
    df = df[keep].dropna().reset_index(drop=True)
    return df
