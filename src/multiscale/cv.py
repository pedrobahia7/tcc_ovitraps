"""Leave-one-epidemic-year-out cross-validation over spatial units.

Given a unit-membership map and the loaded SectorData, trains the frozen
MLP for every (unit, fold) pair and evaluates it on the held-out
epidemic year, alongside a naive persistence baseline.  Scale-agnostic:
the same routine drives city, district, sector and SKATER-region units.

Inputs:
  units — {unit_key: [sector_id, ...]}; SectorData; MultiscaleConfig.
Outputs:
  DataFrame of per-(unit, fold) test metrics + unit population.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .aggregate import SectorData, aggregate_unit
from .config import MultiscaleConfig
from .features import build_unit_features
from .model import (
    compute_metrics,
    naive_prediction,
    predict_mlp,
    train_mlp,
)

logger = logging.getLogger(__name__)

# Minimum usable rows required to train / evaluate a single (unit, fold).
_MIN_TRAIN_ROWS = 10
_MIN_TEST_ROWS = 5


def _fit_eval_fold(
    feats: pd.DataFrame,
    test_year: str,
    cfg: MultiscaleConfig,
    base_record: dict,
) -> dict | None:
    """Train on non-test fold years, evaluate on the held-out year.

    Args:
        feats:       Unit feature rows already restricted to fold years.
        test_year:   Epidemic year held out for testing.
        cfg:         Multiscale configuration.
        base_record: Fields common to the row (scale, unit, pop, …) that
                     are copied into the returned record.

    Returns:
        A metric record dict, or None if train/test rows are too few.
    """
    train_df = feats[feats["year"] != test_year]
    test_df = feats[feats["year"] == test_year]
    if len(train_df) < _MIN_TRAIN_ROWS or len(test_df) < _MIN_TEST_ROWS:
        return None

    scaler, mlp = train_mlp(train_df, cfg)
    y_true = test_df["target"].to_numpy(dtype=float)
    y_mlp = predict_mlp(scaler, mlp, test_df, cfg)
    y_naive = naive_prediction(test_df)

    mlp_m = compute_metrics(y_true, y_mlp)
    naive_m = compute_metrics(y_true, y_naive)
    return {
        **base_record,
        "fold_year": test_year,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "mlp_rmse": mlp_m["rmse"],
        "mlp_mae": mlp_m["mae"],
        "mlp_r2": mlp_m["r2"],
        "mlp_mape": mlp_m["mape"],
        "naive_rmse": naive_m["rmse"],
        "naive_mae": naive_m["mae"],
    }


def _unit_feats(
    members: list[str],
    data: SectorData,
    cfg: MultiscaleConfig,
    fold_years: list[str],
) -> pd.DataFrame:
    """Aggregate a unit and build its fold-year-restricted feature rows."""
    egg_series, dengue_series = aggregate_unit(members, data)
    feats = build_unit_features(egg_series, dengue_series, data.biweeks, cfg)
    return feats[feats["year"].isin(fold_years)]


def evaluate_unit(
    unit_key: str,
    members: list[str],
    data: SectorData,
    cfg: MultiscaleConfig,
    fold_years: list[str],
    scale: str,
) -> list[dict]:
    """Run LOYO CV for one spatial unit across all fold years.

    Args:
        unit_key:   Identifier for this unit (sector id, district name, …).
        members:    Member sector IDs.
        data:       Loaded SectorData.
        cfg:        Multiscale (model) configuration.
        fold_years: Epidemic years to fold over.
        scale:      Scale label ('city' | 'district' | 'sector' | 'skater').

    Returns:
        One record dict per usable fold (possibly empty list).
    """
    feats = _unit_feats(members, data, cfg, fold_years)
    if feats.empty:
        return []

    unit_pop = float(data.pop[[data.idx[s] for s in members]].sum())
    base = {"scale": scale, "unit": unit_key, "n_sectors": len(members),
            "pop": unit_pop}
    records = [
        rec
        for test_year in fold_years
        if (rec := _fit_eval_fold(feats, test_year, cfg, base)) is not None
    ]
    return records


def evaluate_unit_fold(
    unit_key: str,
    members: list[str],
    data: SectorData,
    cfg: MultiscaleConfig,
    fold_years: list[str],
    test_year: str,
    extra: dict | None = None,
) -> dict | None:
    """Evaluate one unit for a single, pre-determined held-out year.

    Used by the SKATER-CV stage, where the partition is learned per fold
    and must only be tested on that fold's held-out year (no re-LOYO).

    Args:
        unit_key:   Identifier for this region unit.
        members:    Member sector IDs.
        data:       Loaded SectorData.
        cfg:        Multiscale (model) configuration.
        fold_years: Epidemic years to fold over (train = fold_years minus
                    test_year).
        test_year:  The single held-out epidemic year for this partition.
        extra:      Optional extra fields merged into the record (e.g. C).

    Returns:
        A metric record dict, or None if unusable.
    """
    feats = _unit_feats(members, data, cfg, fold_years)
    if feats.empty:
        return None
    unit_pop = float(data.pop[[data.idx[s] for s in members]].sum())
    base = {"scale": "skater", "unit": unit_key, "n_sectors": len(members),
            "pop": unit_pop, **(extra or {})}
    return _fit_eval_fold(feats, test_year, cfg, base)


def run_scale(
    scale: str,
    units: dict[str, list[str]],
    data: SectorData,
    cfg: MultiscaleConfig,
    fold_years: list[str],
    log_every: int = 500,
) -> pd.DataFrame:
    """Evaluate every unit at one spatial scale.

    Args:
        scale:      Scale label used in output rows.
        units:      Mapping unit_key → member sector IDs.
        data:       Loaded SectorData.
        cfg:        Multiscale (model) configuration.
        fold_years: Epidemic years to fold over.
        log_every:  Emit a progress log every this many units.

    Returns:
        DataFrame of per-(unit, fold) metric records.  Logs how many
        units yielded no usable fold so truncation is never silent.
    """
    logger.info("Scale '%s': evaluating %d units…", scale, len(units))
    all_records: list[dict] = []
    n_empty = 0
    for i, (key, members) in enumerate(units.items(), start=1):
        recs = evaluate_unit(key, members, data, cfg, fold_years, scale)
        if recs:
            all_records.extend(recs)
        else:
            n_empty += 1
        if i % log_every == 0:
            logger.info("  %s: %d/%d units processed", scale, i, len(units))

    logger.info(
        "Scale '%s' done: %d units usable, %d skipped (no usable fold)",
        scale, len(units) - n_empty, n_empty,
    )
    return pd.DataFrame(all_records)


def pop_weighted_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-unit-per-fold records to per-scale summary rows.

    Computes population-weighted mean test RMSE across units per fold,
    then averages over folds, for both the MLP and the naive baseline.

    Args:
        records: Output of run_scale (or a concat across scales).

    Returns:
        DataFrame [scale, n_units, mean_n_sectors, pop_wt_rmse,
        naive_pop_wt_rmse, mean_r2] — one row per scale.
    """
    if records.empty:
        return pd.DataFrame()

    def _wmean(group: pd.DataFrame, col: str) -> float:
        w = group["pop"].to_numpy(dtype=float)
        v = group[col].to_numpy(dtype=float)
        return float(np.average(v, weights=w)) if w.sum() > 0 else float("nan")

    out = []
    for scale, sdf in records.groupby("scale"):
        # Per fold: population-weighted mean RMSE across units
        per_fold_mlp, per_fold_naive = [], []
        for _, fdf in sdf.groupby("fold_year"):
            per_fold_mlp.append(_wmean(fdf, "mlp_rmse"))
            per_fold_naive.append(_wmean(fdf, "naive_rmse"))
        out.append(
            {
                "scale": scale,
                "n_units": sdf["unit"].nunique(),
                "mean_n_sectors": float(sdf["n_sectors"].mean()),
                "pop_wt_rmse": float(np.mean(per_fold_mlp)),
                "naive_pop_wt_rmse": float(np.mean(per_fold_naive)),
                "mean_r2": float(sdf["mlp_r2"].mean()),
            }
        )
    return pd.DataFrame(out)
