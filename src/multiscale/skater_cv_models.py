"""Entry point: MLP models on cached SKATER-CV partitions.

Reads the per-fold partitions produced by src.multiscale.skater_partitions
(never re-runs SKATER) and trains one frozen MLP per region, for EVERY C
step including C=1, on the fold's training years, tested on its held-out
year.  Region counts per C: C=1 → 1 region (one MLP per fold), C=2 → 2
regions per fold, and so on.

Because this stage only reads cached partitions + the MLP config, the
model architecture / feature lags can be changed and re-run without
recomputing any SKATER partition.

Run:  python -m src.multiscale.skater_cv_models

Both the input partitions and this stage's own outputs are namespaced
by params.yaml[skater].run_label — same convention as the all-years
`skater` stage — so sweeping S_min (or any other skater param) across
multiple run_labels keeps every sweep's results on disk side by side.

Inputs (results/multiscale/partitions/<run_label>/fold_<year>/):
  cluster_assignments.csv, fold_meta.json, stop_info.json
Outputs (results/multiscale/skater_cv/<run_label>/):
  metrics_skater.csv — per-(fold, C, region) test metrics + naive.
  skater_by_c.csv    — population-weighted mean test RMSE/Spearman per C.
  folds_stop_info.csv — SKATER termination reason / final C per fold.
  predictions.csv    — raw [unit, fold_year, C, biweek, y_true, y_pred]
    rows for every region-fold, used to render the predict-vs-target
    panel in scripts/skater_cv_rmse_map.py.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.skater.config import load_config as load_skater_config

from .aggregate import load_sector_data
from .config import load_multiscale_config, load_skater_cv_config
from .cv import evaluate_unit_fold

logger = logging.getLogger(__name__)
_PART_BASE = Path("results/multiscale/partitions")
_OUT_BASE = Path("results/multiscale/skater_cv")


def _per_c_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted mean test RMSE/Spearman per C, pooled across folds.

    Args:
        records: metrics_skater rows (must contain C, pop, *_rmse,
            mlp_spearman).

    Returns:
        DataFrame [C, n_regions, n_folds, pop_wt_rmse, naive_pop_wt_rmse,
        mean_r2, pop_wt_spearman] sorted by C.
    """
    rows = []
    for c_value, cdf in records.groupby("C"):
        w = cdf["pop"].to_numpy(dtype=float)
        rows.append(
            {
                "C": int(c_value),
                "n_regions": len(cdf),
                "n_folds": cdf["fold_year"].nunique(),
                "pop_wt_rmse": float(np.average(cdf["mlp_rmse"], weights=w)),
                "naive_pop_wt_rmse": float(
                    np.average(cdf["naive_rmse"], weights=w)
                ),
                "mean_r2": float(cdf["mlp_r2"].mean()),
                "pop_wt_spearman": float(
                    np.average(cdf["mlp_spearman"], weights=w)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("C").reset_index(drop=True)


def _fold_dirs(part_dir: Path) -> list[Path]:
    """Return the per-fold partition directories, sorted by name."""
    return sorted(p for p in part_dir.glob("fold_*") if p.is_dir())


def main() -> None:
    """Train region MLPs on every cached partition and persist metrics."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    cfg = load_multiscale_config()
    cv_cfg = load_skater_cv_config()
    run_label = load_skater_config().run_label
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    part_dir = _PART_BASE / run_label
    out_dir = _OUT_BASE / run_label
    fold_dirs = _fold_dirs(part_dir)
    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold partitions under {part_dir}. "
            "Run `python -m src.multiscale.skater_partitions` first."
        )

    data = load_sector_data()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    stop_rows: list[dict] = []

    for fold_dir in fold_dirs:
        meta = json.loads((fold_dir / "fold_meta.json").read_text())
        test_year = meta["test_year"]
        stop = json.loads((fold_dir / "stop_info.json").read_text())
        stop_rows.append(
            {
                "fold_year": test_year,
                "stop_reason": stop["reason"],
                "c_final": stop["C_final"],
            }
        )

        asg = pd.read_csv(
            fold_dir / "cluster_assignments.csv", dtype={"sector_id": str}
        )
        logger.info(
            "Fold %s: C=1..%d partitions loaded", test_year, int(asg["C"].max())
        )

        # ── Every C step, including C=1 (single whole-training region) ─
        for c_value, cdf in asg.groupby("C"):
            for cid, gg in cdf.groupby("cluster_id"):
                members = gg["sector_id"].tolist()
                rec, pred_df = evaluate_unit_fold(
                    unit_key=f"C{c_value}__c{cid}",
                    members=members,
                    data=data,
                    cfg=cfg,
                    fold_years=cv_cfg.fold_years,
                    test_year=test_year,
                    extra={"C": int(c_value)},
                    return_predictions=True,
                )
                if rec is not None:
                    all_records.append(rec)
                    all_predictions.append(pred_df)
        logger.info(
            "  fold %s: %d region records so far", test_year, len(all_records)
        )

    # ── Persist ───────────────────────────────────────────────────────
    records = pd.DataFrame(all_records)
    records.to_csv(out_dir / "metrics_skater.csv", index=False)
    logger.info("Saved metrics_skater.csv (%d rows)", len(records))

    pd.DataFrame(stop_rows).to_csv(
        out_dir / "folds_stop_info.csv", index=False
    )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    logger.info("Saved predictions.csv (%d rows)", len(predictions))

    by_c = _per_c_summary(records)
    by_c.to_csv(out_dir / "skater_by_c.csv", index=False)
    logger.info("Saved skater_by_c.csv\n%s", by_c.to_string(index=False))


if __name__ == "__main__":
    main()
