"""Entry point: per-fold SKATER partitions (skater-ALL output layout).

Runs the original SKATER algorithm once per leave-one-out fold on the
training years only — the held-out epidemic year is strictly excluded
from BOTH the MST cost matrix and the pruning objective, so the learned
partition never sees the test year.  Each fold's outputs mirror the
all-years `skater` stage exactly (same CSV/JSON set), only written to a
separate folder, so the model stage can train MLPs without re-running
SKATER.

SKATER strategy parameters (mst_cost, prune_obj, N_min, S_min, seed, …)
come from params.yaml[skater]; only `epidemic_years` and the stop
conditions are overridden per fold via params.yaml[skater_cv].  Outputs
are namespaced by params.yaml[skater].run_label — same convention as
the all-years `skater` stage — so sweeping S_min (or any other skater
param) across multiple run_labels keeps every sweep's results on disk
side by side instead of overwriting.

Run:  python -m src.multiscale.skater_partitions

Outputs (results/multiscale/partitions/<run_label>/fold_<year>/):
  cluster_assignments.csv, q_trajectory.csv, cluster_diagnostics.csv,
  adjacency_edges.csv, mst_edges.csv, run_params.json, stop_info.json,
  fold_meta.json (test_year + train_years).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.skater.adjacency import build_adjacency
from src.skater.config import SkaterConfig, load_config as load_skater_config
from src.skater.data import SkaterData, load_data
from src.skater.run import run_pipeline

from .config import SkaterCvConfig, load_skater_cv_config

logger = logging.getLogger(__name__)
_OUT_BASE = Path("results/multiscale/partitions")


def _fold_skater_config(
    base: SkaterConfig, train_years: list[str], test_year: str,
    cv_cfg: SkaterCvConfig,
) -> SkaterConfig:
    """Build this fold's SKATER config from the base + CV overrides.

    Args:
        base:        params.yaml[skater] config.
        train_years: Epidemic years SKATER may learn from.
        test_year:   Held-out year (only used to label the run).
        cv_cfg:      CV config carrying the per-fold stop overrides.

    Returns:
        A SkaterConfig copy with epidemic_years + stop conditions set.
    """
    return base.model_copy(
        update={
            "epidemic_years": train_years,
            "stop_local_degradation": cv_cfg.cv_stop_local_degradation,
            "global_degradation_threshold": cv_cfg.cv_global_degradation_threshold,
            "C_max": cv_cfg.cv_c_max,
            "run_label": f"cv_fold_{test_year}",
        }
    )


def _strict_mst_matrix(
    skdata: SkaterData, mst_cost: str, test_year: str
) -> np.ndarray:
    """MST cost matrix with the held-out year strictly removed.

    Mirrors the original pipeline's matrix selection, then drops the
    held-out epidemic year's biweeks from the all-biweek egg matrix.
    The epidemic dengue matrix is already restricted to the training
    years via the per-fold `epidemic_years` override.

    Args:
        skdata:    Per-fold SkaterData.
        mst_cost:  cfg.mst_cost strategy key.
        test_year: Held-out epidemic year for this fold.

    Returns:
        The matrix passed to build_mst.
    """
    if mst_cost == "case_corr_dist":
        return skdata.dengue_epic
    mask = np.array(
        [bw.rsplit("W", 1)[0] != test_year for bw in skdata.all_biweeks]
    )
    return skdata.eggs_all[:, mask]


def main() -> None:
    """Run SKATER once per fold and persist skater-ALL-style outputs."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    base = load_skater_config()
    cv_cfg = load_skater_cv_config()
    logger.info("Fold years: %s", cv_cfg.fold_years)

    # ── Queen contiguity graph — geometry only, build once ────────────
    logger.info("Building adjacency graph (once)…")
    seed_skdata = load_data(base)
    adjacency = build_adjacency(
        seed_skdata.geojson, sector_filter=set(seed_skdata.sector_list)
    )

    for test_year in cv_cfg.fold_years:
        train_years = [y for y in cv_cfg.fold_years if y != test_year]
        logger.info("=== Fold: test=%s  train=%s ===", test_year, train_years)

        skcfg = _fold_skater_config(base, train_years, test_year, cv_cfg)
        skdata = load_data(skcfg)
        mst_matrix = _strict_mst_matrix(skdata, skcfg.mst_cost, test_year)

        out_dir = _OUT_BASE / base.run_label / f"fold_{test_year}"
        run_pipeline(skcfg, skdata, adjacency, out_dir, mst_matrix)

        with open(out_dir / "fold_meta.json", "w") as fh:
            json.dump(
                {"test_year": test_year, "train_years": train_years}, fh,
                indent=2,
            )

    logger.info(
        "All fold partitions written under %s", _OUT_BASE / base.run_label
    )


if __name__ == "__main__":
    main()
