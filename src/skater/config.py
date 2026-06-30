"""SKATER pipeline configuration (Pydantic v2).

All tuneable parameters live in params.yaml under the `skater:` key.
This module loads them into a validated SkaterConfig object that every
other module receives — no magic strings scattered through the codebase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class SkaterConfig(BaseModel):
    """All tunable parameters for the SKATER regionalization pipeline.

    Loaded from params.yaml[skater:] via load_config().
    Changing a value here (or in params.yaml) is the only thing needed
    to swap strategies or adjust guard thresholds.
    """

    # ── Strategy selectors ────────────────────────────────────────────
    mst_cost: Literal[
        "egg_corr_dist", "case_corr_dist",
        "egg_spearman", "case_spearman",
        "egg_euclidean", "case_euclidean",
    ] = Field(
        "egg_corr_dist",
        description=(
            "Which dissimilarity metric to use when building the MST. "
            "Valid values: "
            "'egg_corr_dist'  — 1 − Pearson(eggs_i, eggs_j) over all biweeks; "
            "'case_corr_dist' — 1 − Pearson(dengue_i, dengue_j) over epidemic "
            "biweeks only, lag=0; "
            "'egg_spearman'   — 1 − Spearman(eggs_i, eggs_j) over all biweeks; "
            "'case_spearman'  — 1 − Spearman(dengue_i, dengue_j) over epidemic "
            "biweeks only; "
            "'egg_euclidean'  — L2‖eggs_i − eggs_j‖, all biweeks "
            "(Assunção 2006 paper-style); "
            "'case_euclidean' — L2‖cases_i − cases_j‖, epidemic biweeks "
            "(paper-style, cases-only run)."
        ),
    )
    prune_obj: Literal[
        "corr_q", "corr_q_spearman", "egg_ssd", "case_ssd", "eggs_dengue_corr"
    ] = Field(
        "corr_q",
        description=(
            "Which objective to maximise during greedy pruning. "
            "Valid values: "
            "'corr_q'          — best signed Pearson(lagged_eggs, dengue_rate) (custom); "
            "'corr_q_spearman' — best signed Spearman(lagged_eggs, dengue_rate), "
            "rank-based and outlier-robust; "
            "'egg_ssd'         — negative within-cluster SSD of egg time series "
            "(Assunção 2006 paper-style, eggs-only run); "
            "'case_ssd'        — negative within-cluster SSD of dengue-rate time series "
            "(paper-style, cases-only run); "
            "'eggs_dengue_corr' — backwards-compat alias for 'corr_q'."
        ),
    )

    # ── Lag search window ─────────────────────────────────────────────
    k_min: int = Field(
        2, ge=1,
        description="Minimum lag (in biweeks) tested by the pruning objective.",
    )
    k_max: int = Field(
        6, ge=1,
        description="Maximum lag (in biweeks) tested by the pruning objective.",
    )

    # ── Stopping conditions ───────────────────────────────────────────
    C_max: Optional[int] = Field(
        30,
        ge=2,
        description=(
            "Hard ceiling on the number of clusters produced. "
            "None = no ceiling — the algorithm runs until another "
            "stopping condition fires (global degradation, local "
            "degradation, or no valid cut remains)."
        ),
    )
    stop_local_degradation: bool = Field(
        False,
        description=(
            "Reject any candidate cut where BOTH resulting child "
            "clusters score strictly below their parent (q_a < q_parent "
            "AND q_b < q_parent). Cuts that raise at least one child's "
            "score are still allowed — concentrating purity into one "
            "child at the expense of the other is acceptable. "
            "If every remaining cut is rejected this way the algorithm "
            "stops, just as when no valid cut exists."
        ),
    )
    global_degradation_threshold: Optional[float] = Field(
        None,
        description=(
            "Stop the algorithm when the best available cut would change "
            "the global objective Q by less than this value "
            "(i.e. best_dQ < threshold). "
            "None = condition disabled. "
            "0.0 = stop as soon as any cut decreases Q. "
            "Positive values enforce a minimum improvement per step; "
            "negative values tolerate mild degradation before stopping."
        ),
    )
    N_min: int = Field(
        20, ge=1,
        description=(
            "Minimum valid (eggs, dengue) pairs a cluster must have. "
            "Cuts that would produce a cluster below this are rejected."
        ),
    )
    S_min: int = Field(
        3, ge=1,
        description=(
            "Minimum number of sectors per cluster. "
            "Prevents degenerate singletons or tiny fragments."
        ),
    )

    # ── MST weight quality control ────────────────────────────────────
    mst_min_overlap: int = Field(
        10, ge=1,
        description=(
            "Minimum number of biweeks where both sectors have non-NaN eggs "
            "before computing Pearson correlation. Below this the edge weight "
            "falls back to 2.0 (maximum dissimilarity)."
        ),
    )

    # ── Epidemic years ────────────────────────────────────────────────
    epidemic_years: list[str] = Field(
        default_factory=lambda: [
            "2012_13",
            "2015_16",
            "2018_19",
            "2023_24",
        ],
        description=(
            "Year labels (matching biweek prefix format, e.g. '2012_13') "
            "used to filter biweeks for the pruning objective. "
            "Only epidemic years are used because off-season dengue signal "
            "is too flat and noisy to yield a meaningful correlation."
        ),
    )

    # ── Reproducibility ───────────────────────────────────────────────
    seed: int = Field(
        42,
        description="Global random seed for numpy and Python random.",
    )

    # ── Run identity ──────────────────────────────────────────────────
    run_label: str = Field(
        "default",
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9_\-]*$",
        description=(
            "Label for this run's output subdirectory. "
            "Results are written to results/skater/<run_label>/. "
            "Change before each dvc repro to preserve multiple runs "
            "on disk for concordance analysis. "
            "Allowed characters: letters, digits, underscores, hyphens."
        ),
    )


def load_config(
    params_path: Path = Path("params.yaml"),
) -> SkaterConfig:
    """Load and validate SKATER parameters from a DVC params file.

    Args:
        params_path: Path to the YAML params file (default: params.yaml).

    Returns:
        A fully-validated SkaterConfig instance.
    """
    with open(params_path) as fh:
        raw = yaml.safe_load(fh)
    # Fall back to all defaults if the skater: key is missing
    return SkaterConfig(**(raw.get("skater") or {}))
