"""SKATER pipeline configuration (Pydantic v2).

All tuneable parameters live in params.yaml under the `skater:` key.
This module loads them into a validated SkaterConfig object that every
other module receives — no magic strings scattered through the codebase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SkaterConfig(BaseModel):
    """All tunable parameters for the SKATER regionalization pipeline.

    Loaded from params.yaml[skater:] via load_config().
    Changing a value here (or in params.yaml) is the only thing needed
    to swap strategies or adjust guard thresholds.
    """

    # ── Strategy selectors ────────────────────────────────────────────
    mst_cost: Literal["egg_corr_dist"] = Field(
        "egg_corr_dist",
        description=(
            "Which dissimilarity metric to use when building the MST. "
            "'egg_corr_dist' = 1 − Pearson(eggs_i, eggs_j)."
        ),
    )
    prune_obj: Literal["eggs_dengue_corr"] = Field(
        "eggs_dengue_corr",
        description=(
            "Which objective to maximise during greedy pruning. "
            "'eggs_dengue_corr' = best signed Pearson(lagged_eggs, dengue_rate)."
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
    C_max: int = Field(
        30, ge=2,
        description="Maximum number of clusters to produce.",
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
