"""Configuration for the multiscale MLP comparison (Pydantic v2).

Two independent config blocks in params.yaml so the DVC stages cache
correctly:

  skater_cv: — partition-defining (fold_years + per-fold SKATER stop
               overrides).  Changing these re-runs the CV partitions.
  multiscale: — model-defining (MLP arch + feature lags).  Changing
               these re-runs only the model stages, never SKATER.

Mirrors the pattern of src/skater/config.py.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SkaterCvConfig(BaseModel):
    """Cross-validation + per-fold SKATER stop settings (partition stage).

    These define which epidemic years are folded and how the per-fold
    SKATER runs stop.  They are deliberately separate from the MLP
    architecture so that tuning the model never invalidates the cached
    partitions.
    """

    fold_years: list[str] = Field(
        default_factory=lambda: ["2015_16", "2018_19", "2023_24"],
        description=(
            "Epidemic years used for leave-one-year-out CV.  Each is held "
            "out once as the test fold; SKATER and the MLP both train on "
            "the remaining years."
        ),
    )

    # ── Per-fold SKATER stop overrides ────────────────────────────────
    # Override params.yaml[skater] for the cross-validated runs so the
    # sole stopping rule is "run until no split improves the global
    # objective Q".  The main `skater` DVC stage is untouched.
    cv_stop_local_degradation: bool = Field(
        False,
        description=(
            "Per-fold SKATER: enable the local-degradation per-cut guard. "
            "False removes it so only the global 'no improving split' rule "
            "and geometry guards (S_min/N_min) stop the algorithm."
        ),
    )
    cv_global_degradation_threshold: float | None = Field(
        0.0,
        description=(
            "Per-fold SKATER: stop when the best available cut changes Q "
            "by less than this. 0.0 = 'run until no split improves Q'. "
            "None disables it (run to cv_c_max, allowing Q to decrease)."
        ),
    )
    cv_c_max: int | None = Field(
        None,
        description=(
            "Per-fold SKATER hard ceiling on clusters. None = no ceiling; "
            "the 'no improving split' rule / geometry guards decide the stop."
        ),
    )


class MultiscaleConfig(BaseModel):
    """MLP architecture + feature construction (model stages only).

    A single frozen architecture is reused for every spatial unit at
    every scale — no per-unit hyper-parameter search — so that only the
    spatial aggregation varies across scales, never model capacity.
    """

    # ── Frozen MLP architecture ───────────────────────────────────────
    hidden_layer_sizes: list[int] = Field(
        default_factory=lambda: [32, 16],
        description="MLP hidden layer sizes (frozen, no grid search).",
    )
    alpha: float = Field(
        0.01, gt=0.0,
        description="L2 regularisation strength for the MLP.",
    )
    max_iter: int = Field(
        5000, ge=1,
        description="Maximum solver iterations (with early stopping).",
    )

    # ── Feature construction ──────────────────────────────────────────
    eb_lags: list[int] = Field(
        default_factory=lambda: [1, 2, 3],
        description="Lags (biweeks) of the unit EB dengue rate used as features.",
    )
    egg_lags: list[int] = Field(
        default_factory=lambda: [3, 4],
        description=(
            "Lags (biweeks) of the unit IDW egg series used as features. "
            "Matches the biologically relevant incubation window."
        ),
    )

    # ── Reproducibility ───────────────────────────────────────────────
    seed: int = Field(
        42,
        description="Global random seed for numpy, random and the MLP.",
    )

    # ── Derived helpers ───────────────────────────────────────────────
    @property
    def hidden_tuple(self) -> tuple[int, ...]:
        """Hidden layer sizes as the tuple sklearn expects."""
        return tuple(self.hidden_layer_sizes)

    @property
    def feature_order(self) -> list[str]:
        """Ordered feature column names implied by the lag windows.

        Returns:
            EB-rate lag columns, then egg lag columns, then the two
            cyclical seasonality columns — the exact column order the
            trained MLP expects at predict time.
        """
        cols = [f"eb_rate_lag{k}" for k in self.eb_lags]
        cols += [f"egg_lag{k}" for k in self.egg_lags]
        cols += ["week_sin", "week_cos"]
        return cols


def _load_block(params_path: Path, key: str) -> dict:
    """Return the given top-level block from a YAML params file."""
    with open(params_path) as fh:
        raw = yaml.safe_load(fh)
    return raw.get(key) or {}


def load_multiscale_config(
    params_path: Path = Path("params.yaml"),
) -> MultiscaleConfig:
    """Load and validate the multiscale (model) config block."""
    return MultiscaleConfig(**_load_block(params_path, "multiscale"))


def load_skater_cv_config(
    params_path: Path = Path("params.yaml"),
) -> SkaterCvConfig:
    """Load and validate the skater_cv (partition/CV) config block."""
    return SkaterCvConfig(**_load_block(params_path, "skater_cv"))
