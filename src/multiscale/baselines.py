"""Entry point: multiscale baseline models (city, district, sector).

Trains the frozen MLP under leave-one-epidemic-year-out CV at three
fixed spatial scales that do NOT depend on SKATER:

  city     — 1 unit  (whole city).
  district — 9 units  (BH Regionais, geojson NM_SUBDIST).
  sector   — 1 unit per census sector (finest scale).

Run:  python -m src.multiscale.baselines

Outputs (results/multiscale/baselines/):
  metrics_city.csv | metrics_district.csv | metrics_sector.csv
      — per-(unit, fold) test metrics + naive baseline.
  summary_baselines.csv
      — population-weighted mean test RMSE per scale.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import (
    city_units,
    district_units,
    load_sector_data,
    sector_units,
)
from .config import load_multiscale_config, load_skater_cv_config
from .cv import pop_weighted_summary, run_scale

logger = logging.getLogger(__name__)
_OUT_DIR = Path("results/multiscale/baselines")


def main() -> None:
    """Run the three baseline scales end-to-end and persist results."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    cfg = load_multiscale_config()
    cv_cfg = load_skater_cv_config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    logger.info("MultiscaleConfig: %s", cfg.model_dump())
    logger.info("fold_years: %s", cv_cfg.fold_years)

    data = load_sector_data()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run each baseline scale ───────────────────────────────────────
    scale_builders = {
        "city": city_units,
        "district": district_units,
        "sector": sector_units,
    }
    all_records: list[pd.DataFrame] = []
    for scale, builder in scale_builders.items():
        units = builder(data)
        records = run_scale(scale, units, data, cfg, cv_cfg.fold_years)
        records.to_csv(_OUT_DIR / f"metrics_{scale}.csv", index=False)
        logger.info(
            "Saved metrics_%s.csv (%d rows)", scale, len(records)
        )
        all_records.append(records)

    # ── Population-weighted per-scale summary ─────────────────────────
    summary = pop_weighted_summary(pd.concat(all_records, ignore_index=True))
    summary.to_csv(_OUT_DIR / "summary_baselines.csv", index=False)
    logger.info("Saved summary_baselines.csv\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
