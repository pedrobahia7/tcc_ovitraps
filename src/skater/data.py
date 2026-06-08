"""Load and pre-process ovitraps/dengue data for SKATER.

Produces aligned numpy matrices that all downstream modules expect:

  eggs_all    — IDW egg counts for every biweek (used for MST weights)
  eggs_epic   — IDW egg counts restricted to epidemic biweeks (pruning)
  dengue_epic — EB dengue rate per 1000, epidemic biweeks only (pruning)
  pop_vector  — median sector population (for dengue rate weighting)
  year_ids    — epidemic-year label for each epidemic biweek column

All matrices share the same row order (sector_list) and column order
(biweek lists).  Sectors missing from either eggs or dengue data are
excluded so every downstream computation works on a consistent set.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SkaterConfig

logger = logging.getLogger(__name__)

# ── Input file paths ──────────────────────────────────────────────────
# IDW egg counts interpolated to each sector centroid from nearby ovitraps
_EGGS_PATH = Path(
    "data/dvc/add_population_info/sector_centroids_with_idw.csv"
)
# EB (Empirical Bayes) smoothed dengue rates + population per sector
_DENGUE_PATH = Path(
    "data/dvc/add_population_info/dengue_per_capita.csv"
)
# 2022 IBGE census sector polygons with population attributes
_GEO_PATH = Path(
    "data/dvc/process_population_data/"
    "bh_sectors_2022_with_populations.geojson"
)


@dataclass
class SkaterData:
    """All pre-processed inputs needed by the SKATER pipeline.

    Every array is aligned: row i corresponds to sector_list[i], and
    columns correspond to the matching biweek list.

    Attributes:
        sector_list:      Sorted list of sector IDs present in both
                          eggs and dengue datasets.
        eggs_all:         (n_sectors, n_all_biweeks) — IDW egg values
                          for every available biweek.  Used for MST cost.
        all_biweeks:      Column labels for eggs_all.
        eggs_epic:        (n_sectors, n_epic_biweeks) — IDW egg values
                          for epidemic biweeks only.  Used for pruning.
        dengue_epic:      (n_sectors, n_epic_biweeks) — EB dengue rate
                          per 1000 people.  Used as pruning target.
        pop_vector:       (n_sectors,) — median population per sector
                          across all biweeks.  Used to weight dengue rates
                          when aggregating clusters.
        epidemic_biweeks: Column labels for eggs_epic / dengue_epic.
        year_ids:         (n_epic_biweeks,) — epidemic-year label for
                          each epidemic biweek, e.g. '2012_13'.
        geojson:          Raw GeoJSON FeatureCollection used to build the
                          adjacency graph and render the map dashboard.
    """

    sector_list: list[str]
    eggs_all: np.ndarray        # (n_sectors, n_all_biweeks)  — for MST cost
    all_biweeks: list[str]
    eggs_epic: np.ndarray       # (n_sectors, n_epic_biweeks) — for pruning
    dengue_epic: np.ndarray     # (n_sectors, n_epic_biweeks) eb_rate_per_1000
    pop_vector: np.ndarray      # (n_sectors,)
    epidemic_biweeks: list[str]
    year_ids: np.ndarray        # (n_epic_biweeks,) str array
    geojson: dict


def load_data(cfg: SkaterConfig) -> SkaterData:
    """Load all inputs and return aligned numpy matrices.

    Processing steps:
      1. Load eggs CSV → pivot (sector × biweek) → eggs_all.
      2. Identify epidemic biweeks by matching year prefix against
         cfg.epidemic_years.
      3. Slice eggs_all columns to get eggs_epic.
      4. Load dengue CSV → pivot → dengue_epic (epidemic biweeks only).
      5. Derive pop_vector as the median population across biweeks.

    Args:
        cfg: Pipeline configuration (epidemic years, etc.).

    Returns:
        SkaterData with all matrices aligned on sector_list.
    """
    logger.info("Loading IDW eggs…")
    eggs_raw = pd.read_csv(
        _EGGS_PATH,
        usecols=["CD_SETOR", "biweek", "idw_egg_value"],
        dtype={"CD_SETOR": str},
    ).rename(columns={"CD_SETOR": "sector_id"})

    logger.info("Loading dengue per capita…")
    dengue_raw = pd.read_csv(
        _DENGUE_PATH,
        usecols=["sector_id", "biweek", "population", "eb_rate_per_1000"],
        dtype={"sector_id": str},
    )

    logger.info("Loading GeoJSON…")
    with open(_GEO_PATH) as fh:
        geojson = json.load(fh)

    # ── Keep only sectors with both eggs and dengue data ──────────────
    egg_sectors = set(eggs_raw["sector_id"].unique())
    dng_sectors = set(dengue_raw["sector_id"].unique())
    common = sorted(egg_sectors & dng_sectors)
    logger.info("%d sectors with both eggs and dengue data", len(common))

    # ── Build full biweek egg matrix ──────────────────────────────────
    eggs_pivot = (
        eggs_raw[eggs_raw["sector_id"].isin(common)]
        .pivot(index="sector_id", columns="biweek", values="idw_egg_value")
        .reindex(index=common)
    )
    all_biweeks: list[str] = list(eggs_pivot.columns)
    eggs_all: np.ndarray = eggs_pivot.to_numpy(dtype=float)

    # ── Identify epidemic biweeks ─────────────────────────────────────
    # Biweek format: "YYYY_MMWNN" — split on last 'W' to get year prefix
    bw_series = pd.Series(all_biweeks)
    epic_mask: np.ndarray = (
        bw_series.str.rsplit("W", n=1).str[0].isin(cfg.epidemic_years).values
    )
    epidemic_biweeks = [bw for bw, m in zip(all_biweeks, epic_mask) if m]

    # Label each epidemic biweek with its year prefix (e.g. '2012_13')
    year_ids = np.array(
        [bw.rsplit("W", 1)[0] for bw in epidemic_biweeks]
    )

    # ── Slice egg matrix to epidemic biweeks ──────────────────────────
    eggs_epic: np.ndarray = eggs_all[:, epic_mask]

    # ── Build dengue matrix (epidemic biweeks only) ───────────────────
    dengue_pivot = (
        dengue_raw[dengue_raw["sector_id"].isin(common)]
        .pivot(
            index="sector_id",
            columns="biweek",
            values="eb_rate_per_1000",
        )
        # Align rows and columns to match eggs matrices exactly
        .reindex(index=common, columns=epidemic_biweeks)
    )
    dengue_epic: np.ndarray = dengue_pivot.to_numpy(dtype=float)

    # ── Derive population vector ──────────────────────────────────────
    # Median across biweeks — more stable than a single snapshot for
    # sectors with interpolated or missing population in some years.
    pop_series = (
        dengue_raw[dengue_raw["sector_id"].isin(common)]
        .groupby("sector_id")["population"]
        .median()
        .reindex(common)
        .fillna(1.0)   # 0-population sectors get weight 1 to avoid /0
    )
    pop_vector: np.ndarray = pop_series.to_numpy(dtype=float)

    logger.info(
        "eggs_all: %s  eggs_epic: %s  dengue_epic: %s  n_epic_bw: %d",
        eggs_all.shape,
        eggs_epic.shape,
        dengue_epic.shape,
        len(epidemic_biweeks),
    )
    return SkaterData(
        sector_list=common,
        eggs_all=eggs_all,
        all_biweeks=all_biweeks,
        eggs_epic=eggs_epic,
        dengue_epic=dengue_epic,
        pop_vector=pop_vector,
        epidemic_biweeks=epidemic_biweeks,
        year_ids=year_ids,
        geojson=geojson,
    )
