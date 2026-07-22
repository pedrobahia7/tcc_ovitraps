"""Sector-level data loading and SKATER-identical spatial aggregation.

A spatial *unit* is any set of census sectors (the whole city, a
district, a single sector, or a SKATER region).  Every unit's egg and
dengue time series is built with the exact same rule SKATER uses inside
src/skater/prune.py::_aggregate:

  eggs   — simple nanmean of member sectors' IDW egg values per biweek
           (each ovitrap-derived sector weighted equally).
  dengue — population-weighted mean of member sectors' EB dengue rate
           per 1000 (weights = median sector population), so small
           sectors do not dominate the rate signal.

Inputs:
  data/dvc/add_population_info/dengue_per_capita.csv      — eb_rate, pop
  data/dvc/add_population_info/sector_centroids_with_idw.csv — IDW eggs
  data/dvc/process_population_data/bh_sectors_2022_with_populations.geojson
Outputs:
  SectorData (aligned matrices) + unit-membership dictionaries.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Input paths (shared with the SKATER pipeline) ─────────────────────
_DENGUE_PATH = Path("data/dvc/add_population_info/dengue_per_capita.csv")
_EGGS_PATH = Path(
    "data/dvc/add_population_info/sector_centroids_with_idw.csv"
)
_GEO_PATH = Path(
    "data/dvc/process_population_data/"
    "bh_sectors_2022_with_populations.geojson"
)


@dataclass
class SectorData:
    """Aligned per-sector matrices spanning every available biweek.

    Attributes:
        sectors:  Sorted sector IDs present in both eggs and dengue data.
        biweeks:  Sorted (chronological) biweek labels — matrix columns.
        eb:       (n_sectors, n_biweeks) EB dengue rate per 1000.
        egg:      (n_sectors, n_biweeks) IDW egg values (may contain NaN).
        pop:      (n_sectors,) median population per sector.
        idx:      Mapping sector_id → row index into eb / egg / pop.
    """

    sectors: list[str]
    biweeks: list[str]
    eb: np.ndarray
    egg: np.ndarray
    pop: np.ndarray
    idx: dict[str, int]


def _biweek_year(biweek: str) -> str:
    """Return the epidemic-year prefix of a biweek label ('2015_16W04')."""
    return biweek.rsplit("W", 1)[0]


def load_sector_data() -> SectorData:
    """Load and align per-sector egg, dengue and population matrices.

    Steps:
      1. Load EB dengue rate + population and IDW eggs.
      2. Keep sectors present in both sources.
      3. Pivot each to (sector × biweek), reindexed to the sorted union
         of biweeks so columns are chronological and shared.
      4. Derive population as the per-sector median across biweeks.

    Returns:
        A SectorData with all arrays aligned on `sectors` × `biweeks`.
    """
    logger.info("Loading EB dengue per capita…")
    eb_raw = pd.read_csv(
        _DENGUE_PATH,
        usecols=["sector_id", "biweek", "population", "eb_rate_per_1000"],
        dtype={"sector_id": str},
    )
    logger.info("Loading IDW eggs…")
    egg_raw = pd.read_csv(
        _EGGS_PATH,
        usecols=["CD_SETOR", "biweek", "idw_egg_value"],
        dtype={"CD_SETOR": str},
    ).rename(columns={"CD_SETOR": "sector_id"})

    # ── Keep sectors present in both sources ──────────────────────────
    common = sorted(set(eb_raw["sector_id"]) & set(egg_raw["sector_id"]))
    logger.info("%d sectors with both eggs and dengue data", len(common))

    # ── Pivot to (sector × biweek) on the shared biweek axis ──────────
    eb_pivot = eb_raw.pivot_table(
        index="sector_id", columns="biweek", values="eb_rate_per_1000"
    ).reindex(common)
    biweeks = sorted(eb_pivot.columns)
    eb_pivot = eb_pivot.reindex(columns=biweeks)

    egg_pivot = (
        egg_raw.pivot_table(
            index="sector_id", columns="biweek", values="idw_egg_value"
        )
        .reindex(index=common, columns=biweeks)
    )

    # ── Population vector: median across biweeks (stable per sector) ───
    pop = (
        eb_raw.groupby("sector_id")["population"]
        .median()
        .reindex(common)
        .fillna(1.0)  # zero/absent population → weight 1 to avoid /0
        .to_numpy(dtype=float)
    )

    idx = {s: i for i, s in enumerate(common)}
    logger.info(
        "SectorData: eb %s  egg %s  n_biweeks %d",
        eb_pivot.shape, egg_pivot.shape, len(biweeks),
    )
    return SectorData(
        sectors=common,
        biweeks=biweeks,
        eb=eb_pivot.to_numpy(dtype=float),
        egg=egg_pivot.to_numpy(dtype=float),
        pop=pop,
        idx=idx,
    )


def aggregate_unit(
    members: list[str], data: SectorData
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate member sectors into unit egg and dengue series.

    Replicates src/skater/prune.py::_aggregate exactly:
      egg_series    = nanmean over members (equal weight per sector).
      dengue_series = population-weighted mean of EB rate; falls back to
                      a simple nanmean only when total population is 0.

    Args:
        members: Sector IDs composing the unit.
        data:    Loaded SectorData.

    Returns:
        (egg_series, dengue_series), each shape (n_biweeks,).
    """
    rows = np.array([data.idx[s] for s in members], dtype=int)
    egg_sub = data.egg[rows, :]
    eb_sub = data.eb[rows, :]
    pop_sub = data.pop[rows]

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # All-NaN egg columns (sector with no IDW that biweek) → NaN, and
        # the affected feature rows are dropped later; the warning is noise.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        egg_series = np.nanmean(egg_sub, axis=0)

    pop_total = float(pop_sub.sum())
    if pop_total > 0:
        dengue_series = (eb_sub * pop_sub[:, None]).sum(axis=0) / pop_total
    else:
        with np.errstate(all="ignore"):
            dengue_series = np.nanmean(eb_sub, axis=0)

    return egg_series, dengue_series


# ── Unit-membership builders ──────────────────────────────────────────

def city_units(data: SectorData) -> dict[str, list[str]]:
    """One unit containing every sector."""
    return {"BH": list(data.sectors)}


def sector_units(data: SectorData) -> dict[str, list[str]]:
    """One unit per individual sector (finest scale)."""
    return {s: [s] for s in data.sectors}


def district_units(data: SectorData) -> dict[str, list[str]]:
    """One unit per BH Regional (9 units) via geojson NM_SUBDIST.

    Only sectors present in SectorData are included; districts keep the
    NM_SUBDIST label as the unit key.

    Returns:
        Mapping district name → member sector IDs.
    """
    with open(_GEO_PATH) as fh:
        geojson = json.load(fh)

    sector_set = set(data.sectors)
    units: dict[str, list[str]] = {}
    for feat in geojson["features"]:
        props = feat["properties"]
        sid = str(props["CD_SETOR"])
        if sid not in sector_set:
            continue
        district = props.get("NM_SUBDIST")
        if district is None:
            continue
        units.setdefault(district, []).append(sid)

    logger.info(
        "district_units: %d districts, sizes %s",
        len(units),
        {k: len(v) for k, v in sorted(units.items())},
    )
    return units


def skater_region_units(
    assignments: dict[str, int], c_value: int
) -> dict[str, list[str]]:
    """Group sectors by SKATER cluster id at one C value.

    Args:
        assignments: sector_id → cluster index (from a Snapshot).
        c_value:     Number of clusters this snapshot represents (for keys).

    Returns:
        Mapping unit key ('C{C}__c{cluster_id}') → member sector IDs.
    """
    units: dict[str, list[str]] = {}
    for sid, cid in assignments.items():
        units.setdefault(f"C{c_value}__c{cid}", []).append(sid)
    return units
