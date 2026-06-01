"""
Stage: add_population_info

Merges all population and spatial information into the ovitraps and dengue
datasets in three sequential sub-steps:

  1. Sector assignment — point-in-polygon spatial join to assign a 2022 IBGE
     census sector ID (population_sector) to each dengue case and ovitrap.

  2. Dengue per capita — aggregates dengue cases by sector and biweek, joins
     with the interpolated population table, and computes crude and Empirical
     Bayes smoothed incidence rates per 1,000 inhabitants (Marshall, 1991).

  3. Centroids + IDW — calculates the geometric centroid of every 2022 BH
     sector and estimates the ovitrap egg count for each (sector, biweek) via
     Inverse Distance Weighting (k=6, power=2).

Inputs (produced by process_data and process_population_data):
  - Intermediate dengue/ovitraps CSVs
  - 2022 BH sectors GeoJSON
  - Interpolated weekly population CSV

Outputs:
  - dengue_data.csv       — with population_sector column
  - ovitraps_data.csv     — with population_sector column
  - dengue_per_capita.csv — (sector, biweek) incidence rates
  - centroids.csv         — one row per sector with centroid lat/lon
  - centroids_idw.csv     — one row per (sector, biweek) with idw_egg_value
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from esda.smoothing import Empirical_Bayes
from shapely.geometry import Point

sys.path.append("utils")
import generic
import project_utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PER_CAPITA_MULTIPLIER: int = 1_000
DEFAULT_N_NEIGHBORS: int = 6
DEFAULT_IDW_POWER: float = 2.0


# ================================================================
# %% Step 1: Sector assignment
# ================================================================


def _load_sectors(geojson_path: Path) -> gpd.GeoDataFrame:
    """
    Load a census sectors GeoJSON file and guarantee a sector_id column.

    If the file already contains a column named ``sector_id`` it is used as-is.
    Otherwise the first column whose name contains any of the substrings
    ``sector``, ``setor``, ``cd_``, or ``geocod`` (case-insensitive) is
    aliased to ``sector_id``.

    Parameters
    ----------
    - geojson_path (Path): Path to the GeoJSON file containing sector polygons.

    Returns
    -------
    - gdf (gpd.GeoDataFrame): GeoDataFrame with an added or verified
      ``sector_id`` column.
    """
    gdf = gpd.read_file(geojson_path)
    if "sector_id" not in gdf.columns:
        candidates = [
            c for c in gdf.columns
            if any(x in c.lower() for x in ("sector", "setor", "cd_", "geocod"))
        ]
        if not candidates:
            raise ValueError(
                f"No sector ID column found in {geojson_path}"
            )
        gdf["sector_id"] = gdf[candidates[0]]
        logger.info("Using '%s' as sector_id", candidates[0])
    logger.info("Loaded %d census sectors from %s", len(gdf), geojson_path)
    return gdf


def _assign_sector_ids(
    df: pd.DataFrame,
    sectors_gdf: gpd.GeoDataFrame,
    name: str,
) -> pd.DataFrame:
    """
    Assign a census sector ID to every row in df via a point-in-polygon join.

    Rows with missing or zero coordinates are skipped and receive ``None``.
    The result is written to a new column ``population_sector`` on df.

    Parameters
    ----------
    - df (pd.DataFrame): DataFrame containing ``latitude`` and ``longitude``
      columns.
    - sectors_gdf (gpd.GeoDataFrame): GeoDataFrame of census sector polygons
      with a ``sector_id`` column.
    - name (str): Label used in log messages to identify the dataset
      (e.g. ``"ovitraps"`` or ``"dengue"``).

    Returns
    -------
    - df (pd.DataFrame): Input DataFrame with an added ``population_sector``
      column; unmatched rows contain ``None``.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("%s: missing lat/lon — sector IDs skipped", name)
        df["population_sector"] = None
        return df

    valid = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & (df["latitude"] != 0)
        & (df["longitude"] != 0)
    )
    logger.info(
        "%s: %d / %d records have valid coordinates (%.1f%%)",
        name, int(valid.sum()), len(df), valid.mean() * 100,
    )
    if not valid.any():
        df["population_sector"] = None
        return df

    valid_df = df[valid].copy()
    points_gdf = gpd.GeoDataFrame(
        valid_df,
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(valid_df["longitude"], valid_df["latitude"])
        ],
        crs="EPSG:4326",
    )
    if sectors_gdf.crs != points_gdf.crs:
        sectors_gdf = sectors_gdf.to_crs(points_gdf.crs)

    joined = gpd.sjoin(
        points_gdf,
        sectors_gdf[["sector_id", "geometry"]],
        how="left",
        predicate="within",
    )
    matched = int(joined["sector_id"].notna().sum())
    logger.info(
        "%s: %d / %d records matched a sector", name, matched, int(valid.sum())
    )

    df["population_sector"] = None
    df.loc[valid, "population_sector"] = joined["sector_id"].values
    return df


# ================================================================
# %% Step 2: Dengue per capita
# ================================================================


def _aggregate_case_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate confirmed dengue cases by census sector and biweek.

    Rows with a missing ``population_sector`` or ``biweek`` are dropped before
    aggregation. The ``population_sector`` column is cast to string with
    trailing ``.0`` removed to normalise float-encoded sector IDs.

    Parameters
    ----------
    - df (pd.DataFrame): Dengue DataFrame containing at least
      ``population_sector`` and ``biweek`` columns.

    Returns
    -------
    - case_counts (pd.DataFrame): DataFrame with columns ``sector_id``,
      ``biweek``, and ``case_count`` (one row per sector-biweek combination).
    """
    required = {"population_sector", "biweek"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dengue data missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()
    df["population_sector"] = (
        df["population_sector"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    return (
        df.groupby(["population_sector", "biweek"])
        .size()
        .reset_index(name="case_count")
        .rename(columns={"population_sector": "sector_id"})
    )


def _load_population_biweekly(population_path: Path) -> pd.DataFrame:
    """
    Load and reshape the interpolated population CSV from wide to long format.

    The CSV is expected to have a ``sector_id`` column plus one column per
    epidemic date (e.g. ``2019_20W03``). Each epidemic-date column is melted
    to a long row, converted to its corresponding biweek label, and averaged
    across the weeks that fall in that biweek. Population values are clipped
    at zero and rounded to the nearest integer.

    Parameters
    ----------
    - population_path (Path): Path to the wide-format interpolated population
      CSV file.

    Returns
    -------
    - biweekly (pd.DataFrame): Long-format DataFrame with columns
      ``sector_id``, ``biweek``, and ``population``
      (one row per sector-biweek combination).
    """
    wide = pd.read_csv(population_path)
    if "sector_id" not in wide.columns:
        raise ValueError(
            "Interpolated population must contain 'sector_id'"
        )

    id_cols = ["sector_id"] + [
        c for c in ("population_2010", "population_2022")
        if c in wide.columns
    ]
    week_cols = [c for c in wide.columns if c not in id_cols]
    if not week_cols:
        raise ValueError(
            "No epidemic-week columns in interpolated population data"
        )

    wide["sector_id"] = wide["sector_id"].astype(str)
    long = wide.melt(
        id_vars=["sector_id"],
        value_vars=week_cols,
        var_name="epidemic_date",
        value_name="population",
    )
    long["population"] = pd.to_numeric(long["population"], errors="coerce")
    long["biweek"] = project_utils.epidemic_date_to_biweek(
        long["epidemic_date"]
    )
    biweekly = (
        long.groupby(["sector_id", "biweek"], as_index=False)["population"]
        .mean()
        .round(0)
    )
    biweekly["population"] = biweekly["population"].clip(lower=0).astype(int)
    return biweekly[["sector_id", "biweek", "population"]]


def _compute_dengue_per_capita(
    population: pd.DataFrame,
    case_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join population with case counts and compute per-capita incidence rates.

    Merges the biweekly population table with the aggregated case counts on
    ``(sector_id, biweek)``. Sectors with no recorded cases receive
    ``case_count = 0``. Two rate columns are produced: a crude rate and an
    Empirical Bayes smoothed rate, both expressed per 1 000 inhabitants.

    Parameters
    ----------
    - population (pd.DataFrame): Biweekly population table with columns
      ``sector_id``, ``biweek``, and ``population``.
    - case_counts (pd.DataFrame): Aggregated dengue case counts with columns
      ``sector_id``, ``biweek``, and ``case_count``.

    Returns
    -------
    - df (pd.DataFrame): Merged DataFrame sorted by ``(sector_id, biweek)``
      with added columns ``case_count``, ``cases_per_1000``, and
      ``eb_rate_per_1000``.
    """
    df = population.merge(case_counts, on=["sector_id", "biweek"], how="left")
    df["case_count"] = df["case_count"].fillna(0).astype(int)
    df["cases_per_1000"] = np.where(
        df["population"] > 0,
        df["case_count"] / df["population"] * PER_CAPITA_MULTIPLIER,
        0,
    )
    df["eb_rate_per_1000"] = _empirical_bayes_rate(
        df["case_count"].to_numpy(dtype=np.float64),
        df["population"].to_numpy(dtype=np.float64),
        PER_CAPITA_MULTIPLIER,
    )
    return df.sort_values(["sector_id", "biweek"]).reset_index(drop=True)


def _empirical_bayes_rate(
    events: np.ndarray,
    population: np.ndarray,
    multiplier: float = 1.0,
) -> np.ndarray:
    """
    Compute the Empirical Bayes smoothed incidence rate (Marshall, 1991).

    Rows where ``population`` is zero are excluded from the EB estimation and
    kept at zero in the output to avoid division errors. The raw rate for
    included rows is scaled by ``multiplier`` (e.g. 1 000 for per-thousand
    rates).

    Parameters
    ----------
    - events (np.ndarray): Observed event counts (e.g. dengue cases) per area.
    - population (np.ndarray): At-risk population per area; must be the same
      length as ``events``.
    - multiplier (float): Scaling factor applied to the smoothed rate.
      Default is ``1.0`` (returns a raw rate).

    Returns
    -------
    - result (np.ndarray): EB-smoothed rates scaled by ``multiplier``;
      zero where population is zero.
    """
    events = np.asarray(events, dtype=np.float64)
    population = np.asarray(population, dtype=np.float64)
    valid = population > 0
    result = np.zeros_like(events, dtype=np.float64)
    if valid.any():
        eb = Empirical_Bayes(events[valid], population[valid])
        result[valid] = eb.r.ravel() * multiplier
    return result


# ================================================================
# %% Step 2b: City-wide dengue per capita
# ================================================================


def _compute_citywide_dengue_per_capita(
    dengue_df: pd.DataFrame,
    population_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute city-wide dengue incidence per 1 000 inhabitants per biweek.

    Counts ALL dengue rows with a valid biweek — no sector filter — so
    early years (2006-2012) with missing coordinates are included.

    Parameters
    ----------
    - dengue_df (pd.DataFrame): Dengue DataFrame with at least a ``biweek``
      column (output of process_data, before sector assignment drops rows).
    - population_df (pd.DataFrame): Sector-biweek population table with
      columns ``sector_id``, ``biweek``, ``population``.

    Returns
    -------
    - df (pd.DataFrame): DataFrame with columns ``biweek``, ``case_count``,
      ``city_population``, ``cases_per_1000``, sorted by ``biweek``.
    """
    if "biweek" not in dengue_df.columns:
        raise ValueError("Dengue data missing 'biweek' column")

    case_counts = (
        dengue_df.dropna(subset=["biweek"])
        .groupby("biweek")
        .size()
        .reset_index(name="case_count")
    )
    city_pop = (
        population_df.groupby("biweek", as_index=False)["population"]
        .sum()
        .rename(columns={"population": "city_population"})
    )
    df = case_counts.merge(city_pop, on="biweek", how="left")
    df["cases_per_1000"] = np.where(
        df["city_population"] > 0,
        df["case_count"] / df["city_population"] * PER_CAPITA_MULTIPLIER,
        0.0,
    )
    return df.sort_values("biweek").reset_index(drop=True)


# ================================================================
# %% Step 3: Centroids + IDW
# ================================================================


def _calculate_centroids(sectors_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute the geographic centroid of each census sector polygon.

    Sectors are reprojected to SIRGAS 2000 / UTM zone 23S (EPSG:31983) for
    accurate planar centroid calculation, then back-projected to WGS-84
    (EPSG:4326) for output. Only the columns listed in ``keep`` are retained;
    missing columns are silently skipped.

    Parameters
    ----------
    - sectors_gdf (gpd.GeoDataFrame): GeoDataFrame of 2022 BH census sector
      polygons in any CRS (re-projected internally).

    Returns
    -------
    - df (pd.DataFrame): DataFrame sorted by ``CD_SETOR`` containing sector
      attributes plus ``centroid_latitude`` and ``centroid_longitude`` columns.
    """
    projected = sectors_gdf.to_crs(epsg=31983)
    centroids_geo = projected.geometry.centroid.to_crs(sectors_gdf.crs)
    gdf = sectors_gdf.copy()
    gdf["centroid_longitude"] = centroids_geo.x
    gdf["centroid_latitude"] = centroids_geo.y

    keep = [
        "CD_SETOR", "SITUACAO", "NM_BAIRRO", "NM_SUBDIST",
        "AREA_KM2", "pop_2010", "pop_2022",
        "centroid_latitude", "centroid_longitude",
    ]
    df = gdf[[c for c in keep if c in gdf.columns]].copy()
    return df.sort_values("CD_SETOR").reset_index(drop=True)


def _calculate_eggs_per_centroid(
    centroids_df: pd.DataFrame,
    ovitraps_df: pd.DataFrame,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    power: float = DEFAULT_IDW_POWER,
) -> pd.DataFrame:
    """
    Estimate the egg count at each sector centroid per biweek using IDW.

    For each biweek the ``n_neighbors`` nearest ovitraps (by Euclidean distance
    in lat/lon space) are selected via a KNN query and their egg counts are
    combined with an Inverse Distance Weighting scheme (exponent ``power``).
    If any neighbour lies exactly on the centroid (distance == 0) its value is
    used directly to avoid division by zero.

    Parameters
    ----------
    - centroids_df (pd.DataFrame): DataFrame with columns ``CD_SETOR``,
      ``centroid_latitude``, and ``centroid_longitude`` (one row per sector).
    - ovitraps_df (pd.DataFrame): Ovitraps DataFrame with columns ``biweek``,
      ``narmad``, ``latitude``, ``longitude``, and ``novos``.
    - n_neighbors (int): Number of nearest traps to use for interpolation.
      Default is ``DEFAULT_N_NEIGHBORS`` (6).
    - power (float): Distance decay exponent for IDW. Default is
      ``DEFAULT_IDW_POWER`` (2.0).

    Returns
    -------
    - idw_df (pd.DataFrame): DataFrame with one row per (CD_SETOR, biweek)
      containing ``idw_egg_value``, centroid coordinates, ``n_traps_used``,
      ``min_distance_km``, and ``narmads_used``.
    """
    eggs = ovitraps_df.copy()
    centroid_coords = centroids_df[
        ["centroid_latitude", "centroid_longitude"]
    ].values
    sector_codes = centroids_df["CD_SETOR"].values
    biweeks = sorted(eggs["biweek"].unique())
    logger.info("IDW: %d centroids × %d biweeks", len(centroids_df), len(biweeks))

    results = []
    for i, biweek in enumerate(biweeks):
        if (i + 1) % 20 == 0:
            logger.info("  IDW biweek %d / %d", i + 1, len(biweeks))

        week_data = eggs[eggs["biweek"] == biweek]
        if week_data.empty:
            continue
        week_data = week_data.groupby(
            ["narmad", "latitude", "longitude"], as_index=False
        )["novos"].sum()

        trap_coords = week_data[["latitude", "longitude"]].to_numpy()
        trap_values = week_data["novos"].to_numpy()
        trap_ids = week_data["narmad"].to_numpy()

        k = min(n_neighbors, len(trap_coords))
        dist_knn, nearest_all = generic.nearest_neighbors(
            centroid_coords, trap_coords, k=k
        )

        for j in range(len(centroids_df)):
            idx_n = nearest_all[j]
            d = dist_knn[j]
            v = trap_values[idx_n]
            zero = np.nonzero(d == 0)[0]
            idw_val = (
                float(v[zero[0]])
                if zero.size
                else float(np.sum(v / d**power) / np.sum(1.0 / d**power))
            )
            results.append({
                "CD_SETOR":           sector_codes[j],
                "biweek":             biweek,
                "idw_egg_value":      idw_val,
                "centroid_latitude":  centroid_coords[j, 0],
                "centroid_longitude": centroid_coords[j, 1],
                "n_traps_used":       len(idx_n),
                "min_distance_km":    float(d.min()) * 111.32,
                "narmads_used":       ",".join(str(n) for n in trap_ids[idx_n]),
            })

    idw_df = (
        pd.DataFrame(results)
        .sort_values(["CD_SETOR", "biweek"])
        .reset_index(drop=True)
    )
    return idw_df


# ================================================================
# %% Main
# ================================================================


def main() -> None:
    # Params and paths
    params = yaml.safe_load(open("params.yaml"))
    _dvc = params["all"]["paths"]["data"]["dvc"]

    _in = {
        "ovitraps":                Path(_dvc["process_data"]["ovitraps"]),
        "dengue":                  Path(_dvc["process_data"]["dengue"]),
        "sectors_geojson":         Path(_dvc["process_population_data"]["sectors_geojson"]),
        "population_interpolated": Path(_dvc["process_population_data"]["population_interpolated"]),
    }
    _out = {
        "ovitraps":                   Path(_dvc["add_population_info"]["ovitraps"]),
        "dengue":                     Path(_dvc["add_population_info"]["dengue"]),
        "dengue_per_capita":          Path(_dvc["add_population_info"]["dengue_per_capita"]),
        "dengue_citywide_per_capita": Path(_dvc["add_population_info"]["dengue_citywide_per_capita"]),
        "centroids":                  Path(_dvc["add_population_info"]["centroids"]),
        "centroids_idw":              Path(_dvc["add_population_info"]["centroids_idw"]),
    }
    _out["ovitraps"].parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load all
    # ----------------------------------------------------------------
    logger.info("Loading data...")
    sectors_gdf = _load_sectors(_in["sectors_geojson"])
    ovitraps = pd.read_csv(_in["ovitraps"])
    dengue = pd.read_csv(_in["dengue"])
    population = _load_population_biweekly(_in["population_interpolated"])
    logger.info("  %d sector-biweek population rows", len(population))

    # ----------------------------------------------------------------
    # Transform
    # ----------------------------------------------------------------
    logger.info("=== Step 1: Sector assignment ===")
    ovitraps = _assign_sector_ids(ovitraps, sectors_gdf, "ovitraps")
    dengue = _assign_sector_ids(dengue, sectors_gdf, "dengue")

    logger.info("=== Step 2: Dengue per capita ===")
    case_counts = _aggregate_case_counts(dengue)
    logger.info("  %d sector-biweek combinations with cases", len(case_counts))
    dengue_per_capita_df = _compute_dengue_per_capita(population, case_counts)

    logger.info("=== Step 2b: City-wide dengue per capita ===")
    dengue_citywide_df = _compute_citywide_dengue_per_capita(
        dengue, population
    )
    logger.info(
        "  %d biweeks, %d total cases",
        len(dengue_citywide_df),
        int(dengue_citywide_df["case_count"].sum()),
    )

    logger.info("=== Step 3: Centroids + IDW ===")
    sectors_gdf = _calculate_centroids(sectors_gdf)
    biweek_ovitraps_centroids_df = _calculate_eggs_per_centroid(sectors_gdf, ovitraps)

    # ----------------------------------------------------------------
    # Save all
    # ----------------------------------------------------------------
    logger.info("Saving outputs...")
    _out["dengue_per_capita"].parent.mkdir(parents=True, exist_ok=True)
    ovitraps.to_csv(_out["ovitraps"], index=False)
    dengue.to_csv(_out["dengue"], index=False)
    dengue_per_capita_df.to_csv(_out["dengue_per_capita"], index=False)
    dengue_citywide_df.to_csv(_out["dengue_citywide_per_capita"], index=False)
    sectors_gdf.to_csv(_out["centroids"], index=False)
    biweek_ovitraps_centroids_df.to_csv(_out["centroids_idw"], index=False)
    logger.info("Saved ovitraps               → %s", _out["ovitraps"])
    logger.info("Saved dengue                 → %s", _out["dengue"])
    logger.info(
        "Saved dengue_per_capita      → %s (%d rows)",
        _out["dengue_per_capita"], len(dengue_per_capita_df),
    )
    logger.info(
        "Saved dengue_citywide        → %s (%d rows)",
        _out["dengue_citywide_per_capita"], len(dengue_citywide_df),
    )
    logger.info(
        "Saved centroids              → %s (%d sectors)",
        _out["centroids"], len(sectors_gdf),
    )
    logger.info(
        "Saved centroids_idw          → %s (%d rows)",
        _out["centroids_idw"], len(biweek_ovitraps_centroids_df),
    )

    logger.info("add_population_info completed successfully.")


if __name__ == "__main__":
    main()
