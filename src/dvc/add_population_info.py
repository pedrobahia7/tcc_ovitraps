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

Inputs (produced by fast_processing and process_population_data):
  - Intermediate dengue/ovitraps CSVs  (data/processed/add_population_sectors/)
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_params(params_path: Path = Path("params.yaml")) -> dict:
    with open(params_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_paths(params: dict) -> dict[str, Path]:
    dvc = params["all"]["paths"]["data"]["dvc"]
    return {
        "ovitraps_input":          Path(dvc["fast_processing"]["ovitraps"]),
        "dengue_input":            Path(dvc["fast_processing"]["dengue"]),
        "sectors_geojson":         Path(dvc["process_population_data"]["sectors_geojson"]),
        "population_interpolated": Path(dvc["process_population_data"]["population_interpolated"]),
        "ovitraps_out":            Path(dvc["add_population_info"]["ovitraps"]),
        "dengue_out":              Path(dvc["add_population_info"]["dengue"]),
        "dengue_per_capita":       Path(dvc["add_population_info"]["dengue_per_capita"]),
        "centroids":               Path(dvc["add_population_info"]["centroids"]),
        "centroids_idw":           Path(dvc["add_population_info"]["centroids_idw"]),
    }


def _load_sectors(geojson_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(geojson_path)
    if "sector_id" not in gdf.columns:
        candidates = [
            c for c in gdf.columns
            if any(x in c.lower() for x in ("sector", "setor", "cd_", "geocod"))
        ]
        if not candidates:
            raise ValueError(f"No sector ID column found in {geojson_path}")
        gdf["sector_id"] = gdf[candidates[0]]
        logger.info("Using '%s' as sector_id", candidates[0])
    logger.info("Loaded %d census sectors from %s", len(gdf), geojson_path)
    return gdf


# ---------------------------------------------------------------------------
# Step 1: Sector assignment
# ---------------------------------------------------------------------------


def _assign_sector_ids(
    df: pd.DataFrame,
    sectors_gdf: gpd.GeoDataFrame,
    name: str,
) -> pd.DataFrame:
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
    logger.info("%s: %d / %d records matched a sector", name, matched, int(valid.sum()))

    df["population_sector"] = None
    df.loc[valid, "population_sector"] = joined["sector_id"].values
    return df


def run_sector_assignment(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sectors_gdf = _load_sectors(paths["sectors_geojson"])

    ovitraps = pd.read_csv(paths["ovitraps_input"])
    dengue = pd.read_csv(paths["dengue_input"])

    ovitraps = _assign_sector_ids(ovitraps, sectors_gdf, "ovitraps")
    dengue = _assign_sector_ids(dengue, sectors_gdf, "dengue")

    ovitraps.to_csv(paths["ovitraps_out"], index=False)
    dengue.to_csv(paths["dengue_out"], index=False)
    logger.info("Saved ovitraps → %s", paths["ovitraps_out"])
    logger.info("Saved dengue   → %s", paths["dengue_out"])
    return ovitraps, dengue


# ---------------------------------------------------------------------------
# Step 2: Dengue per capita
# ---------------------------------------------------------------------------


def _aggregate_case_counts(dengue: pd.DataFrame) -> pd.DataFrame:
    required = {"population_sector", "biweek"}
    missing = required - set(dengue.columns)
    if missing:
        raise ValueError(f"Dengue data missing columns: {sorted(missing)}")

    df = dengue.dropna(subset=list(required)).copy()
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
    wide = pd.read_csv(population_path)
    if "sector_id" not in wide.columns:
        raise ValueError("Interpolated population must contain 'sector_id'")

    id_cols = ["sector_id"] + [
        c for c in ("population_2010", "population_2022") if c in wide.columns
    ]
    week_cols = [c for c in wide.columns if c not in id_cols]
    if not week_cols:
        raise ValueError("No epidemic-week columns in interpolated population data")

    wide["sector_id"] = wide["sector_id"].astype(str)
    long = wide.melt(
        id_vars=["sector_id"],
        value_vars=week_cols,
        var_name="epidemic_date",
        value_name="population",
    )
    long["population"] = pd.to_numeric(long["population"], errors="coerce")
    long["biweek"] = project_utils.epidemic_date_to_biweek(long["epidemic_date"])
    biweekly = (
        long.groupby(["sector_id", "biweek"], as_index=False)["population"]
        .mean()
        .round(0)
    )
    biweekly["population"] = biweekly["population"].clip(lower=0).astype(int)
    return biweekly[["sector_id", "biweek", "population"]]


def _empirical_bayes_rate(
    events: np.ndarray,
    population: np.ndarray,
    multiplier: float = 1.0,
) -> np.ndarray:
    events = np.asarray(events, dtype=np.float64)
    population = np.asarray(population, dtype=np.float64)
    valid = population > 0
    result = np.zeros_like(events, dtype=np.float64)
    if valid.any():
        eb = Empirical_Bayes(events[valid], population[valid])
        result[valid] = eb.r.ravel() * multiplier
    return result


def run_dengue_per_capita(dengue: pd.DataFrame, paths: dict[str, Path]) -> None:
    logger.info("Aggregating case counts by sector and biweek...")
    case_counts = _aggregate_case_counts(dengue)
    logger.info("  %d sector-biweek combinations with cases", len(case_counts))

    population = _load_population_biweekly(paths["population_interpolated"])
    logger.info("  %d sector-biweek population rows", len(population))

    merged = population.merge(case_counts, on=["sector_id", "biweek"], how="left")
    merged["case_count"] = merged["case_count"].fillna(0).astype(int)
    merged["cases_per_1000"] = np.where(
        merged["population"] > 0,
        merged["case_count"] / merged["population"] * PER_CAPITA_MULTIPLIER,
        0,
    )
    merged["eb_rate_per_1000"] = _empirical_bayes_rate(
        merged["case_count"].to_numpy(dtype=np.float64),
        merged["population"].to_numpy(dtype=np.float64),
        PER_CAPITA_MULTIPLIER,
    )
    merged = merged.sort_values(["sector_id", "biweek"]).reset_index(drop=True)

    out = paths["dengue_per_capita"]
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    logger.info("Saved dengue per capita → %s (%d rows)", out, len(merged))


# ---------------------------------------------------------------------------
# Step 3: Centroids + IDW
# ---------------------------------------------------------------------------


def _calculate_centroids(
    sectors_gdf: gpd.GeoDataFrame, output_path: Path
) -> pd.DataFrame:
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
    df = df.sort_values("CD_SETOR").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved centroids → %s (%d sectors)", output_path, len(df))
    return df


def _calculate_idw(
    centroids_df: pd.DataFrame,
    ovitraps: pd.DataFrame,
    output_path: Path,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    power: float = DEFAULT_IDW_POWER,
) -> pd.DataFrame:
    eggs = ovitraps.copy()
    centroid_coords = centroids_df[["centroid_latitude", "centroid_longitude"]].values
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
            idx_n = nearest_all[j]   # shape (k,)
            d = dist_knn[j]          # shape (k,)
            v = trap_values[idx_n]
            zero = np.where(d == 0)[0]
            idw_val = (
                float(v[zero[0]])
                if zero.size
                else float(np.sum(v / d**power) / np.sum(1.0 / d**power))
            )
            results.append({
                "CD_SETOR":          sector_codes[j],
                "biweek":            biweek,
                "idw_egg_value":     idw_val,
                "centroid_latitude": centroid_coords[j, 0],
                "centroid_longitude":centroid_coords[j, 1],
                "n_traps_used":      len(idx_n),
                "min_distance_km":   float(d.min()) * 111.32,
                "narmads_used":      ",".join(str(n) for n in trap_ids[idx_n]),
            })

    idw_df = (
        pd.DataFrame(results)
        .sort_values(["CD_SETOR", "biweek"])
        .reset_index(drop=True)
    )
    idw_df.to_csv(output_path, index=False)
    logger.info("Saved centroids IDW → %s (%d rows)", output_path, len(idw_df))
    return idw_df


def run_centroids_and_idw(ovitraps: pd.DataFrame, paths: dict[str, Path]) -> None:
    sectors_gdf = _load_sectors(paths["sectors_geojson"])
    centroids_df = _calculate_centroids(sectors_gdf, paths["centroids"])
    _calculate_idw(centroids_df, ovitraps, paths["centroids_idw"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    params = _load_params()
    paths = _resolve_paths(params)

    logger.info("=== Step 1: Sector assignment ===")
    ovitraps, dengue = run_sector_assignment(paths)

    logger.info("=== Step 2: Dengue per capita ===")
    run_dengue_per_capita(dengue, paths)

    logger.info("=== Step 3: Centroids + IDW ===")
    run_centroids_and_idw(ovitraps, paths)

    logger.info("add_population_info completed successfully.")


if __name__ == "__main__":
    main()
