"""Calculate dengue cases per capita (per 1,000 population) by sector and biweek.

Aggregates dengue case counts per population sector and biweek, then
joins with the interpolated population data to compute the per-capita rate.
Also applies Empirical Bayes smoothing (Marshall, 1991) to correct for
variance instability in small-population sectors.

Biweek grouping follows the project convention:
    biweek_num = ((week_num + 1) // 2) * 2
    biweek = epi_year + 'W' + biweek_num (zero-padded)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from esda.smoothing import Empirical_Bayes


DEFAULT_PARAMS_PATH = Path("params.yaml")

PER_CAPITA_MULTIPLIER = 1_000


def epidemic_date_to_biweek(dates: pd.Series) -> pd.Series:
    """Convert epidemic_date values (e.g. '2018_19W51') to biweek labels.

    Uses the project-wide convention: biweek_num = ((week_num + 1) // 2) * 2.
    """
    dates = dates.astype(str)
    epi_year = dates.str.split("W").str[0]
    week_num = dates.str.split("W").str[1].astype(int)
    biweek_num = ((week_num + 1) // 2) * 2
    return epi_year + "W" + biweek_num.astype(str).str.zfill(2)


def load_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
    """Load the project parameters from params.yaml."""
    with open(params_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_paths(params: dict) -> tuple[Path, Path, Path]:
    """Resolve dengue, interpolated population, and output paths from params."""
    data = params["all"]["paths"]["data"]
    dengue_path = Path(data["processed"]["dengue"])
    population_path = Path(data["processed"]["population_interpolated"])
    output_path = Path(data["processed"]["dengue_per_capita"])
    return dengue_path, population_path, output_path


def load_dengue_cases(dengue_path: str | Path) -> pd.DataFrame:
    """Load dengue data and aggregate case counts by sector and biweek.

    Returns a DataFrame with columns: sector_id, biweek, case_count.
    """
    df = pd.read_csv(dengue_path, low_memory=False)

    required = {"population_sector", "epidemic_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dengue data is missing columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["population_sector", "epidemic_date"]).copy()
    df["population_sector"] = (
        df["population_sector"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    df["biweek"] = epidemic_date_to_biweek(df["epidemic_date"])

    case_counts = (
        df.groupby(["population_sector", "biweek"])
        .size()
        .reset_index(name="case_count")
        .rename(columns={"population_sector": "sector_id"})
    )
    return case_counts


def load_interpolated_population(
    population_path: str | Path,
) -> pd.DataFrame:
    """Load the wide-format interpolated population table and aggregate to biweeks.

    Returns a long-format DataFrame with columns:
    sector_id, biweek, population  (mean population per biweek).
    """
    wide = pd.read_csv(population_path)

    if "sector_id" not in wide.columns:
        raise ValueError(
            "Interpolated population data must contain 'sector_id'"
        )

    id_cols = ["sector_id"]
    extra_id = [
        c
        for c in ("population_2010", "population_2022")
        if c in wide.columns
    ]
    id_cols += extra_id

    week_cols = [c for c in wide.columns if c not in id_cols]
    if not week_cols:
        raise ValueError(
            "No epidemic-week columns found in interpolated population data"
        )

    wide["sector_id"] = wide["sector_id"].astype(str)

    long = wide.melt(
        id_vars=["sector_id"],
        value_vars=week_cols,
        var_name="epidemic_date",
        value_name="population",
    )
    long["population"] = pd.to_numeric(long["population"], errors="coerce")

    # Aggregate weekly population to biweekly (mean of the two weeks)
    long["biweek"] = epidemic_date_to_biweek(long["epidemic_date"])
    biweekly = (
        long.groupby(["sector_id", "biweek"], as_index=False)["population"]
        .mean()
        .round(0)
    )
    biweekly["population"] = biweekly["population"].clip(lower=0)
    biweekly = biweekly.astype({"population": int})
    return biweekly[["sector_id", "biweek", "population"]]


def compute_per_capita(
    case_counts: pd.DataFrame,
    population_biweekly: pd.DataFrame,
) -> pd.DataFrame:
    """Join biweekly cases with population and compute per-capita rate.

    Sectors with zero population get rate = 0.
    Sector-biweeks with no dengue cases are filled with 0 case_count.

    Returns DataFrame with columns:
    sector_id, biweek, case_count, population, cases_per_1000.
    """
    merged = population_biweekly.merge(
        case_counts,
        on=["sector_id", "biweek"],
        how="left",
    )
    merged["case_count"] = merged["case_count"].fillna(0).astype(int)

    merged["cases_per_1000"] = np.where(
        merged["population"] > 0,
        (merged["case_count"] / merged["population"])
        * PER_CAPITA_MULTIPLIER,
        0,
    )

    merged["eb_rate_per_1000"] = empirical_bayes_rate(
        merged["case_count"].to_numpy(dtype=np.float64),
        merged["population"].to_numpy(dtype=np.float64),
        PER_CAPITA_MULTIPLIER,
    )

    merged = merged.sort_values(["sector_id", "biweek"]).reset_index(
        drop=True
    )
    return merged


def empirical_bayes_rate(
    events: np.ndarray,
    population: np.ndarray,
    multiplier: float = 1.0,
) -> np.ndarray:
    """Compute Empirical Bayes smoothed rates (Marshall, 1991).

    Wrapper around PySAL's ``esda.smoothing.Empirical_Bayes``.
    Parameters are estimated once from *all* observations, then each
    observation's crude rate is shrunk toward the global mean proportionally
    to its population at risk.

    Parameters
    ----------
    events : np.ndarray
        Event counts (e.g. dengue cases) per observation.
    population : np.ndarray
        Population at risk per observation.
    multiplier : float
        Scaling factor applied to both crude and smoothed rates
        (e.g. 1000 for "per 1,000 population").

    Returns
    -------
    np.ndarray
        EB-smoothed rates, same length as *events*.
    """
    events = np.asarray(events, dtype=np.float64)
    population = np.asarray(population, dtype=np.float64)

    valid = population > 0
    result = np.zeros_like(events, dtype=np.float64)

    if not valid.any():
        return result

    e = events[valid]
    b = population[valid]

    eb = Empirical_Bayes(e, b)
    result[valid] = eb.r.ravel() * multiplier

    return result


def save_per_capita(df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist the per-capita table to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate dengue cases per capita by sector and epidemic week."
    )
    parser.add_argument("--params-path", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--dengue-path", default=None)
    parser.add_argument("--population-path", default=None)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the per-capita calculation pipeline."""
    args = parse_args()
    params = load_params(args.params_path)
    default_dengue, default_pop, default_out = resolve_paths(params)

    dengue_path = (
        Path(args.dengue_path) if args.dengue_path else default_dengue
    )
    population_path = (
        Path(args.population_path) if args.population_path else default_pop
    )
    output_path = (
        Path(args.output_path) if args.output_path else default_out
    )

    print(f"Loading dengue data from {dengue_path} ...")
    case_counts = load_dengue_cases(dengue_path)
    print(f"  {len(case_counts):,} sector-biweek combinations with cases")

    print(f"Loading interpolated population from {population_path} ...")
    population_biweekly = load_interpolated_population(population_path)
    print(f"  {len(population_biweekly):,} sector-biweek rows")

    print(
        "Computing biweekly per-capita rates (crude + Empirical Bayes) ..."
    )
    per_capita = compute_per_capita(case_counts, population_biweekly)
    print(f"  {len(per_capita):,} rows in output")
    print(
        f"  Non-zero case rows: {(per_capita['case_count'] > 0).sum():,}"
    )
    print(
        f"  EB rate range: "
        f"{per_capita['eb_rate_per_1000'].min():.4f} – "
        f"{per_capita['eb_rate_per_1000'].max():.4f}"
    )

    save_per_capita(per_capita, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
