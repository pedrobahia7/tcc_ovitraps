"""Build weekly linear population interpolations aligned to epidemic weeks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


DEFAULT_PARAMS_PATH = Path("params.yaml")
REQUIRED_POPULATION_COLUMNS = {
    "sector_id",
    "population_2010",
    "population_2022",
}
ID_COLUMNS = ["sector_id", "population_2010", "population_2022"]


def load_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
    """Load the project parameters from params.yaml."""
    with open(params_path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def resolve_default_paths(
    params: dict,
) -> tuple[Path, Path, Path]:
    """Resolve the default population, dengue, and output paths."""
    data_paths = params["all"]["paths"]["data"]
    population_path = Path(
        data_paths["processed"]["census_equivalence"][
            "population_2010_to_2022"
        ]
    )
    dengue_path = Path(data_paths["processed"]["dengue"])
    output_path = Path(data_paths["processed"]["population_interpolated"])
    return population_path, dengue_path, output_path


def load_population_data(file_path: str | Path) -> pd.DataFrame:
    """Load and validate the sector population comparison table."""
    population_data = pd.read_csv(file_path)

    missing_columns = REQUIRED_POPULATION_COLUMNS.difference(
        population_data.columns
    )
    if missing_columns:
        raise ValueError(
            f"Population data is missing required columns: {sorted(missing_columns)}"
        )

    population_data = population_data.copy()
    population_data["sector_id"] = population_data["sector_id"].astype(str)
    population_data["population_2010"] = population_data[
        "population_2010"
    ].astype("int64")
    population_data["population_2022"] = population_data[
        "population_2022"
    ].astype("int64")

    return population_data.sort_values("sector_id").reset_index(drop=True)


def load_epidemic_weeks(file_path: str | Path) -> list[str]:
    """Load the ordered epidemic weeks from the processed dengue dataset."""
    dengue_data = pd.read_csv(file_path, low_memory=False)

    if "epidemic_date" not in dengue_data.columns:
        if {"anoepid", "semepid"}.issubset(dengue_data.columns):
            from utils import project_utils

            dengue_data["epidemic_date"] = project_utils.get_epidemic_date(
                dengue_data
            )
        else:
            raise ValueError(
                "Dengue data must contain epidemic_date or the anoepid/semepid columns"
            )

    epidemic_weeks = (
        pd.Index(
            dengue_data["epidemic_date"].dropna().astype(str).unique()
        )
        .sort_values()
        .tolist()
    )

    if not epidemic_weeks:
        raise ValueError(
            "No epidemic weeks were found in the dengue dataset"
        )

    return epidemic_weeks


def build_week_fractions(number_of_weeks: int) -> np.ndarray:
    """Create a linear fraction for each epidemic week position."""
    if number_of_weeks <= 0:
        raise ValueError("number_of_weeks must be greater than zero")
    if number_of_weeks == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, 1.0, num=number_of_weeks, dtype=np.float64)


def _round_to_int(values: np.ndarray) -> np.ndarray:
    """Round positive numeric values to the nearest integer."""
    return np.floor(values + 0.5).astype(np.int64)


def build_interpolated_population_table(
    population_data: pd.DataFrame,
    epidemic_weeks: Iterable[str],
) -> pd.DataFrame:
    """Interpolate the 2010 and 2022 population values across epidemic weeks.

    The interpolation is linear over the ordered epidemic-week sequence. The
    output keeps one row per sector and one column per epidemic week, all stored
    as integers.
    """
    epidemic_weeks = list(epidemic_weeks)
    if not epidemic_weeks:
        raise ValueError("epidemic_weeks must not be empty")

    missing_columns = REQUIRED_POPULATION_COLUMNS.difference(
        population_data.columns
    )
    if missing_columns:
        raise ValueError(
            f"Population data is missing required columns: {sorted(missing_columns)}"
        )

    population_frame = population_data.loc[:, ID_COLUMNS].copy()
    population_frame["sector_id"] = population_frame["sector_id"].astype(
        str
    )

    start_values = population_frame["population_2010"].to_numpy(
        dtype=np.float64
    )
    end_values = population_frame["population_2022"].to_numpy(
        dtype=np.float64
    )
    week_fractions = build_week_fractions(len(epidemic_weeks))

    interpolated_matrix = start_values[:, None] + (
        (end_values - start_values)[:, None] * week_fractions[None, :]
    )
    # Clamp negative values to zero (artifact from sectors where 2022 pop < 2010 pop)
    interpolated_matrix = np.maximum(interpolated_matrix, 0)
    interpolated_matrix = _round_to_int(interpolated_matrix)

    weekly_columns = pd.DataFrame(
        interpolated_matrix,
        columns=epidemic_weeks,
        index=population_frame.index,
    )

    interpolated_population = pd.concat(
        [
            population_frame.reset_index(drop=True),
            weekly_columns.reset_index(drop=True),
        ],
        axis=1,
    )

    return interpolated_population


def melt_population_table(population_table: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide interpolation table into a long format for plotting."""
    week_columns = [
        column
        for column in population_table.columns
        if column not in REQUIRED_POPULATION_COLUMNS
    ]

    melted_table = population_table.melt(
        id_vars=ID_COLUMNS,
        value_vars=week_columns,
        var_name="epidemic_date",
        value_name="interpolated_population",
    )
    melted_table["week_index"] = melted_table["epidemic_date"].map(
        {week: index for index, week in enumerate(week_columns)}
    )

    return melted_table.sort_values(
        ["sector_id", "week_index"]
    ).reset_index(drop=True)


def save_interpolated_population_table(
    population_table: pd.DataFrame,
    file_path: str | Path,
) -> None:
    """Persist the interpolated population table to CSV."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    population_table.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate the population comparison table across the epidemic-week "
            "sequence from the processed dengue dataset."
        )
    )
    parser.add_argument("--params-path", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--population-path", default=None)
    parser.add_argument("--dengue-path", default=None)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the interpolation pipeline."""
    args = parse_args()
    params = load_params(args.params_path)
    default_population_path, default_dengue_path, default_output_path = (
        resolve_default_paths(params)
    )

    population_path = (
        Path(args.population_path)
        if args.population_path
        else default_population_path
    )
    dengue_path = (
        Path(args.dengue_path) if args.dengue_path else default_dengue_path
    )
    output_path = (
        Path(args.output_path) if args.output_path else default_output_path
    )

    population_data = load_population_data(population_path)
    epidemic_weeks = load_epidemic_weeks(dengue_path)
    interpolated_population = build_interpolated_population_table(
        population_data,
        epidemic_weeks,
    )
    save_interpolated_population_table(
        interpolated_population, output_path
    )

    print(
        "Interpolated population table saved to "
        f"{output_path} with {len(interpolated_population):,} sectors and "
        f"{len(epidemic_weeks):,} epidemic weeks."
    )


if __name__ == "__main__":
    main()
