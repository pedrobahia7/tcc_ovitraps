import numpy as np
import pandas as pd
import sys
sys.path.append("src")
from census_equivalence_2010_2022 import build_extrapolated_population_table


def test_build_extrapolated_population_table_uses_integer_week_values() -> None:
    """Interpolation/extrapolation should keep integer values for all weeks."""
    population_data = pd.DataFrame(
        {
            "sector_id": ["s_decrease", "s_increase"],
            "population_2010": [514, 259],
            "population_2022": [467, 448],
        }
    )
    epidemic_weeks = ["2006_07W32", "2010_11W01", "2022_23W01", "2024_25W44"]

    interpolated = build_extrapolated_population_table(population_data, epidemic_weeks)

    assert interpolated.columns.tolist() == [
        "sector_id",
        *epidemic_weeks,
    ]
    for week in epidemic_weeks:
        assert np.issubdtype(interpolated[week].dtype, np.integer)


def test_build_extrapolated_population_table_preserves_anchors_and_extrapolates() -> None:
    """Anchor weeks should match real values and limits must extrapolate linearly."""
    population_data = pd.DataFrame(
        {
            "sector_id": ["s_decrease", "s_increase"],
            "population_2010": [514, 259],
            "population_2022": [467, 448],
        }
    )
    epidemic_weeks = ["2006_07W32", "2010_11W01", "2022_23W01", "2024_25W44"]

    interpolated = build_extrapolated_population_table(population_data, epidemic_weeks)

    decreasing = interpolated[interpolated["sector_id"] == "s_decrease"].iloc[0]
    increasing = interpolated[interpolated["sector_id"] == "s_increase"].iloc[0]

    # Real anchor points must remain unchanged.
    assert decreasing["2010_11W01"] == 514
    assert decreasing["2022_23W01"] == 467
    assert increasing["2010_11W01"] == 259
    assert increasing["2022_23W01"] == 448

    # For negative slope: values before 2010 are higher; after 2022 are lower.
    assert decreasing["2006_07W32"] > decreasing["2010_11W01"]
    assert decreasing["2024_25W44"] < decreasing["2022_23W01"]

    # For positive slope: values before 2010 are lower; after 2022 are higher.
    assert increasing["2006_07W32"] < increasing["2010_11W01"]
    assert increasing["2024_25W44"] > increasing["2022_23W01"]


def test_build_extrapolated_population_table_raises_with_empty_week_list() -> None:
    """The function should reject an empty epidemic-week list."""
    population_table = pd.DataFrame(
        {
            "sector_id": ["1"],
            "population_2010": [100],
            "population_2022": [130],
        }
    )

    try:
        build_extrapolated_population_table(population_table, [])
    except ValueError as error:
        assert "must not be empty" in str(error)
    else:
        raise AssertionError("Expected ValueError for empty epidemic_weeks")
