"""Tests for the dengue cases per capita dataset."""

import re
import os

import numpy as np
import pandas as pd
import pytest
import yaml

params = yaml.safe_load(open("params.yaml"))

PER_CAPITA_MULTIPLIER = 1_000


def _epidemic_date_valid(s: str) -> bool:
    m = re.match(r"^20(\d{2})_(\d{2})W(\d{1,2})$", str(s))
    if not m:
        return False
    y1, y2, week = map(int, m.groups())
    return y2 == y1 + 1 and 1 <= week <= 53


class TestDenguePerCapita:
    """Validate the dengue_per_capita.csv output."""

    @pytest.fixture(scope="class")
    def paths(self):
        proc = params["all"]["paths"]["data"]["processed"]
        return {
            "per_capita": proc["dengue_per_capita"],
            "dengue": proc["dengue"],
            "population": proc["population_interpolated"],
        }

    @pytest.fixture(scope="class")
    def per_capita_df(self, paths):
        path = paths["per_capita"]
        if not os.path.exists(path):
            pytest.skip(f"Per-capita file not found: {path}")
        return pd.read_csv(path)

    @pytest.fixture(scope="class")
    def dengue_df(self, paths):
        path = paths["dengue"]
        if not os.path.exists(path):
            pytest.skip(f"Dengue file not found: {path}")
        return pd.read_csv(path, low_memory=False)

    @pytest.fixture(scope="class")
    def population_df(self, paths):
        path = paths["population"]
        if not os.path.exists(path):
            pytest.skip(f"Population file not found: {path}")
        return pd.read_csv(path)

    # ------------------------------------------------------------------
    # Structure tests
    # ------------------------------------------------------------------

    def test_file_exists_and_not_empty(self, paths):
        path = paths["per_capita"]
        assert os.path.exists(path), f"File should exist: {path}"
        assert os.path.getsize(path) > 0, "File should not be empty"

    def test_required_columns(self, per_capita_df):
        required = {
            "sector_id",
            "epidemic_date",
            "case_count",
            "population",
            "cases_per_1000",
        }
        missing = required - set(per_capita_df.columns)
        assert not missing, f"Missing columns: {sorted(missing)}"

    def test_no_duplicate_sector_week(self, per_capita_df):
        dupes = per_capita_df.duplicated(
            subset=["sector_id", "epidemic_date"]
        ).sum()
        assert dupes == 0, f"Found {dupes} duplicate sector-week rows"

    # ------------------------------------------------------------------
    # Data type tests
    # ------------------------------------------------------------------

    def test_case_count_is_integer(self, per_capita_df):
        assert pd.api.types.is_integer_dtype(
            per_capita_df["case_count"]
        ), "case_count should be integer"

    def test_population_is_numeric(self, per_capita_df):
        assert pd.api.types.is_numeric_dtype(
            per_capita_df["population"]
        ), "population should be numeric"

    def test_cases_per_1000_is_numeric(self, per_capita_df):
        assert pd.api.types.is_numeric_dtype(
            per_capita_df["cases_per_1000"]
        ), "cases_per_1000 should be numeric"

    # ------------------------------------------------------------------
    # Value validity tests
    # ------------------------------------------------------------------

    def test_case_count_non_negative(self, per_capita_df):
        assert (per_capita_df["case_count"] >= 0).all(), (
            "case_count must be non-negative"
        )

    def test_population_not_null(self, per_capita_df):
        assert per_capita_df["population"].notna().all(), (
            "population should not have NaN values"
        )

    def test_population_non_negative(self, per_capita_df):
        assert (per_capita_df["population"] >= 0).all(), (
            "population must be non-negative (interpolation clamps to zero)"
        )

    def test_cases_per_1000_nan_when_population_non_positive(
        self, per_capita_df
    ):
        non_pos = per_capita_df[per_capita_df["population"] <= 0]
        if not non_pos.empty:
            assert non_pos["cases_per_1000"].isna().all(), (
                "cases_per_1000 should be NaN when population is <= 0"
            )

    def test_cases_per_1000_non_negative(self, per_capita_df):
        valid = per_capita_df["cases_per_1000"].dropna()
        assert (valid >= 0).all(), "cases_per_1000 must be non-negative"

    def test_zero_cases_yield_zero_rate(self, per_capita_df):
        zero_cases = per_capita_df[
            (per_capita_df["case_count"] == 0)
            & (per_capita_df["population"] > 0)
        ]
        if not zero_cases.empty:
            assert (zero_cases["cases_per_1000"] == 0.0).all(), (
                "Sectors with zero cases should have rate 0"
            )

    # ------------------------------------------------------------------
    # Sector ID format
    # ------------------------------------------------------------------

    def test_sector_ids_bh_format(self, per_capita_df):
        sector_ids = per_capita_df["sector_id"].astype(str)
        bh_pattern = sector_ids.str.startswith("3106200")
        assert bh_pattern.all(), (
            "All sector IDs should start with 3106200 (Belo Horizonte)"
        )

    # ------------------------------------------------------------------
    # Epidemic date format
    # ------------------------------------------------------------------

    def test_epidemic_dates_valid(self, per_capita_df):
        valid = per_capita_df["epidemic_date"].apply(_epidemic_date_valid)
        assert valid.all(), (
            "All epidemic_date values should match YYYY_YYWww format"
        )

    # ------------------------------------------------------------------
    # Calculation accuracy (sample check)
    # ------------------------------------------------------------------

    def test_per_capita_formula(self, per_capita_df):
        """Verify that cases_per_1000 == case_count / population * 1000."""
        valid = per_capita_df[
            (per_capita_df["population"] > 0)
            & per_capita_df["cases_per_1000"].notna()
        ].copy()
        if valid.empty:
            pytest.skip("No rows with valid population to check formula")
        expected = (
            valid["case_count"] / valid["population"]
        ) * PER_CAPITA_MULTIPLIER
        assert np.allclose(
            valid["cases_per_1000"].values, expected.values, atol=1e-6
        ), "Per-capita formula mismatch"

    # ------------------------------------------------------------------
    # Cross-validation with source data
    # ------------------------------------------------------------------

    def test_total_cases_match_dengue_source(
        self, per_capita_df, dengue_df
    ):
        """Total aggregated cases should match the dengue source (for sectors with data)."""
        dengue_valid = dengue_df.dropna(
            subset=["population_sector", "epidemic_date"]
        )
        source_total = len(dengue_valid)
        computed_total = per_capita_df["case_count"].sum()
        assert computed_total == source_total, (
            f"Total cases mismatch: per_capita has {computed_total:,}, "
            f"dengue source has {source_total:,}"
        )

    def test_sector_coverage(self, per_capita_df, population_df):
        """Per-capita sectors should match the interpolated population sectors."""
        pop_sectors = set(population_df["sector_id"].astype(str).unique())
        pc_sectors = set(per_capita_df["sector_id"].astype(str).unique())
        assert pc_sectors == pop_sectors, (
            f"Sector mismatch: {len(pc_sectors - pop_sectors)} extra, "
            f"{len(pop_sectors - pc_sectors)} missing"
        )

    def test_week_coverage(self, per_capita_df, population_df):
        """Per-capita weeks should match the interpolated population weeks."""
        pop_weeks = set(
            c
            for c in population_df.columns
            if c not in ("sector_id", "population_2010", "population_2022")
        )
        pc_weeks = set(per_capita_df["epidemic_date"].unique())
        assert pc_weeks == pop_weeks, (
            f"Week mismatch: {len(pc_weeks - pop_weeks)} extra, "
            f"{len(pop_weeks - pc_weeks)} missing"
        )
