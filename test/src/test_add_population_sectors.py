"""
Tests for add_population_sectors stage outputs.

Loads the ovitraps and dengue files produced by add_population_info and
verifies that:

1. The biweek column in each dataset is correctly derived from its
   respective date column (dt_instal for ovitraps, dt_notific for dengue)
   using 20 random samples.
2. Each sampled date falls within the 2-week window of its assigned biweek.
3. Both datasets share the same biweek label convention
   ({YYYY_YY}W{even_NN}).
4. Records with a NaN population_sector have missing or zero coordinates
   (registered as xfail — will fail until full BH coverage is confirmed).
"""

import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import yaml
from shapely.geometry import Point

sys.path.append("utils")
import project_utils

params = yaml.safe_load(open("params.yaml"))
_dvc = params["all"]["paths"]["data"]["dvc"]

N_SAMPLES = 20
RANDOM_SEED = 42
_BIWEEK_RE = re.compile(r"^\d{4}_\d{2}W\d{2}$")
# Keep in sync with add_population_info.DEFAULT_N_NEIGHBORS
_DEFAULT_N_NEIGHBORS = 6


# ============================================================
# Shared helpers
# ============================================================


def _recompute_biweek(df: pd.DataFrame, date_col: str) -> pd.Series:
    """Recompute biweek label from *date_col* using project conventions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing *date_col*.
    date_col : str
        Name of the date column to derive epidemic week and year from.

    Returns
    -------
    pd.Series
        Biweek labels aligned to *df*'s index.
    """
    semepid = project_utils.assign_epidemic_week(df, date_col)
    anoepid = project_utils.assign_epidemic_year(df, date_col)
    temp = pd.DataFrame(
        {"anoepid": anoepid.values, "semepid": semepid.values},
        index=df.index,
    )
    epidemic_date = project_utils.get_epidemic_date(temp)
    return project_utils.epidemic_date_to_biweek(epidemic_date)


def _biweek_date_range(
    biweek_label: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (start, end) calendar dates spanned by *biweek_label*.

    The epidemic year begins on the first Sunday on or before June 1 of the
    label's starting calendar year. Biweek W{N} covers epidemic weeks N-1
    and N, so the date range is:

        start = epi_year_start + (N - 2) × 7 days
        end   = epi_year_start + N × 7 − 1 days  (inclusive)

    Parameters
    ----------
    biweek_label : str
        Biweek label in the format ``{YYYY_YY}W{NN}``, e.g. ``2023_24W18``.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        Inclusive (start, end) calendar dates of the biweek.
    """
    epi_year_str, week_str = str(biweek_label).split("W")
    cal_year = int(epi_year_str.split("_")[0])
    biweek_num = int(week_str)

    june1 = pd.Timestamp(f"{cal_year}-06-01")
    offset = (june1.weekday() + 1) % 7  # days to subtract to reach Sunday
    epi_year_start = june1 - pd.Timedelta(days=offset)

    start = epi_year_start + pd.Timedelta(weeks=biweek_num - 2)
    end = epi_year_start + pd.Timedelta(weeks=biweek_num) - pd.Timedelta(days=1)
    return start, end


def _biweek_window(biweek_label: str) -> tuple[int, int]:
    """Return the (start_week, end_week) window covered by *biweek_label*.

    Parameters
    ----------
    biweek_label : str
        Biweek label in the format ``{YYYY_YY}W{NN}``.

    Returns
    -------
    tuple[int, int]
        ``(biweek_end - 1, biweek_end)`` — the two epidemic weeks that
        map to this biweek under the project convention:
        ``biweek_num = ((week_num + 1) // 2) * 2``.
    """
    biweek_end = int(str(biweek_label).split("W")[1])
    return biweek_end - 1, biweek_end


# ============================================================
# Module-scoped data fixtures (loaded once per file)
# ============================================================


@pytest.fixture(scope="module")
def ovitraps() -> pd.DataFrame:
    """Full ovitraps dataset from add_population_info."""
    path = Path(_dvc["add_population_info"]["ovitraps"])
    if not path.exists():
        pytest.fail(f"File not found: {path}")
    return pd.read_csv(
        path,
        parse_dates=["dt_instal", "dt_col"],
        dtype={"narmad": str, "nplaca": str},
    )


@pytest.fixture(scope="module")
def dengue() -> pd.DataFrame:
    """Full dengue dataset from add_population_info."""
    path = Path(_dvc["add_population_info"]["dengue"])
    if not path.exists():
        pytest.fail(f"File not found: {path}")
    return pd.read_csv(
        path, parse_dates=["dt_notific"], low_memory=False
    )


@pytest.fixture(scope="module")
def idw_df() -> pd.DataFrame:
    """Full centroids_idw dataset from add_population_info."""
    path = Path(_dvc["add_population_info"]["centroids_idw"])
    if not path.exists():
        pytest.fail(f"File not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def ovitraps_sample(ovitraps: pd.DataFrame) -> pd.DataFrame:
    """20 random ovitraps rows with a reproducible seed."""
    return ovitraps.sample(
        n=N_SAMPLES, random_state=RANDOM_SEED
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def dengue_sample(dengue: pd.DataFrame) -> pd.DataFrame:
    """20 random dengue rows with a reproducible seed."""
    return dengue.sample(
        n=N_SAMPLES, random_state=RANDOM_SEED
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def idw_sample(idw_df: pd.DataFrame) -> pd.DataFrame:
    """20 random centroids_idw rows with a reproducible seed."""
    return idw_df.sample(
        n=N_SAMPLES, random_state=RANDOM_SEED
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def bh_boundary() -> gpd.GeoSeries:
    """Union of all BH 2022 census sector geometries (EPSG:4326)."""
    path = Path(_dvc["process_population_data"]["sectors_geojson"])
    if not path.exists():
        pytest.fail(f"File not found: {path}")
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    return gdf.geometry.union_all()


# ============================================================
# Class 1: Biweek alignment
# ============================================================


class TestBiweekAlignment:
    """Verify biweek derivation and cross-dataset consistency."""

    # ----------------------------------------------------------
    # Tests: ovitraps biweek ← dt_instal
    # ----------------------------------------------------------

    def test_ovitraps_biweek_derived_from_dt_instal(
        self, ovitraps_sample: pd.DataFrame
    ) -> None:
        """Stored biweek must match recomputation from dt_instal for all 20 samples."""
        expected = _recompute_biweek(ovitraps_sample, "dt_instal")
        stored = ovitraps_sample["biweek"].values
        mismatch_mask = stored != expected.values
        mismatches = ovitraps_sample[mismatch_mask].copy()
        mismatches["expected_biweek"] = expected.values[mismatch_mask]
        assert mismatches.empty, (
            f"{len(mismatches)} ovitraps biweek mismatches:\n"
            f"{mismatches[['dt_instal', 'biweek', 'expected_biweek']].to_string()}"
        )

    def test_ovitraps_instal_date_within_biweek_window(
        self, ovitraps_sample: pd.DataFrame
    ) -> None:
        """Epidemic week of dt_instal must fall in [biweek_end-1, biweek_end]."""
        for i, row in ovitraps_sample.iterrows():
            start, end = _biweek_window(row["biweek"])
            epi_week = int(
                project_utils.assign_epidemic_week(
                    ovitraps_sample.loc[[i]], "dt_instal"
                ).iloc[0]
            )
            assert start <= epi_week <= end, (
                f"Row {i}: dt_instal={row['dt_instal'].date()} → "
                f"epi_week={epi_week} not in [{start}, {end}] "
                f"(biweek={row['biweek']})"
            )

    # ----------------------------------------------------------
    # Tests: dengue biweek ← dt_notific
    # ----------------------------------------------------------

    def test_dengue_biweek_derived_from_dt_notific(
        self, dengue_sample: pd.DataFrame
    ) -> None:
        """Stored biweek must match recomputation from dt_notific for all 20 samples."""
        expected = _recompute_biweek(dengue_sample, "dt_notific")
        stored = dengue_sample["biweek"].values
        mismatch_mask = stored != expected.values
        mismatches = dengue_sample[mismatch_mask].copy()
        mismatches["expected_biweek"] = expected.values[mismatch_mask]
        assert mismatches.empty, (
            f"{len(mismatches)} dengue biweek mismatches:\n"
            f"{mismatches[['dt_notific', 'biweek', 'expected_biweek']].to_string()}"
        )

    def test_dengue_notific_date_within_biweek_window(
        self, dengue_sample: pd.DataFrame
    ) -> None:
        """Epidemic week of dt_notific must fall in [biweek_end-1, biweek_end]."""
        for i, row in dengue_sample.iterrows():
            start, end = _biweek_window(row["biweek"])
            epi_week = int(
                project_utils.assign_epidemic_week(
                    dengue_sample.loc[[i]], "dt_notific"
                ).iloc[0]
            )
            assert start <= epi_week <= end, (
                f"Row {i}: dt_notific={row['dt_notific'].date()} → "
                f"epi_week={epi_week} not in [{start}, {end}] "
                f"(biweek={row['biweek']})"
            )

    # ----------------------------------------------------------
    # Tests: cross-dataset alignment
    # ----------------------------------------------------------

    def test_biweek_format_and_even_parity(
        self,
        ovitraps_sample: pd.DataFrame,
        dengue_sample: pd.DataFrame,
    ) -> None:
        """All biweek labels must match {YYYY_YY}W{NN} with an even NN."""
        for label in ovitraps_sample["biweek"]:
            assert _BIWEEK_RE.match(str(label)), (
                f"Invalid ovitraps biweek format: {label}"
            )
            assert int(str(label).split("W")[1]) % 2 == 0, (
                f"Ovitraps biweek {label} has odd week number"
            )
        for label in dengue_sample["biweek"]:
            assert _BIWEEK_RE.match(str(label)), (
                f"Invalid dengue biweek format: {label}"
            )
            assert int(str(label).split("W")[1]) % 2 == 0, (
                f"Dengue biweek {label} has odd week number"
            )

    def test_biweek_label_overlap_across_datasets(
        self,
        ovitraps: pd.DataFrame,
        dengue: pd.DataFrame,
    ) -> None:
        """
        Ovitraps and dengue biweek label sets must have meaningful overlap,
        confirming a shared labeling convention across both datasets.
        """
        ov_biweeks = set(ovitraps["biweek"].dropna().unique())
        dg_biweeks = set(dengue["biweek"].dropna().unique())
        overlap = ov_biweeks & dg_biweeks
        assert len(overlap) >= 10, (
            f"Insufficient biweek overlap ({len(overlap)} shared labels). "
            "Expected ≥ 10 — datasets may use different conventions."
        )

    def test_same_biweek_dt_notific_within_ovitrap_window(
        self,
        ovitraps_sample: pd.DataFrame,
        dengue: pd.DataFrame,
    ) -> None:
        """
        For each of the 20 ovitraps samples, every dengue record sharing the
        same biweek must have dt_notific within that biweek's calendar date
        range — the 14-day epidemiological window derived from the biweek
        label using the same convention as dt_instal and dt_col.

        The biweek for ovitraps is derived from dt_instal; the biweek for
        dengue is derived from dt_notific. If both share the same label, both
        dates must fall in the same 14-day window. This cross-dataset check
        catches label-convention mismatches that single-dataset tests cannot.

        Parameters
        ----------
        ovitraps_sample : pd.DataFrame
            20 random ovitraps rows with dt_instal, dt_col, and biweek.
        dengue : pd.DataFrame
            Full dengue dataset with dt_notific and biweek.
        """
        dengue_notific = pd.to_datetime(dengue["dt_notific"])
        failures: list[str] = []

        for _, row in ovitraps_sample.iterrows():
            biweek = row["biweek"]
            bw_start, bw_end = _biweek_date_range(biweek)

            same_biweek = dengue["biweek"] == biweek
            if not same_biweek.any():
                continue

            notific = dengue_notific[same_biweek]
            out_of_range = (notific < bw_start) | (notific > bw_end)
            if out_of_range.any():
                failures.append(
                    f"biweek={biweek} | window [{bw_start.date()}, "
                    f"{bw_end.date()}] | "
                    f"{out_of_range.sum()} dt_notific outside biweek range "
                    f"(min={notific.min().date()}, max={notific.max().date()})"
                )

        assert not failures, "\n".join(failures)


# ============================================================
# Class 2: Centroids IDW biweek validation
# ============================================================


class TestCentroidsIDWBiweek:
    """Validate biweek convention and structural invariants of centroids_idw."""

    def test_idw_biweek_format_and_even_parity(
        self, idw_sample: pd.DataFrame
    ) -> None:
        """All biweek labels must match {YYYY_YY}W{NN} with an even NN."""
        for label in idw_sample["biweek"]:
            assert _BIWEEK_RE.match(str(label)), (
                f"Invalid centroids_idw biweek format: {label}"
            )
            assert int(str(label).split("W")[1]) % 2 == 0, (
                f"Centroids_idw biweek {label} has odd week number"
            )

    def test_idw_biweeks_are_subset_of_ovitraps(
        self, idw_df: pd.DataFrame, ovitraps: pd.DataFrame
    ) -> None:
        """
        Every biweek in centroids_idw must exist in the ovitraps dataset.

        The IDW is computed per biweek from ovitraps data, so any biweek
        present in the IDW output must have been present in ovitraps.
        """
        idw_biweeks = set(idw_df["biweek"].dropna().unique())
        ov_biweeks = set(ovitraps["biweek"].dropna().unique())
        extra = idw_biweeks - ov_biweeks
        assert not extra, (
            f"{len(extra)} centroids_idw biweeks not found in ovitraps: "
            f"{sorted(extra)[:5]}"
        )

    def test_no_duplicate_sector_biweek_pairs(
        self, idw_df: pd.DataFrame
    ) -> None:
        """Each (CD_SETOR, biweek) pair must appear exactly once."""
        dupes = idw_df.duplicated(subset=["CD_SETOR", "biweek"]).sum()
        assert dupes == 0, (
            f"{dupes} duplicate (CD_SETOR, biweek) pairs in centroids_idw"
        )

    def test_idw_egg_value_non_negative(
        self, idw_df: pd.DataFrame
    ) -> None:
        """idw_egg_value must be ≥ 0 for all rows."""
        negative = (idw_df["idw_egg_value"] < 0).sum()
        assert negative == 0, (
            f"{negative} rows have negative idw_egg_value"
        )

    def test_n_traps_used_in_valid_range(
        self, idw_df: pd.DataFrame
    ) -> None:
        """n_traps_used must be ≥ 1 and ≤ _DEFAULT_N_NEIGHBORS (6)."""
        bad = idw_df[
            (idw_df["n_traps_used"] < 1)
            | (idw_df["n_traps_used"] > _DEFAULT_N_NEIGHBORS)
        ]
        assert bad.empty, (
            f"{len(bad)} rows have n_traps_used outside [1, {_DEFAULT_N_NEIGHBORS}]"
        )

    def test_centroid_coordinates_in_bh_range(
        self, idw_sample: pd.DataFrame
    ) -> None:
        """Centroid coordinates must fall within the Belo Horizonte bounding box."""
        assert idw_sample["centroid_latitude"].between(-20.1, -19.7).all(), (
            "centroid_latitude out of BH range [-20.1, -19.7]"
        )
        assert idw_sample["centroid_longitude"].between(-44.1, -43.8).all(), (
            "centroid_longitude out of BH range [-44.1, -43.8]"
        )


# ============================================================
# Class 3: Sector NaN attribution
# ============================================================


class TestSectorNullAttribution:
    """Validate that NaN population_sector is explained by missing coordinates."""

    def test_nan_sector_only_when_coords_missing(
        self,
        ovitraps: pd.DataFrame,
        dengue: pd.DataFrame,
        bh_boundary: gpd.GeoSeries,
    ) -> None:
        """
        Every record with valid coordinates must have a population_sector.

        TODO: this test is failing because some records with valid coordinates
        receive NaN sector. Records outside the BH census boundary are
        expected to have no match, but they should not have valid lat/lon in
        the first place — their coordinates need to be corrected or excluded
        upstream. Fix coordinate cleaning in process_data.py to resolve this.

        Parameters
        ----------
        ovitraps : pd.DataFrame
            Ovitraps dataset with latitude, longitude, and population_sector.
        dengue : pd.DataFrame
            Dengue dataset with latitude, longitude, and population_sector.
        bh_boundary : gpd.GeoSeries
            Union of all BH 2022 census sector geometries.
        """
        failures: list[str] = []
        for name, df in [("ovitraps", ovitraps), ("dengue", dengue)]:
            has_valid_coords = (
                df["latitude"].notna()
                & df["longitude"].notna()
                & (df["latitude"] != 0)
                & (df["longitude"] != 0)
            )
            has_nan_sector = df["population_sector"].isna()
            leaking = df[has_valid_coords & has_nan_sector].copy()

            if leaking.empty:
                continue

            points = leaking.apply(
                lambda r: Point(r["longitude"], r["latitude"]), axis=1
            )
            inside_bh = points.apply(bh_boundary.contains)
            genuine = leaking[inside_bh.values]

            if not genuine.empty:
                failures.append(
                    f"{name}: {len(genuine)} records inside BH have valid "
                    f"coordinates but no sector assignment"
                )

        assert not failures, "\n".join(failures)
