"""Tests for the sector centroids and IDW interpolation datasets."""

import re
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml

params = yaml.safe_load(open("params.yaml"))


def _biweek_valid(s: str) -> bool:
    """Validate biweek format: YYYY_YYWww where ww is even."""
    m = re.match(r"^20(\d{2})_(\d{2})W(\d{1,2})$", str(s))
    if not m:
        return False
    y1, y2, week = map(int, m.groups())
    return y2 == y1 + 1 and 2 <= week <= 54 and week % 2 == 0


class TestSectorCentroids:
    """Validate bh_sectors_2022_centroids.csv."""

    @pytest.fixture(scope="class")
    def paths(self):
        proc = params["all"]["paths"]["data"]["processed"]
        return {
            "centroids": proc["centroids"],
            "centroids_idw": proc["centroids_idw"],
            "sectors_geojson": proc["census_equivalence"]["sectors_geojson"],
            "ovitraps": proc["ovitraps"],
        }

    @pytest.fixture(scope="class")
    def centroids_df(self, paths):
        path = paths["centroids"]
        if not os.path.exists(path):
            pytest.skip(f"Centroids file not found: {path}")
        return pd.read_csv(path)

    @pytest.fixture(scope="class")
    def sectors_gdf(self, paths):
        path = paths["sectors_geojson"]
        if not os.path.exists(path):
            pytest.skip(f"GeoJSON file not found: {path}")
        return gpd.read_file(path)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def test_file_exists_and_not_empty(self, paths):
        path = paths["centroids"]
        assert os.path.exists(path), f"File should exist: {path}"
        assert os.path.getsize(path) > 0, "File should not be empty"

    def test_required_columns(self, centroids_df):
        required = {
            "CD_SETOR",
            "centroid_latitude",
            "centroid_longitude",
        }
        missing = required - set(centroids_df.columns)
        assert not missing, f"Missing columns: {sorted(missing)}"

    def test_no_duplicate_sectors(self, centroids_df):
        dupes = centroids_df.duplicated(subset=["CD_SETOR"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate sector rows"

    # ------------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------------

    def test_latitude_is_numeric(self, centroids_df):
        assert pd.api.types.is_numeric_dtype(
            centroids_df["centroid_latitude"]
        ), "centroid_latitude should be numeric"

    def test_longitude_is_numeric(self, centroids_df):
        assert pd.api.types.is_numeric_dtype(
            centroids_df["centroid_longitude"]
        ), "centroid_longitude should be numeric"

    # ------------------------------------------------------------------
    # Value validity
    # ------------------------------------------------------------------

    def test_latitude_not_null(self, centroids_df):
        assert centroids_df["centroid_latitude"].notna().all(), (
            "centroid_latitude should not have NaN values"
        )

    def test_longitude_not_null(self, centroids_df):
        assert centroids_df["centroid_longitude"].notna().all(), (
            "centroid_longitude should not have NaN values"
        )

    def test_latitude_bh_range(self, centroids_df):
        lat = centroids_df["centroid_latitude"]
        assert (lat >= -20.1).all() and (lat <= -19.7).all(), (
            "Latitudes should be within Belo Horizonte bounds (-20.1 to -19.7)"
        )

    def test_longitude_bh_range(self, centroids_df):
        lon = centroids_df["centroid_longitude"]
        assert (lon >= -44.1).all() and (lon <= -43.8).all(), (
            "Longitudes should be within Belo Horizonte bounds (-44.1 to -43.8)"
        )

    def test_sector_ids_bh_format(self, centroids_df):
        sector_ids = centroids_df["CD_SETOR"].astype(str)
        bh_pattern = sector_ids.str.startswith("3106200")
        assert bh_pattern.all(), (
            "All sector IDs should start with 3106200 (Belo Horizonte)"
        )

    # ------------------------------------------------------------------
    # Cross-validation with GeoJSON
    # ------------------------------------------------------------------

    def test_sector_coverage_matches_geojson(self, centroids_df, sectors_gdf):
        geo_sectors = set(sectors_gdf["CD_SETOR"].astype(str).unique())
        centroid_sectors = set(centroids_df["CD_SETOR"].astype(str).unique())
        assert centroid_sectors == geo_sectors, (
            f"Sector mismatch: {len(centroid_sectors - geo_sectors)} extra, "
            f"{len(geo_sectors - centroid_sectors)} missing"
        )


class TestCentroidsIDW:
    """Validate sector_centroids_with_idw.csv."""

    @pytest.fixture(scope="class")
    def paths(self):
        proc = params["all"]["paths"]["data"]["processed"]
        return {
            "centroids": proc["centroids"],
            "centroids_idw": proc["centroids_idw"],
            "ovitraps": proc["ovitraps"],
        }

    @pytest.fixture(scope="class")
    def idw_df(self, paths):
        path = paths["centroids_idw"]
        if not os.path.exists(path):
            pytest.skip(f"IDW file not found: {path}")
        return pd.read_csv(path)

    @pytest.fixture(scope="class")
    def centroids_df(self, paths):
        path = paths["centroids"]
        if not os.path.exists(path):
            pytest.skip(f"Centroids file not found: {path}")
        return pd.read_csv(path)

    @pytest.fixture(scope="class")
    def ovitraps_df(self, paths):
        path = paths["ovitraps"]
        if not os.path.exists(path):
            pytest.skip(f"Ovitraps file not found: {path}")
        return pd.read_csv(path, low_memory=False)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def test_file_exists_and_not_empty(self, paths):
        path = paths["centroids_idw"]
        assert os.path.exists(path), f"File should exist: {path}"
        assert os.path.getsize(path) > 0, "File should not be empty"

    def test_required_columns(self, idw_df):
        required = {
            "CD_SETOR",
            "biweek",
            "idw_egg_value",
            "centroid_latitude",
            "centroid_longitude",
            "n_traps_used",
            "min_distance_km",
            "narmads_used",
        }
        missing = required - set(idw_df.columns)
        assert not missing, f"Missing columns: {sorted(missing)}"

    def test_no_duplicate_sector_biweek(self, idw_df):
        dupes = idw_df.duplicated(subset=["CD_SETOR", "biweek"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate sector-biweek rows"

    # ------------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------------

    def test_idw_egg_value_is_numeric(self, idw_df):
        assert pd.api.types.is_numeric_dtype(
            idw_df["idw_egg_value"]
        ), "idw_egg_value should be numeric"

    def test_n_traps_used_is_integer(self, idw_df):
        assert pd.api.types.is_integer_dtype(
            idw_df["n_traps_used"]
        ), "n_traps_used should be integer"

    def test_min_distance_km_is_numeric(self, idw_df):
        assert pd.api.types.is_numeric_dtype(
            idw_df["min_distance_km"]
        ), "min_distance_km should be numeric"

    # ------------------------------------------------------------------
    # Value validity
    # ------------------------------------------------------------------

    def test_idw_egg_value_not_null(self, idw_df):
        assert idw_df["idw_egg_value"].notna().all(), (
            "idw_egg_value should not have NaN values"
        )

    def test_idw_egg_value_non_negative(self, idw_df):
        assert (idw_df["idw_egg_value"] >= 0).all(), (
            "idw_egg_value must be non-negative"
        )

    def test_n_traps_used_positive(self, idw_df):
        assert (idw_df["n_traps_used"] > 0).all(), (
            "n_traps_used must be positive"
        )

    def test_min_distance_km_non_negative(self, idw_df):
        assert (idw_df["min_distance_km"] >= 0).all(), (
            "min_distance_km must be non-negative"
        )

    def test_narmads_used_not_empty(self, idw_df):
        assert idw_df["narmads_used"].notna().all(), (
            "narmads_used should not have NaN values"
        )
        assert (idw_df["narmads_used"].astype(str).str.len() > 0).all(), (
            "narmads_used should not be empty strings"
        )

    # ------------------------------------------------------------------
    # Biweek format
    # ------------------------------------------------------------------

    def test_biweeks_valid(self, idw_df):
        valid = idw_df["biweek"].apply(_biweek_valid)
        assert valid.all(), (
            "All biweek values should match YYYY_YYWww format with even week number"
        )

    # ------------------------------------------------------------------
    # Sector ID format
    # ------------------------------------------------------------------

    def test_sector_ids_bh_format(self, idw_df):
        sector_ids = idw_df["CD_SETOR"].astype(str)
        bh_pattern = sector_ids.str.startswith("3106200")
        assert bh_pattern.all(), (
            "All sector IDs should start with 3106200 (Belo Horizonte)"
        )

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def test_sector_coverage_matches_centroids(self, idw_df, centroids_df):
        centroid_sectors = set(centroids_df["CD_SETOR"].astype(str).unique())
        idw_sectors = set(idw_df["CD_SETOR"].astype(str).unique())
        assert idw_sectors == centroid_sectors, (
            f"Sector mismatch: {len(idw_sectors - centroid_sectors)} extra, "
            f"{len(centroid_sectors - idw_sectors)} missing"
        )

    def test_biweek_coverage_matches_ovitraps(self, idw_df, ovitraps_df):
        """IDW biweeks should match the biweeks derived from ovitraps data."""
        ovi = ovitraps_df.dropna(subset=["latitude", "longitude", "epidemic_date"])
        epi_year = ovi["epidemic_date"].str.split("W").str[0]
        week_num = ovi["epidemic_date"].str.split("W").str[1].astype(int)
        biweek_num = ((week_num + 1) // 2) * 2
        ovi_biweeks = set(
            epi_year + "W" + biweek_num.astype(str).str.zfill(2)
        )
        idw_biweeks = set(idw_df["biweek"].unique())
        assert idw_biweeks == ovi_biweeks, (
            f"Biweek mismatch: {len(idw_biweeks - ovi_biweeks)} extra, "
            f"{len(ovi_biweeks - idw_biweeks)} missing"
        )

    def test_coordinates_match_centroids(self, idw_df, centroids_df):
        """IDW centroid coordinates should match the centroids file."""
        idw_coords = (
            idw_df.groupby("CD_SETOR")[["centroid_latitude", "centroid_longitude"]]
            .first()
            .sort_index()
        )
        cent_coords = (
            centroids_df.set_index("CD_SETOR")[["centroid_latitude", "centroid_longitude"]]
            .sort_index()
        )
        common = idw_coords.index.intersection(cent_coords.index)
        assert len(common) > 0, "No common sectors to compare"
        assert np.allclose(
            idw_coords.loc[common].values,
            cent_coords.loc[common].values,
            atol=1e-6,
        ), "Centroid coordinates in IDW file don't match centroids file"
