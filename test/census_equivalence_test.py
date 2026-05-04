"""
Pytest unit tests for census equivalence pipeline stage
"""

import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path


@pytest.fixture(scope="module")
def processed_dir():
    """Fixture providing the processed data directory path."""
    return Path("data/processed")


@pytest.fixture(scope="module")
def population_file(processed_dir):
    """Fixture providing the population equivalence file."""
    return processed_dir / "population_data.csv"


@pytest.fixture(scope="module")
def sector_linkage_file(processed_dir):
    """Fixture providing the sector linkage file."""
    return processed_dir / "sector_equivalence_2010_to_2022.csv"


@pytest.fixture(scope="module")
def geojson_2022_file(processed_dir):
    """Fixture providing the 2022 GeoJSON file."""
    return processed_dir / "bh_sectors_2022_with_populations.geojson"


class TestCensusEquivalenceFiles:
    """Test class for verifying census equivalence output files."""

    def test_population_file_exists(self, population_file):
        """Test that population equivalence file exists."""
        assert population_file.exists(), (
            f"Population file not found: {population_file}"
        )

    def test_sector_linkage_exists(self, sector_linkage_file):
        """Test that sector linkage file exists."""
        assert sector_linkage_file.exists(), (
            f"Sector linkage not found: {sector_linkage_file}"
        )

    def test_geojson_2022_exists(self, geojson_2022_file):
        """Test that 2022 GeoJSON file exists."""
        assert geojson_2022_file.exists(), (
            f"2022 GeoJSON not found: {geojson_2022_file}"
        )


class TestPopulationEquivalenceData:
    """Test class for population equivalence data validation."""

    def test_population_data_structure(self, population_file):
        """Test population file has correct structure."""
        pop_df = pd.read_csv(population_file)

        # Check columns
        expected_columns = [
            "sector_id",
            "population_2010",
            "population_2022",
        ]
        assert all(col in pop_df.columns for col in expected_columns), (
            f"Missing columns. Expected: {expected_columns}, Found: {list(pop_df.columns)}"
        )

        # Check data types
        assert pd.api.types.is_numeric_dtype(pop_df["sector_id"]), (
            "sector_id should be numeric (sector code)"
        )
        assert pd.api.types.is_numeric_dtype(pop_df["population_2010"]), (
            "population_2010 should be numeric"
        )
        assert pd.api.types.is_numeric_dtype(pop_df["population_2022"]), (
            "population_2022 should be numeric"
        )

    def test_population_data_content(self, population_file):
        """Test population file has reasonable content."""
        pop_df = pd.read_csv(population_file)

        # Check for reasonable number of sectors (should be thousands for BH)
        assert len(pop_df) > 1000, (
            f"Expected > 1000 sectors, got {len(pop_df)}"
        )
        assert len(pop_df) < 10000, (
            f"Expected < 10000 sectors, got {len(pop_df)}"
        )

        # Check population values are non-negative
        assert (pop_df["population_2010"] >= 0).all(), (
            "Population 2010 values should be non-negative"
        )
        assert (pop_df["population_2022"] >= 0).all(), (
            "Population 2022 values should be non-negative"
        )

        # Check total population is reasonable (BH should have millions)
        total_2010 = pop_df["population_2010"].sum()
        total_2022 = pop_df["population_2022"].sum()
        assert total_2010 > 1_000_000, (
            f"Total 2010 population too low: {total_2010:,.0f}"
        )
        assert total_2010 < 5_000_000, (
            f"Total 2010 population too high: {total_2010:,.0f}"
        )
        assert total_2022 > 1_000_000, (
            f"Total 2022 population too low: {total_2022:,.0f}"
        )
        assert total_2022 < 5_000_000, (
            f"Total 2022 population too high: {total_2022:,.0f}"
        )


class TestSectorLinkageData:
    """Test class for sector linkage data validation."""

    def test_sector_linkage_structure(self, sector_linkage_file):
        """Test sector linkage has correct structure."""
        linkage_df = pd.read_csv(sector_linkage_file)

        expected_columns = [
            "code_2010",
            "code_2022",
            "fraction_of_2010",
            "fraction_of_2022",
            "intersection_area",
        ]
        assert all(
            col in linkage_df.columns for col in expected_columns
        ), (
            f"Missing columns. Expected: {expected_columns}, Found: {list(linkage_df.columns)}"
        )

    def test_fraction_values(self, sector_linkage_file):
        """Test that fraction values are reasonable."""
        linkage_df = pd.read_csv(sector_linkage_file)

        # Fractions should be non-negative
        assert (linkage_df["fraction_of_2010"] >= 0).all(), (
            "fraction_of_2010 should be >= 0"
        )
        assert (linkage_df["fraction_of_2022"] >= 0).all(), (
            "fraction_of_2022 should be >= 0"
        )

        # Most fractions should be <= 1, but some can be slightly > 1 due to geometric precision
        assert (linkage_df["fraction_of_2010"] <= 1.01).all(), (
            "fraction_of_2010 should be <= 1.01"
        )
        assert (linkage_df["fraction_of_2022"] <= 1.01).all(), (
            "fraction_of_2022 should be <= 1.01"
        )


class TestGeoJSONFiles:
    """Test class for GeoJSON files validation."""

    def test_geojson_2022_structure(self, geojson_2022_file):
        """Test 2022 GeoJSON has correct structure."""
        gdf_2022 = gpd.read_file(geojson_2022_file)

        # Check required columns
        assert "CD_SETOR" in gdf_2022.columns, "Missing CD_SETOR column"
        assert "pop_2010" in gdf_2022.columns, "Missing pop_2010 column"
        assert "pop_2022" in gdf_2022.columns, "Missing pop_2022 column"
        assert "geometry" in gdf_2022.columns, "Missing geometry column"

        # Check CRS
        assert gdf_2022.crs is not None, "CRS should be defined"
        assert str(gdf_2022.crs).startswith("EPSG"), (
            "CRS should be EPSG format"
        )

    def test_geometry_validity(self, geojson_2022_file):
        """Test that geometries are valid."""
        gdf_2022 = gpd.read_file(geojson_2022_file)

        # Check for valid geometries
        assert gdf_2022.geometry.is_valid.all(), (
            "Some 2022 geometries are invalid"
        )

        # Check for non-empty geometries
        assert not gdf_2022.geometry.is_empty.any(), (
            "Some 2022 geometries are empty"
        )


class TestOverallConservation:
    """Test class for overall population conservation and consistency."""

    def test_population_csv_geojson_consistency(
        self, population_file, geojson_2022_file
    ):
        """Test that population CSV and GeoJSON totals are consistent."""
        pop_df = pd.read_csv(population_file)
        gdf_2022 = gpd.read_file(geojson_2022_file)

        total_csv_2010 = pop_df["population_2010"].sum()
        total_geojson_2010 = gdf_2022["pop_2010"].sum()
        total_csv_2022 = pop_df["population_2022"].sum()
        total_geojson_2022 = gdf_2022["pop_2022"].sum()

        # Allow small tolerance for rounding differences
        tolerance = 0.01

        rate_2010 = (
            abs(total_csv_2010 - total_geojson_2010) / total_csv_2010 * 100
        )
        rate_2022 = (
            abs(total_csv_2022 - total_geojson_2022) / total_csv_2022 * 100
        )

        assert rate_2010 < tolerance, (
            f"2010 population mismatch between CSV and GeoJSON: {rate_2010:.4f}%"
        )
        assert rate_2022 < tolerance, (
            f"2022 population mismatch between CSV and GeoJSON: {rate_2022:.4f}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
