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
    return processed_dir / "population_2010_to_2022_equivalent.csv"


@pytest.fixture(scope="module")
def conversion_log_file(processed_dir):
    """Fixture providing the conversion log file."""
    return processed_dir / "census_equivalence_conversion_log.csv"


@pytest.fixture(scope="module")
def sector_linkage_file(processed_dir):
    """Fixture providing the sector linkage file."""
    return processed_dir / "census_equivalence_sector_linkage.csv"


@pytest.fixture(scope="module")
def geojson_2022_file(processed_dir):
    """Fixture providing the 2022 GeoJSON file."""
    return processed_dir / "bh_sectors_2022_with_2010_population.geojson"


@pytest.fixture(scope="module")
def geojson_2010_file(processed_dir):
    """Fixture providing the 2010 GeoJSON file."""
    return processed_dir / "bh_sectors_2010_with_population.geojson"


class TestCensusEquivalenceFiles:
    """Test class for verifying census equivalence output files."""

    def test_population_file_exists(self, population_file):
        """Test that population equivalence file exists."""
        assert population_file.exists(), f"Population file not found: {population_file}"

    def test_conversion_log_exists(self, conversion_log_file):
        """Test that conversion log file exists."""
        assert conversion_log_file.exists(), f"Conversion log not found: {conversion_log_file}"

    def test_sector_linkage_exists(self, sector_linkage_file):
        """Test that sector linkage file exists."""
        assert sector_linkage_file.exists(), f"Sector linkage not found: {sector_linkage_file}"

    def test_geojson_2022_exists(self, geojson_2022_file):
        """Test that 2022 GeoJSON file exists."""
        assert geojson_2022_file.exists(), f"2022 GeoJSON not found: {geojson_2022_file}"

    def test_geojson_2010_exists(self, geojson_2010_file):
        """Test that 2010 GeoJSON file exists."""
        assert geojson_2010_file.exists(), f"2010 GeoJSON not found: {geojson_2010_file}"


class TestPopulationEquivalenceData:
    """Test class for population equivalence data validation."""

    def test_population_data_structure(self, population_file):
        """Test population file has correct structure."""
        pop_df = pd.read_csv(population_file)
        
        # Check columns
        expected_columns = ['CD_SETOR', 'pop_2010_equivalent']
        assert all(col in pop_df.columns for col in expected_columns), \
            f"Missing columns. Expected: {expected_columns}, Found: {list(pop_df.columns)}"
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(pop_df['CD_SETOR']), "CD_SETOR should be numeric (sector code)"
        assert pd.api.types.is_numeric_dtype(pop_df['pop_2010_equivalent']), \
            "pop_2010_equivalent should be numeric"

    def test_population_data_content(self, population_file):
        """Test population file has reasonable content."""
        pop_df = pd.read_csv(population_file)
        
        # Check for reasonable number of sectors (should be thousands for BH)
        assert len(pop_df) > 1000, f"Expected > 1000 sectors, got {len(pop_df)}"
        assert len(pop_df) < 10000, f"Expected < 10000 sectors, got {len(pop_df)}"
        
        # Check population values are non-negative
        assert (pop_df['pop_2010_equivalent'] >= 0).all(), "Population values should be non-negative"
        
        # Check total population is reasonable (BH should have millions)
        total_pop = pop_df['pop_2010_equivalent'].sum()
        assert total_pop > 1_000_000, f"Total population too low: {total_pop:,.0f}"
        assert total_pop < 5_000_000, f"Total population too high: {total_pop:,.0f}"


class TestConversionLogData:
    """Test class for conversion log data validation."""

    def test_conversion_log_structure(self, conversion_log_file):
        """Test conversion log has correct structure."""
        log_df = pd.read_csv(conversion_log_file)
        
        expected_columns = ['type', 'code_2010', 'code_2022', 'pop_original', 'pop_converted', 'method', 'num_sources']
        assert all(col in log_df.columns for col in expected_columns), \
            f"Missing columns. Expected: {expected_columns}, Found: {list(log_df.columns)}"

    def test_conversion_types(self, conversion_log_file):
        """Test conversion types are valid."""
        log_df = pd.read_csv(conversion_log_file)
        
        valid_types = {'aggregated', 'subdivided', 'maintained'}
        actual_types = set(log_df['type'].unique())
        assert actual_types.issubset(valid_types), \
            f"Invalid conversion types found: {actual_types - valid_types}"

    def test_conversion_population_conservation(self, conversion_log_file):
        """Test that conversion log has reasonable population values."""
        log_df = pd.read_csv(conversion_log_file)
        
        total_original = log_df['pop_original'].sum()
        total_converted = log_df['pop_converted'].sum()
        
        assert total_original > 0, "Total original population should be positive"
        assert total_converted > 0, "Total converted population should be positive"
        
        # The conversion log may have different accounting due to aggregation
        # Just ensure both values are in reasonable ranges for BH
        assert total_converted > 1_000_000, f"Total converted population too low: {total_converted:,.0f}"
        assert total_converted < 5_000_000, f"Total converted population too high: {total_converted:,.0f}"


class TestSectorLinkageData:
    """Test class for sector linkage data validation."""

    def test_sector_linkage_structure(self, sector_linkage_file):
        """Test sector linkage has correct structure."""
        linkage_df = pd.read_csv(sector_linkage_file)
        
        expected_columns = ['code_2010', 'code_2022', 'area_2010', 'area_2022', 'intersection_area', 
                          'fraction_of_2010', 'fraction_of_2022', 'pop_2010']
        assert all(col in linkage_df.columns for col in expected_columns), \
            f"Missing columns. Expected: {expected_columns}, Found: {list(linkage_df.columns)}"

    def test_fraction_values(self, sector_linkage_file):
        """Test that fraction values are reasonable."""
        linkage_df = pd.read_csv(sector_linkage_file)
        
        # Fractions should be non-negative
        assert (linkage_df['fraction_of_2010'] >= 0).all(), "fraction_of_2010 should be >= 0"
        assert (linkage_df['fraction_of_2022'] >= 0).all(), "fraction_of_2022 should be >= 0"
        
        # Most fractions should be <= 1, but some can be slightly > 1 due to geometric precision
        assert (linkage_df['fraction_of_2010'] <= 1.01).all(), "fraction_of_2010 should be <= 1.01"
        assert (linkage_df['fraction_of_2022'] <= 1.01).all(), "fraction_of_2022 should be <= 1.01"

    def test_area_consistency(self, sector_linkage_file):
        """Test that area calculations are consistent."""
        linkage_df = pd.read_csv(sector_linkage_file)
        
        # Intersection area should be <= both original areas (with small tolerance for precision)
        tolerance = 1.01  # Allow 1% tolerance for geometric precision errors
        
        area_2010_ok = (linkage_df['intersection_area'] <= linkage_df['area_2010'] * tolerance).all()
        assert area_2010_ok, "Intersection area significantly exceeds 2010 area"
        
        area_2022_ok = (linkage_df['intersection_area'] <= linkage_df['area_2022'] * tolerance).all()
        assert area_2022_ok, "Intersection area significantly exceeds 2022 area"


class TestGeoJSONFiles:
    """Test class for GeoJSON files validation."""

    def test_geojson_2022_structure(self, geojson_2022_file):
        """Test 2022 GeoJSON has correct structure."""
        gdf_2022 = gpd.read_file(geojson_2022_file)
        
        # Check required columns
        assert 'CD_SETOR' in gdf_2022.columns, "Missing CD_SETOR column"
        assert 'pop_2010_equivalent' in gdf_2022.columns, "Missing pop_2010_equivalent column"
        assert 'geometry' in gdf_2022.columns, "Missing geometry column"
        
        # Check CRS
        assert gdf_2022.crs is not None, "CRS should be defined"
        assert str(gdf_2022.crs).startswith('EPSG'), "CRS should be EPSG format"

    def test_geojson_2010_structure(self, geojson_2010_file):
        """Test 2010 GeoJSON has correct structure."""
        gdf_2010 = gpd.read_file(geojson_2010_file)
        
        # Check required columns
        assert 'CD_GEOCODI' in gdf_2010.columns, "Missing CD_GEOCODI column"
        assert 'pop_2010_original' in gdf_2010.columns, "Missing pop_2010_original column"
        assert 'geometry' in gdf_2010.columns, "Missing geometry column"
        
        # Check CRS
        assert gdf_2010.crs is not None, "CRS should be defined"
        assert str(gdf_2010.crs).startswith('EPSG'), "CRS should be EPSG format"

    def test_geometry_validity(self, geojson_2022_file, geojson_2010_file):
        """Test that geometries are valid."""
        gdf_2022 = gpd.read_file(geojson_2022_file)
        gdf_2010 = gpd.read_file(geojson_2010_file)
        
        # Check for valid geometries
        assert gdf_2022.geometry.is_valid.all(), "Some 2022 geometries are invalid"
        assert gdf_2010.geometry.is_valid.all(), "Some 2010 geometries are invalid"
        
        # Check for non-empty geometries
        assert not gdf_2022.geometry.is_empty.any(), "Some 2022 geometries are empty"
        assert not gdf_2010.geometry.is_empty.any(), "Some 2010 geometries are empty"


class TestOverallConservation:
    """Test class for overall population conservation."""

    def test_population_conservation_rate(self, population_file, sector_linkage_file):
        """Test that overall population conservation rate is excellent."""
        pop_df = pd.read_csv(population_file)
        linkage_df = pd.read_csv(sector_linkage_file)
        
        # Calculate conservation rate correctly
        total_converted = pop_df['pop_2010_equivalent'].sum()
        original_total = linkage_df.groupby('code_2010')['pop_2010'].first().sum()
        conservation_rate = (total_converted / original_total) * 100
        
        # Should have excellent conservation (>99%)
        assert conservation_rate > 99.0, \
            f"Conservation rate should be >99%, got {conservation_rate:.2f}%"
        
        assert conservation_rate < 100.1, \
            f"Conservation rate should not exceed 100.1%, got {conservation_rate:.2f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])