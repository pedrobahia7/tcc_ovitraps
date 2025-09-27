import re
import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime
import yaml
import sys
sys.path.append("utils")
import project_utils

params = yaml.safe_load(open("params.yaml"))

def epidemic_date_valid(s):
    m = re.match(r"^20(\d{2})_(\d{2})W(\d{1,2})$", s)
    if not m:
        return False
    y1, y2, week = map(int, m.groups())
    return y2 == y1 + 1 and 1 <= week <= 53

class TestProcessedData:
    """Test cases for validating the final processed data files."""
    
    @pytest.fixture(scope="class")
    def data_paths(self):
        """Fixture providing paths to processed data files."""
        return {
            'dengue': params["all"]["paths"]["data"]["processed"]["dengue"],
            'ovitraps': params["all"]["paths"]["data"]["processed"]["ovitraps"],
            'health_centers': params["all"]["paths"]["data"]["processed"]["health_centers"],
            'daily_ovitraps': params["all"]["paths"]["data"]["processed"]["daily_ovitraps"]
        }
    
    @pytest.fixture(scope="class")
    def dengue_data(self, data_paths):
        """Load dengue processed data."""
        if os.path.exists(data_paths['dengue']):
            print('Loading dengue data from:', data_paths['dengue'])
            return pd.read_csv(data_paths['dengue'])
        else:
            pytest.skip(f"Dengue data file not found: {data_paths['dengue']}")
    
    @pytest.fixture(scope="class")
    def ovitraps_data(self, data_paths):
        """Load ovitraps processed data."""
        if os.path.exists(data_paths['ovitraps']):
            print('Loading ovitraps data from:', data_paths['ovitraps'])
            return pd.read_csv(data_paths['ovitraps'])
        else:
            pytest.skip(f"Ovitraps data file not found: {data_paths['ovitraps']}")
    
    @pytest.fixture(scope="class")
    def health_centers_data(self, data_paths):
        """Load health centers processed data."""
        if os.path.exists(data_paths['health_centers']):
            print('Loading health centers data from:', data_paths['health_centers'])
            return pd.read_csv(data_paths['health_centers'])
        else:
            pytest.skip(f"Health centers data file not found: {data_paths['health_centers']}")

    @pytest.fixture(scope="class")
    def daily_ovitraps_data(self, data_paths):
        """Load daily ovitraps processed data."""
        if os.path.exists(data_paths['daily_ovitraps']):
            print('Loading daily ovitraps data from:', data_paths['daily_ovitraps'])
            return pd.read_csv(data_paths['daily_ovitraps'])
        else:
            pytest.skip(f"Daily ovitraps data file not found: {data_paths['daily_ovitraps']}")

    def test_files_exist(self, data_paths):
        """Test that all expected processed data files exist."""
        for name, path in data_paths.items():
            assert os.path.exists(path), f"{name} data file should exist at {path}"
            assert os.path.getsize(path) > 0, f"{name} data file should not be empty"

    # Health Centers Data Tests
    def test_health_centers_structure(self, health_centers_data):
        """Test health centers data structure and content."""
        # Check required columns
        required_columns = ['health_center', 'latitude', 'longitude']
        for col in required_columns:
            assert col in health_centers_data.columns, f"Missing column: {col}"
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(health_centers_data['latitude']), "latitude should be numeric"
        assert pd.api.types.is_numeric_dtype(health_centers_data['longitude']), "longitude should be numeric"
        
        # Check for missing values in critical columns
        assert health_centers_data['health_center'].notnull().all(), "health_center should not have null values"
        assert health_centers_data['latitude'].notnull().all(), "latitude should not have null values"
        assert health_centers_data['longitude'].notnull().all(), "longitude should not have null values"
        
        # Check coordinate ranges (assuming Brazilian coordinates)
        assert health_centers_data['latitude'].between(-35, 5).all(), "latitude should be in valid range for Brazil"
        assert health_centers_data['longitude'].between(-75, -30).all(), "longitude should be in valid range for Brazil"

    def test_health_centers_data_quality(self, health_centers_data):
        """Test health centers data quality."""
        # Check for duplicate health centers
        duplicates = health_centers_data['health_center'].duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate health centers"
        
        # Check that health center names are properly formatted (no special characters from corruption)
        problem_chars = ['Ã', 'Ç', 'õ', 'Ê']
        for char in problem_chars:
            corrupt_names = health_centers_data['health_center'].str.contains(char, na=False).sum()
            assert corrupt_names == 0, f"Found {corrupt_names} health center names with character '{char}'"

    # Dengue Data Tests
    def test_dengue_data_structure(self, dengue_data):
        """Test dengue data structure and required columns."""
        required_columns = [
            'semana', 'ano', 'anoepid', 'semepid', 'dt_notific', 
            'Dengue', 'closest_health_center', 'epidemic_date'
        ]
        
        for col in required_columns:
            assert col in dengue_data.columns, f"Missing column: {col}"
        
        # Check that only confirmed dengue cases remain
        assert (dengue_data['Dengue'] != 'N').all(), "Should only contain confirmed dengue cases"

    def test_dengue_data_types(self, dengue_data):
        """Test dengue data types."""
        # Numeric columns
        numeric_columns = ['semana', 'ano', 'semepid']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(dengue_data[col]), f"{col} should be numeric"
        
        # String columns
        string_columns = ['Dengue', 'closest_health_center', 'epidemic_date','anoepid']
        for col in string_columns:
            assert pd.api.types.is_string_dtype(dengue_data[col].dropna()), f"{col} should be string"

        # Check date columns can be parsed
        try:
            pd.to_datetime(dengue_data['dt_notific'])
        except:
            pytest.fail("Date columns should be parseable as datetime")

        # Check epidemic_date format
        dengue_data["valid"] = dengue_data["epidemic_date"].apply(epidemic_date_valid)
        assert dengue_data["valid"].all(), "epidemic_date should be in format YYYY_YYWww"

    def test_dengue_data_quality(self, dengue_data):
        """Test dengue data quality."""
        # Check for missing values in critical columns
        critical_columns = ['semana', 'ano', 'anoepid', 'semepid', 'dt_notific']
        for col in critical_columns:
            null_count = dengue_data[col].isnull().sum()
            assert null_count == 0, f"Found {null_count} null values in {col}"
        
        # Check epidemic week ranges
        assert dengue_data['semana'].between(1, 53).all(), "semana should be between 1 and 53"
        assert dengue_data['semepid'].between(1, 53).all(), "semepid should be between 1 and 53"
        
        # Check year ranges (assuming reasonable range)
        assert dengue_data['ano'].between(2000, 2030).all(), "ano should be in reasonable range"

        # Check for duplicates
        assert dengue_data.duplicated().sum() == 0, "Dengue data should not have duplicates"

    # Ovitraps Data Tests
    def test_ovitraps_data_structure(self, ovitraps_data):
        """Test ovitraps data structure and required columns."""
        required_columns = [
            'semana', 'dt_col', 'dt_instal', 'narmad', 'novos',
            'semepid', 'anoepid', 'days_expo', 'eggs_per_day',
            'closest_health_center', 'epidemic_date'
        ]
        
        for col in required_columns:
            assert col in ovitraps_data.columns, f"Missing column: {col}"

    def test_ovitraps_data_types(self, ovitraps_data):
        """Test ovitraps data types."""
        # Numeric columns
        numeric_columns = ['semana', 'narmad', 'novos', 'semepid', 'days_expo', 'eggs_per_day']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(ovitraps_data[col]), f"{col} should be numeric"
        
        # Check date columns can be parsed
        pd.to_datetime(ovitraps_data['dt_col'], errors="raise")
        pd.to_datetime(ovitraps_data['dt_instal'], errors="raise")

        # Check epidemic_date format
        ovitraps_data["valid"] = ovitraps_data["epidemic_date"].apply(epidemic_date_valid)
        assert ovitraps_data["valid"].all(), "epidemic_date should be in format YYYY_YYWww"

    def test_ovitraps_data_quality(self, ovitraps_data):
        """Test ovitraps data quality."""
        # Check for missing values in critical columns
        critical_columns = ['dt_col', 'dt_instal', 'narmad', 'novos']
        for col in critical_columns:
            null_count = ovitraps_data[col].isnull().sum()
            assert null_count == 0, f"Found {null_count} null values in {col}"
        
        # Check that all novos values are non-negative
        assert (ovitraps_data['novos'] >= 0).all(), "novos should be non-negative"
        
        # Check that days_expo is reasonable (not negative, not too large)
        assert (ovitraps_data['days_expo'] >= 4).all(), "days_expo should be greater than 4 days (filter should remove shorter expositions)"
        assert (ovitraps_data['days_expo'] <= 21).all(), "days_expo should be <= 21 days (filter should remove longer expositions)"

        # Check for duplicates
        assert ovitraps_data[critical_columns].duplicated().sum() == 0, "Ovitraps data should not have duplicates"

        # Check if manually corrected records are present
        corrected_record = ovitraps_data[
            (ovitraps_data["narmad"] == 906071) & (ovitraps_data["dt_col"] == "2022-08-18")
        ]
        assert not corrected_record.empty, "Corrected record should be present"
        assert corrected_record.iloc[0]['novos'] == 50, "Corrected record should have novos == 50"

    def test_ovitraps_date_logic(self, ovitraps_data):
        """Test ovitraps date logic and calculations."""
        dt_col = pd.to_datetime(ovitraps_data['dt_col'])
        dt_instal = pd.to_datetime(ovitraps_data['dt_instal'])

        # Check that days_expo calculation is correct
        calculated_days = (dt_col - dt_instal).dt.days
        assert (ovitraps_data['days_expo'] == calculated_days).all(), "days_expo calculation should be correct"
        
        # Check eggs_per_day calculation (where days_expo > 0)
        expected_eggs_per_day = ovitraps_data.loc[:, 'novos'] / ovitraps_data.loc[:, 'days_expo']
        actual_eggs_per_day = ovitraps_data.loc[:, 'eggs_per_day']
        assert ((np.isclose(actual_eggs_per_day.values, expected_eggs_per_day.values, atol=0.01, equal_nan=True)).all(),
                 "eggs_per_day calculation should be correct")
            

    @pytest.mark.slow
    def test_ovitrap_overlapping_installations(self, ovitraps_data):
        """Comprehensive data integrity test (marked as slow)."""
        
        # Check that installation date is before collection date
        assert ((ovitraps_data['dt_instal'] < ovitraps_data['dt_col']).all(),
            "Installation date should be before collection date for all records")
        
        # For ovitraps: same trap shouldn't have overlapping installation periods
        # (this is complex but important)
        
        overlapped_traps = project_utils.get_overlapped_samples(ovitraps_data, processed_name=True)
        assert len(overlapped_traps) == 0, f"Trap has overlapping periods"

    # Cross-dataset Tests
    def test_health_center_consistency(self, dengue_data, ovitraps_data, health_centers_data):
        """Test that health center assignments are consistent across datasets."""
        # Get unique health centers from each dataset
        dengue_centers = set(dengue_data['closest_health_center'].dropna().unique())
        ovitraps_centers = set(ovitraps_data['closest_health_center'].dropna().unique())
        available_centers = set(health_centers_data['health_center'].unique())
        
        # Check that all assigned health centers exist in the health centers dataset
        invalid_dengue = dengue_centers - available_centers
        invalid_ovitraps = ovitraps_centers - available_centers
        
        assert len(invalid_dengue) == 0, f"Dengue data has invalid health centers: {invalid_dengue}"
        assert len(invalid_ovitraps) == 0, f"Ovitraps data has invalid health centers: {invalid_ovitraps}"
