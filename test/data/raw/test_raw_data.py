"""
Tests for validating the raw data files.
These tests ensure that the raw data files are correctly formatted,
contain the expected columns, and have no critical data quality issues.

Files tested:
- Dengue2007_2025.csv
- MasterDataExtend062025.csv
- CENTRO_SAUDE_new.csv
"""


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

diagnosis_file = params["all"]["paths"]["test"]["data"]["raw"]["diagnosis_raw_data"]
with open(diagnosis_file, "w", encoding="utf-8") as f:
    pass

class TestRawData:
    """Test cases for validating the final processed data files."""
    
    @pytest.fixture(scope="class")
    def data_paths(self):
        """Fixture providing paths to processed data files."""
        return {
            'dengue': params["all"]["paths"]["data"]["raw"]["dengue_csv"],
            'ovitraps': params["all"]["paths"]["data"]["raw"]["ovitraps_csv"],
            'health_centers': params["all"]["paths"]["data"]["raw"]["health_centers_csv"],
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

    def test_files_exist(self, data_paths):
        """Test that all expected processed data files exist."""
        for name, path in data_paths.items():
            assert os.path.exists(path), f"{name} data file should exist at {path}"
            assert os.path.getsize(path) > 0, f"{name} data file should not be empty"

    # Health Centers Data Tests
    def test_health_centers_structure(self, health_centers_data):
        """Test health centers data structure and content."""
        # Check required columns
        required_columns = ['CENTRO DE SAÚDE', 'LATITUDE', 'LONGITUDE']
        for col in required_columns:
            assert col in health_centers_data.columns, f"Missing column: {col}"
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(health_centers_data['LATITUDE']), "LATITUDE should be numeric"
        assert pd.api.types.is_numeric_dtype(health_centers_data['LONGITUDE']), "LONGITUDE should be numeric"
        
        # Check for missing values in critical columns
        assert health_centers_data['CENTRO DE SAÚDE'].notnull().all(), "health_center should not have null values"
        assert health_centers_data['LATITUDE'].notnull().all(), "LATITUDE should not have null values"
        assert health_centers_data['LONGITUDE'].notnull().all(), "LONGITUDE should not have null values"
        
        # Check coordinate ranges (assuming Brazilian coordinates)
        assert health_centers_data['LATITUDE'].between(-35, 5).all(), "LATITUDE should be in valid range for Brazil"
        assert health_centers_data['LONGITUDE'].between(-75, -30).all(), "LONGITUDE should be in valid range for Brazil"

    def test_health_centers_data_quality(self, health_centers_data):
        """Test health centers data quality."""
        # Check for duplicate health centers
        duplicates = health_centers_data['CENTRO DE SAÚDE'].duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate health centers"
        
        # Check that health center names are properly formatted (no special characters from corruption)
        problem_chars = ['Ã', 'Ç', 'õ', 'Ê']
        for char in problem_chars:
            corrupt_names = health_centers_data['CENTRO DE SAÚDE'].str.contains(char, na=False).sum()
            assert corrupt_names == 0, f"Found {corrupt_names} health center names with character '{char}'"

    # Dengue Data Tests
    def test_dengue_data_structure(self, dengue_data):
        """Test dengue data structure and required columns."""
        required_columns = [
            'SemEpi', 'Ano_Caso', 'anoCepid', 'dt_notific', 
            'Dengue',
        ]
        
        for col in required_columns:
            assert col in dengue_data.columns, f"Missing column: {col}"
        
     
    def test_dengue_data_types(self, dengue_data):
        """Test dengue data types."""
        # Numeric columns
        numeric_columns = [ 'SemEpi', 'Ano_Caso']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(dengue_data[col]), f"{col} should be numeric"
        
        # String columns
        string_columns = ['Dengue']
        for col in string_columns:
            assert pd.api.types.is_string_dtype(dengue_data[col].dropna()), f"{col} should be string"

        # Check date columns can be parsed
        try:
            pd.to_datetime(dengue_data['dt_notific'])
        except:
            pytest.fail("Date columns should be parseable as datetime")


    def test_dengue_data_quality(self, dengue_data):
        """Test dengue data quality."""
        # Check for missing values in critical columns
        critical_columns = ['SemEpi', 'Ano_Caso', 'dt_notific']
        for col in critical_columns:
            null_count = dengue_data[col].isnull().sum()
            assert null_count == 0, f"Found {null_count} null values in {col}"
        
        # Check epidemic week ranges
        assert(dengue_data['SemEpi'] % 100).between(1, 53).all(), "SemEpi should be between 1 and 53"
        
        # Check year ranges (assuming reasonable range)
        assert dengue_data['Ano_Caso'].between(2000, 2030).all(), "ano should be in reasonable range"

        # Check for duplicates
        try:
            assert dengue_data.duplicated().sum() == 0, "Dengue data should not have duplicates"
        except AssertionError:
            print("Warning: Dengue data has duplicates, consider reviewing data cleaning steps.")

    # Ovitraps Data Tests
    def test_ovitraps_data_structure(self, ovitraps_data):
        """Test ovitraps data structure and required columns."""
        required_columns = [
            'semepi', 'dtcol', 'dtinstal', 'narmad', 'novos',
            'anoepid',
        ]
        
        for col in required_columns:
            assert col in ovitraps_data.columns, f"Missing column: {col}"

    def test_ovitraps_data_types(self, ovitraps_data):
        """Test ovitraps data types."""
        # Numeric columns
        numeric_columns = ['semepi', 'narmad', 'novos']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(ovitraps_data[col]), f"{col} should be numeric"
        
        # Check date columns can be parsed
        try:
            pd.to_datetime(ovitraps_data['dtcol'])
            pd.to_datetime(ovitraps_data['dtinstal'])
        except:
            pytest.fail("Date columns should be parseable as datetime")

    def test_ovitraps_data_quality(self, ovitraps_data):
        """Test ovitraps data quality."""
        # Check if 'nplaca' is unique
        assert ovitraps_data['nplaca'].is_unique, "'nplaca' should be unique"

        # Check if 'narmad','dtinstal' is unique
        assert ovitraps_data[['narmad','dtinstal']].drop_duplicates().shape[0] == ovitraps_data.shape[0], "'narmad' and 'dtinstal' combination should be unique"
        
        # Check for missing values in critical columns
        critical_columns = ['dtcol', 'dtinstal', 'narmad', 'novos']
        for col in critical_columns:
            null_count = ovitraps_data[col].isnull().sum()
            try:
                assert null_count == 0, f"Found {null_count} null values in {col}"
            except AssertionError:
                with open(diagnosis_file, "a") as f:
                    f.write(f"{null_count} Null values found in {col}:\n\n")
                    null_rows = ovitraps_data[ovitraps_data[col].isnull()]
                    f.write(null_rows[['narmad', 'novos','dtinstal', 'dtcol']].to_string(index=False) +  "\n -------------------------------- \n")
        
        # Check that all novos values are non-negative
        assert (ovitraps_data['novos'].dropna() >= 0).all(), "novos should be non-negative"

        ovitraps_data['days_expo'] = (pd.to_datetime(ovitraps_data['dtcol']) - pd.to_datetime(ovitraps_data['dtinstal'])).dt.days

        # Check that days_expo is reasonable (not negative, not too large)
        try:
            assert (ovitraps_data['days_expo'] > 0).all(), "days_expo should be positive"
        except AssertionError:
            with open(diagnosis_file, "a") as f:
                negative_days = ovitraps_data[ovitraps_data['days_expo'] <= 0]
                f.write(f"{negative_days.shape[0]} Negative or zero values found in 'days_expo':\n\n")
                f.write(negative_days[['narmad', 'dtinstal', 'dtcol', 'days_expo', 'novos']].to_string(index=False) + "\n -------------------------------- \n")
        try:
            assert (ovitraps_data['days_expo'] <= 30).all(), "days_expo should be <= 30 days (filter should remove longer expositions)"
        except AssertionError:
            with open(diagnosis_file, "a") as f:
                large_days = ovitraps_data[ovitraps_data['days_expo'] > 30]
                f.write(f"{large_days.shape[0]} Too large values found in 'days_expo':\n\n")
                f.write(large_days[['narmad', 'dtinstal', 'dtcol', 'days_expo', 'novos']].to_string(index=False) +  "\n -------------------------------- \n")

        # Check for duplicates
        assert ovitraps_data.duplicated().sum() == 0, "Ovitraps data should not have duplicates"
        

    @pytest.mark.slow
    def test_ovitrap_overlapping_installations(self, ovitraps_data):
        """Comprehensive data integrity test (marked as slow)."""
        
        # Check that installation date is before collection date
        assert ((ovitraps_data['dtinstal'] < ovitraps_data['dtcol']).all(),
            "Installation date should be before collection date for all records")
        
        # For ovitraps: same trap shouldn't have overlapping installation periods
        # (this is complex but important)
        overlapping_traps = project_utils.get_overlapped_samples(ovitraps_data)
        try:
            assert len(overlapping_traps) == 0, f"Traps have overlapping periods > 1 day"
        except AssertionError:
            with open(diagnosis_file, "a") as f:
                f.write(f"{len(overlapping_traps)} Overlapping installation periods found \n\n")
                final_list = [
                    ovitraps_data[ovitraps_data['nplaca'].isin(pair)][
                        ['narmad', 'nplaca', 'dtinstal', 'dtcol', 'novos']
                    ].to_string(index=False)
                    for pair in overlapping_traps
                ]

                f.write("\n".join(final_list) + "\n")
                        
