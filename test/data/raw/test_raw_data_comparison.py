"""
Tests to check consistency between different raw data files

These tests ensure that the raw data files have the same id and format than 
files of the same type from previous years.

Files tested:
- dengue data
- ovitraps data
- health center data

"""

import re
import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime

diagnosis_file = os.path.join(os.path.dirname(__file__), "diagnosis_raw_comparison.txt")
with open(diagnosis_file, "w", encoding="utf-8") as f:
    pass


class TestRawDataComparison:
    @pytest.fixture(scope="class")
    def data_paths(self):
        """Fixture providing paths to processed data files."""
        return {
            'dengue': 'data/raw/Dengue2007_2025.csv',
            'ovitraps': 'data/raw/MasterDataExtend062025.csv',
            'health_centers': 'data/raw/CENTRO_SAUDE_new.csv',
            'old_ovitraps': ["data/final_data.csv"]
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
    def old_ovitraps_data(self, data_paths):
        """Load old ovitraps processed data."""
        old_ovitraps_data = {}
        for path in data_paths['old_ovitraps']:
            if os.path.exists(path):
                print('Loading old ovitraps data from:', path)
                old_ovitraps_data[path] = pd.read_csv(path)
            else:
                pytest.skip(f"Old ovitraps data file not found: {path}")
        if old_ovitraps_data:
            return old_ovitraps_data
        else:
            pytest.skip("No old ovitraps data files found.")

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
            if isinstance(path, list):
                for p in path:
                    assert os.path.exists(p), f"{name} data file should exist at {p}"
                    assert os.path.getsize(p) > 0, f"{name} data file should not be empty"
            elif isinstance(path, str):
                assert os.path.exists(path), f"{name} data file should exist at {path}"
                assert os.path.getsize(path) > 0, f"{name} data file should not be empty"
            else:
                raise ValueError(f"Invalid path type for {name}: {type(path)}")
            

    def test_nplaca_consistency(self, ovitraps_data, old_ovitraps_data):
        """
        Test consistency of 'nplaca' between old and new ovitraps files.
        If an 'nplaca' value is present in both files, it must refer to the same sample.
        """
        key_fields = ["dtinstal", "dtcol", "narmad", "novos"]

        for old_df_name, old_df in old_ovitraps_data.items():
            # Ensure both DataFrames have 'nplaca'
            assert "nplaca" in ovitraps_data.columns, "'nplaca' column missing in new ovitraps data"
            assert "nplaca" in old_df.columns, "'nplaca' column missing in old ovitraps data"

            # Keep only relevant columns
            new = ovitraps_data[["nplaca"] + key_fields].dropna(subset=["nplaca"])
            old = old_df[["nplaca"] + key_fields].dropna(subset=["nplaca"])

            # Inner join on nplaca
            merged = new.merge(old, on="nplaca", suffixes=("_new", "_old"))

            # Find mismatches in any key field
            mismatches = merged[
                (merged[[f + "_new" for f in key_fields]].astype(str).values !=
                merged[[f + "_old" for f in key_fields]].astype(str).values).any(axis=1)
            ]

            try:
                assert mismatches.empty, f"Inconsistent 'nplaca' sample(s)"
            except AssertionError:
                with open(diagnosis_file, "a", encoding="utf-8") as f:
                    f.write(f"{mismatches.shape[0]} Inconsistent 'nplaca' sample(s) found in {old_df_name}:\n")
                    f.write(mismatches.to_string(index=False))
                    f.write("\n" + "-"*80 + "\n\n")


    def test_real_identificator_consistency(self, ovitraps_data, old_ovitraps_data):
        """
        Test consistency of ('narmad', 'dtcol', 'dtinstal') between old and new ovitraps files.
        If the tuple value is present in both files, it must have the same 'novos'.
        """
        key_fields = ["dtinstal", "dtcol", "narmad"]

        for old_df_name, old_df in old_ovitraps_data.items():
            # Ensure both DataFrames have 'nplaca'
            assert all([field in ovitraps_data.columns for field in key_fields]), "some column missing in new ovitraps data"
            assert all([field in old_df.columns for field in key_fields]), "some column missing in old ovitraps data"

            # Keep only relevant columns
            new = ovitraps_data[key_fields + ["novos"]].dropna(subset=key_fields)
            old = old_df[key_fields + ["novos"]].dropna(subset=key_fields)

            # Inner join on nplaca
            merged = new.merge(old, on=key_fields, suffixes=("_new", "_old"))

            # Find mismatches in any key field
            mismatches = merged[
                (merged[["novos" + "_new"]].astype(str).values !=
                merged[["novos" + "_old"]].astype(str).values).any(axis=1)
            ]

            try:
                assert mismatches.empty, f"Inconsistent 'novos' sample(s)"
            except AssertionError:
                with open(diagnosis_file, "a", encoding="utf-8") as f:
                    f.write(f"{mismatches.shape[0]} Inconsistent 'novos' sample(s) found in {old_df_name}:\n")
                    f.write(mismatches.to_string(index=False))
                    f.write("\n" + "-"*80 + "\n\n")

    def test_real_identificator_presence(self, ovitraps_data, old_ovitraps_data):
        """
        Test existence of ('narmad', 'dtcol', 'dtinstal') in both old and new ovitraps files.
        """
        key_fields = ["dtinstal", "dtcol", "narmad"]

        for old_df_name, old_df in old_ovitraps_data.items():
            # Ensure both DataFrames have 'nplaca'
            assert all([field in ovitraps_data.columns for field in key_fields]), "some column missing in new ovitraps data"
            assert all([field in old_df.columns for field in key_fields]), "some column missing in old ovitraps data"

            # Keep only relevant columns
            new = set(ovitraps_data[key_fields])
            old = set(old_df[key_fields])
            try:
                assert new - old == 0, f"Sample(s) in new ovitraps not present in old ovitraps"
            except AssertionError:
                with open(diagnosis_file, "a", encoding="utf-8") as f:
                    f.write(f"{len(new - old)} sample(s) found in new ovitraps not present in {old_df_name}:\n")
                    f.write("\n".join([str(s) for s in new - old]) + "\n")
                    f.write("\n" + "-"*80 + "\n\n")
            try:
                assert old - new == 0, f"Sample(s) in old ovitraps not present in new ovitraps"
            except AssertionError:
                with open(diagnosis_file, "a", encoding="utf-8") as f:
                    f.write(f"{len(old - new)} sample(s) found in {old_df_name} not present in new ovitraps:\n")
                    f.write("\n".join([str(s) for s in old - new]) + "\n")
                    f.write("\n" + "-"*80 + "\n\n")