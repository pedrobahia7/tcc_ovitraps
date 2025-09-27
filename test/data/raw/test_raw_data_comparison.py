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
import yaml
params = yaml.safe_load(open("params.yaml"))

diagnosis_file = params["all"]["paths"]["test"]["data"]["raw"]["diagnosis_raw_comparison"]
with open(diagnosis_file, "w", encoding="utf-8") as f:
    pass


class TestRawDataComparison:
    @pytest.fixture(scope="class")
    def data_paths(self):
        """Fixture providing paths to processed data files."""
        return {
            'dengue': params["all"]["paths"]["data"]["raw"]["dengue_csv"],
            'ovitraps': params["all"]["paths"]["data"]["raw"]["ovitraps_csv"],
            'health_centers': params["all"]["paths"]["data"]["raw"]["health_centers_csv"],
            'old_ovitraps': ["data/final_data.csv","data/raw/MasterDataExtend062025_v1.csv"]
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

        After talking to Dilermando, it's not possible to consider 'nplaca' as consistent between datasets.
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
                    f.write(mismatches[:10].to_string(index=False))
                    f.write("\n" + "-"*80 + "\n\n")

    def test_real_identificator_consistency(self, ovitraps_data, old_ovitraps_data):
        """
        Test consistency of ('narmad', 'dtcol', 'dtinstal') between old and new ovitraps files.
        If the tuple value is present in both files, it must have the same 'novos'.

        Dilermando said it's uncommon but possible that the dataset are corrected after some time, 
        so new 'novos' may appear.
        """
        key_fields = ["dtinstal", "narmad"]

        for old_df_name, old_df in old_ovitraps_data.items():
            # Ensure both DataFrames have 'nplaca'
            assert all([field in ovitraps_data.columns for field in key_fields]), "some column missing in new ovitraps data"
            assert all([field in old_df.columns for field in key_fields]), "some column missing in old ovitraps data"

            # Keep only relevant columns
            new = ovitraps_data[key_fields + ["dtcol", "novos"]].dropna(subset=key_fields)
            old = old_df[key_fields + ["dtcol", "novos"]].dropna(subset=key_fields)

            # Inner join on nplaca
            merged = new.merge(old, on=key_fields, suffixes=("_new", "_old"))

            # Find mismatches in any key field
            mismatches = merged[
                merged["novos_new"].astype(str).ne(merged["novos_old"].astype(str)) |
                merged["dtcol_new"].astype(str).ne(merged["dtcol_old"].astype(str))
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
        Test existence of ('narmad', 'dtinstal') in both old and new ovitraps files.
        """
        key_fields = ["dtinstal", "narmad"]

        for old_df_name, old_df in old_ovitraps_data.items():
            # Ensure both DataFrames have 'nplaca'
            assert all([field in ovitraps_data.columns for field in key_fields]), "some column missing in new ovitraps data"
            assert all([field in old_df.columns for field in key_fields]), "some column missing in old ovitraps data"

            # Keep only relevant columns
            new = set(map(tuple, ovitraps_data[key_fields].values.tolist()))
            old = set(map(tuple, old_df[key_fields].values.tolist()))
            try:
                assert len(new - old) == 0, f"Sample(s) in new ovitraps not present in old ovitraps"
            except AssertionError:
                with open(diagnosis_file, "a", encoding="utf-8") as f:
                    diff = list(new - old)   # convert set to list
                    diff.sort(key=lambda x: x[0])
                    f.write(f"{len(diff)} sample(s) found in new ovitraps not present in {old_df_name}:\n")
                    f.write("\n".join([str(s) for s in diff]) + "\n")
                    f.write("\n" + "-"*80 + "\n\n") 
            try:
                assert len(old - new) == 0, f"Sample(s) in old ovitraps not present in new ovitraps"
            except AssertionError:
                with open(diagnosis_file, "a", encoding="utf-8") as f:
                    diff = list(old - new)   # convert set to list
                    diff.sort(key=lambda x: x[0])
                    f.write(f"{len(diff)} sample(s) found in {old_df_name} not present in new ovitraps:\n")
                    f.write("\n".join([str(s) for s in diff]) + "\n")
                    f.write("\n" + "-"*80 + "\n\n")