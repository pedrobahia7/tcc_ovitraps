"""
Stage: process_data

Single-pass processing over the three core raw datasets (health centers,
dengue cases, ovitraps). Merges the former slow_processing and
fast_processing stages.

Health Centers:
  - Corrects known OCR/transcription typos in facility names
  - Standardises column names to lowercase snake_case

Dengue Cases:
  - Reprojects QGIS planar coordinates to WGS-84 lat/lon
  - Assigns nearest health center via BallTree (haversine, O(n log m))
  - Renames columns to snake_case
  - Assigns epidemic week and year from the notification date
  - Drops unconfirmed cases (Dengue == 'N') and duplicates
  - Derives epidemic_date and biweek columns

Ovitraps:
  - Reprojects QGIS planar coordinates to WGS-84 lat/lon
  - Assigns nearest health center via BallTree (haversine, O(n log m))
  - Renames columns to snake_case
  - Corrects known data errors: typo dates, out-of-range collection
    dates, overlapping trap samples, and coordinate duplicates
  - Normalises date columns and enforces a valid exposition window
    (4-21 days); samples outside this range get dt_col = dt_instal + 7d
  - Derives days_expo, eggs_per_day, epidemic_date, and biweek columns
  - Computes daily ovitraps aggregation

Outputs are written to data/dvc/process_data/ and consumed by
add_population_info.
"""

# %% Import libraries
import os
import sys
import pandas as pd
import yaml

sys.path.append("utils")
import project_utils

params = yaml.safe_load(open("params.yaml"))
_in = params['all']['paths']['data']['dvc']['convert_to_csv']
_out = params['all']['paths']['data']['dvc']['process_data']

# %% Load all data
print("Loading data")
health_centers = pd.read_csv(_in['health_centers_csv'])
dengue_data = pd.read_csv(_in['dengue_csv'])
ovitraps_data = pd.read_csv(_in['ovitraps_csv'])

# %% Prepare output folder
os.makedirs(_out['folder'], exist_ok=True)


# ================================================================
# %% Health Centers
# ================================================================
print("Processing health centers data")

_HC_COL = "CENTRO DE SAÚDE"
health_centers[_HC_COL] = health_centers[_HC_COL].replace(
    {
        "BONSUO": "BONSUCESSO",
        "TARO": "TARCISIO",
        "DE IA": "DE CASSIA",
        "FRANO": "FRANCISCO",
        "DE RO": "DE CASTRO",
    },
    regex=True,
)

health_centers = health_centers.rename(
    columns={
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        _HC_COL: "health_center",
    },
)

health_centers.to_csv(_out['health_centers'], index=False)


# ================================================================
# %% Dengue Cases
# ================================================================
print("Processing dengue data")

# Reproject coordinates and assign nearest health center
dengue_data = project_utils.convert_qgis_to_latlon(dengue_data)
dengue_data["closest_health_center"] = project_utils.closest_health_center(
    dengue_data, health_centers
)

# Rename columns
dengue_data = dengue_data.rename(
    columns={
        "SemEpi": "semana",
        "Ano_Caso": "ano",
        "anoCepid": "anoepid",
    },
)

# Extract week number from semana
dengue_data["semana"] = dengue_data["semana"].apply(
    lambda x: int(str(x)[-2:])
)

# Assign epidemic week and year based on notification date
dengue_data["semepid"] = project_utils.assign_epidemic_week(
    dengue_data, "dt_notific"
)
dengue_data["anoepid"] = project_utils.assign_epidemic_year(
    dengue_data, "dt_notific"
)

# Drop unconfirmed cases and duplicates
dengue_data = (
    dengue_data[dengue_data["Dengue"] != "N"]
    .drop_duplicates()
    .reset_index(drop=True)
)

# Derived columns
dengue_data["epidemic_date"] = project_utils.get_epidemic_date(dengue_data)
dengue_data["biweek"] = project_utils.epidemic_date_to_biweek(
    dengue_data["epidemic_date"]
)

dengue_data.to_csv(_out['dengue'], index=False)


# ================================================================
# %% Ovitraps
# ================================================================
print("Processing ovitraps data")

# Reproject coordinates and assign nearest health center
ovitraps_data = project_utils.convert_qgis_to_latlon(ovitraps_data)
ovitraps_data["closest_health_center"] = project_utils.closest_health_center(
    ovitraps_data, health_centers
)

# Rename columns
ovitraps_data = ovitraps_data.rename(
    columns={
        "semepi": "semana",
        "dtcol": "dt_col",
        "dtinstal": "dt_instal",
    },
)

# Extract week number from semana
ovitraps_data["semana"] = ovitraps_data["semana"].apply(
    lambda x: int(str(x)[-2:])
)

# Assign epidemic week and year based on installation date
ovitraps_data["semepid"] = project_utils.assign_epidemic_week(
    ovitraps_data, "dt_instal"
)
ovitraps_data["anoepid"] = project_utils.assign_epidemic_year(
    ovitraps_data, "dt_instal"
)

# Correct known typo date
ovitraps_data.loc[
    ovitraps_data["dt_col"] == "2032-09-14", "dt_col"
] = "2023-09-14"


def correct_value(narmad, dt_col, column, new_value):
    mask = (ovitraps_data["narmad"] == narmad) & (
        True if dt_col is None
        else (ovitraps_data["dt_col"] == dt_col)
    )
    assert not ovitraps_data.loc[mask, column].empty, (
        f"No rows for narmad={narmad}, dt_col={dt_col} in {column}"
    )
    ovitraps_data.loc[mask, column] = new_value


# Correct known data errors in collection dates and egg counts
correct_value(901011, "2017-04-20", "dt_col", "2016-03-08")
correct_value(901013, "2017-04-20", "dt_col", "2016-03-08")
correct_value(901199, "2021-01-27", "dt_col", "2020-04-13")
correct_value(909027, "2025-05-08", "dt_col", "2024-05-08")
correct_value(906071, "2022-08-18", "novos", 50)

# Correct coordinates for traps with wrong or duplicate positions.
# 904068: CEP is far from recorded coordinates — corrected from inspection.
correct_value(904068, None, "latitude", -19.87999615)
correct_value(904068, None, "longitude", -43.92464141)

# 908105 was deactivated in 2016; 908104 is still active at the same CEP.
# Offset 908105 slightly so both are spatially distinct.
aux_lat, aux_lon = (
    ovitraps_data.loc[
        ovitraps_data["narmad"] == 908104, ["latitude", "longitude"]
    ]
    .drop_duplicates()
    .iloc[0]
)
correct_value(908105, None, "latitude", aux_lat + 0.0002)
correct_value(908105, None, "longitude", aux_lon + 0.0002)

# 903151 and 903152 share the same CEP and were both deactivated in 2023.
# Offset 903152 slightly.
aux_lat, aux_lon = (
    ovitraps_data.loc[
        ovitraps_data["narmad"] == 903152, ["latitude", "longitude"]
    ]
    .drop_duplicates()
    .iloc[0]
)
correct_value(903152, None, "latitude", aux_lat - 0.0002)
correct_value(903152, None, "longitude", aux_lon - 0.0002)

# Normalise date columns to datetime
ovitraps_data["dt_col"] = pd.to_datetime(
    ovitraps_data["dt_col"], format="mixed"
).dt.normalize()
ovitraps_data["dt_instal"] = pd.to_datetime(
    ovitraps_data["dt_instal"], format="mixed"
).dt.normalize()

# Fix invalid collection dates: equal to, before, or missing vs dt_instal
_fallback = ovitraps_data["dt_instal"] + pd.Timedelta(days=7)
ovitraps_data.loc[
    ovitraps_data["dt_col"] == ovitraps_data["dt_instal"], "dt_col"
] = _fallback
ovitraps_data.loc[
    ovitraps_data["dt_instal"] > ovitraps_data["dt_col"], "dt_col"
] = _fallback
ovitraps_data.loc[ovitraps_data["dt_col"].isna(), "dt_col"] = _fallback

# Fix overlapping sample windows: reset the first sample's dt_col
overlapped_traps = project_utils.get_overlapped_samples(
    ovitraps_data, processed_name=True
)
overlap_mask = ovitraps_data["nplaca"].isin(
    {pair[0] for pair in overlapped_traps}
)
ovitraps_data.loc[overlap_mask, "dt_col"] = (
    ovitraps_data.loc[overlap_mask, "dt_instal"] + pd.Timedelta(days=7)
)

# Drop rows missing critical data
ovitraps_data = ovitraps_data[ovitraps_data["novos"].notna()]

# Days of exposition
ovitraps_data["days_expo"] = (
    ovitraps_data["dt_col"] - ovitraps_data["dt_instal"]
).dt.days

# Clamp exposition to valid range [4, 21] days
_out_of_range = (
    (ovitraps_data["days_expo"] > 21) | (ovitraps_data["days_expo"] < 4)
)
ovitraps_data.loc[_out_of_range, "dt_col"] = (
    ovitraps_data["dt_instal"] + pd.Timedelta(days=7)
)
ovitraps_data["days_expo"] = (
    ovitraps_data["dt_col"] - ovitraps_data["dt_instal"]
).dt.days

ovitraps_data["eggs_per_day"] = (
    ovitraps_data["novos"] / ovitraps_data["days_expo"]
)
ovitraps_data["epidemic_date"] = project_utils.get_epidemic_date(
    ovitraps_data
)
ovitraps_data["biweek"] = project_utils.epidemic_date_to_biweek(
    ovitraps_data["epidemic_date"]
)

ovitraps_data["narmad"] = ovitraps_data["narmad"].astype(int).astype(str)
ovitraps_data["nplaca"] = ovitraps_data["nplaca"].astype(int).astype(str)

print("Computing daily ovitraps")
daily_ovitraps = project_utils.get_daily_ovitraps(ovitraps_data)
daily_ovitraps.to_csv(
    _out["daily_ovitraps"], index=True, date_format="%Y-%m-%d"
)

ovitraps_data.to_csv(_out["ovitraps"], index=False)
