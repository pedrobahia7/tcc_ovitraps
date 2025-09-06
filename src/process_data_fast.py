# %% Import libraries
import os
import pandas as pd
import sys

sys.path.append("utils")
import project_utils

# %% Load all data
print("Loading data")
dengue_data = pd.read_csv("data/processed/slow/dengue_data_slow.csv")
ovitraps_data = pd.read_csv("data/processed/slow/ovitraps_data_slow.csv")
health_centers = pd.read_csv("data/processed/slow/health_centers_slow.csv")

# %% Prepare folders
os.makedirs("data/processed/", exist_ok=True)


# %% Health Centers data
# Load

# Save
health_centers.to_csv("data/processed/health_centers.csv", index=False)

# %% Dengue data
# Process
print("Processing dengue data")

# Rename columns
dengue_data.rename(
    columns={
        "SemEpi": "semepid",
        "Ano_Caso": "ano",
        "anoCepid": "anoepid",
    },
    inplace=True,
)

# Extract week from 'semepid' and convert to integer
dengue_data["semepid"] = dengue_data["semepid"].apply(
    lambda x: int(str(x)[-2:])
)

# Drop rows without confirmed Dengue cases
dengue_data.drop(
    dengue_data[dengue_data["Dengue"] == "N"].index,
    axis=0,
    inplace=True,
)

dengue_data.reset_index(drop=True, inplace=True)

# Add useful columns
dengue_data["epidemic_date"] = project_utils.get_epidemic_date(dengue_data)

# Save
dengue_data.to_csv("data/processed/dengue_data.csv", index=False)


# %% Ovitraps data
# Process
print("Processing ovitraps data")

# Rename columns for consistency
ovitraps_data.rename(
    columns={
        "semepi": "semepid",
        "dtcol": "dt_col",
        "dtinstal": "dt_instal",
    },
    inplace=True,
)

# Convert semepid to string with at least two digits
ovitraps_data["semepid"] = ovitraps_data["semepid"].apply(
    lambda x: f"{int(x) - 100:02d}" if pd.notnull(x) else x
)

# Correct wrong dates
ovitraps_data.loc[ovitraps_data["dt_col"] == "2032-09-14", "dt_col"] = (
    "2023-09-14"
)
ovitraps_data.loc[
    (ovitraps_data["narmad"] == 901011)
    & (ovitraps_data["dt_col"] == "2017-04-20"),
    "dt_col",
] = "2016-03-08"

ovitraps_data.loc[
    (ovitraps_data["narmad"] == 901013)
    & (ovitraps_data["dt_col"] == "2017-04-20"),
    "dt_col",
] = "2016-03-08"

ovitraps_data.loc[
    (ovitraps_data["narmad"] == 901199)
    & (ovitraps_data["dt_col"] == "2021-01-27"),
    "dt_col",
] = "2020-04-13"

ovitraps_data.loc[
    (ovitraps_data["dt_col"] == "2032-09-14"),
    "dt_col",
] = "2023-09-14"

ovitraps_data.loc[
    (ovitraps_data["narmad"] == 909027)
    & (ovitraps_data["dt_col"] == "2025-05-08"),
    "dt_col",
] = "2024-05-08"


# Add useful columns
ovitraps_data["days_expo"] = (
    pd.to_datetime(ovitraps_data["dt_col"])
    - pd.to_datetime(ovitraps_data["dt_instal"])
).dt.days

ovitraps_data = ovitraps_data.drop(
    ovitraps_data[ovitraps_data["days_expo"] > 30].index,
)

ovitraps_data["epidemic_date"] = project_utils.get_epidemic_date(
    ovitraps_data
)
# Save Data
ovitraps_data.to_csv("data/processed/ovitraps_data.csv", index=False)

# Get daily ovitraps
print("Daily ovitraps logic")
daily_ovitraps = project_utils.get_daily_ovitraps(ovitraps_data)
daily_ovitraps.to_csv(
    "data/processed/daily_ovitraps.csv",
    index=True,
    date_format="%Y-%m-%d",
)
