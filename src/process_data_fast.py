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
        "SemEpi": "semana",
        "Ano_Caso": "ano",
        "anoCepid": "anoepid",
    },
    inplace=True,
)

# Extract week from 'semepid' and convert to integer
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

# Drop rows without confirmed Dengue cases
dengue_data.drop(
    dengue_data[dengue_data["Dengue"] == "N"].index,
    axis=0,
    inplace=True,
)

dengue_data.reset_index(drop=True, inplace=True)

### Add useful columns ###
# Epidemic date
dengue_data["epidemic_date"] = project_utils.get_epidemic_date(dengue_data)

# Save
dengue_data.to_csv("data/processed/dengue_data.csv", index=False)


# %% Ovitraps data
# Process
print("Processing ovitraps data")

# Rename columns for consistency
ovitraps_data.rename(
    columns={
        "semepi": "semana",
        "dtcol": "dt_col",
        "dtinstal": "dt_instal",
    },
    inplace=True,
)

# Convert semana into integer
ovitraps_data["semana"] = ovitraps_data["semana"].apply(
    lambda x: int(str(x)[-2:])
)

# Assign epidemic week based on installation date
ovitraps_data["semepid"] = project_utils.assign_epidemic_week(
    ovitraps_data, "dt_instal"
)

ovitraps_data["anoepid"] = project_utils.assign_epidemic_year(
    ovitraps_data, "dt_instal"
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


### Add useful columns ###
# Days of exposition
ovitraps_data["days_expo"] = (
    pd.to_datetime(ovitraps_data["dt_col"])
    - pd.to_datetime(ovitraps_data["dt_instal"])
).dt.days

# Mean eggs per day of exposition
ovitraps_data["eggs_per_day"] = ovitraps_data["novos"] / ovitraps_data[
    "days_expo"
].replace(0, pd.NA)

# Filter traps with exposition time > 30 days and negative exposition time
ovitraps_data = ovitraps_data.drop(
    ovitraps_data[ovitraps_data["days_expo"] > 30].index,
    axis=0,
)

ovitraps_data = ovitraps_data.drop(
    ovitraps_data[ovitraps_data["days_expo"] < 0].index,
    axis=0,
)

# Epidemic date
ovitraps_data["epidemic_date"] = project_utils.get_epidemic_date(
    ovitraps_data
)
# Save Data
ovitraps_data.to_csv("data/processed/ovitraps_data.csv", index=False)
