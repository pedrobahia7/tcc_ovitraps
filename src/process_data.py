# %% Import libraries
import pandas as pd
import sys
import os

sys.path.append("utils")
import project_utils

# %% Load all data
print("Loading data")
if os.path.exists("data/raw/CENTRO_SAUDE_new.csv"):
    print("Loading health centers from CSV")
    health_centers = pd.read_csv("data/raw/CENTRO_SAUDE_new.csv")
else:
    print(
        "Health centers from CSV doesn't exist. Loading from Excel and convert."
    )
    health_centers = pd.read_excel("data/raw/CENTRO_SAUDE_new.xlsx")
    health_centers.to_csv("data/raw/CENTRO_SAUDE_new.csv", index=False)

if os.path.exists("data/raw/Dengue2007_2025.csv"):
    print("Loading dengue data from CSV")
    dengue_data = pd.read_csv("data/raw/Dengue2007_2025.csv")
else:
    print(
        "Dengue data from CSV doesn't exist. Loading from Excel and convert."
    )
    dengue_data = pd.read_excel("data/raw/Dengue2007_2025.xlsx")
    dengue_data.to_csv("data/raw/Dengue2007_2025.csv", index=False)

if os.path.exists("data/raw/MasterDataExtend062025.csv"):
    print("Loading ovitraps data from CSV")
    ovitraps_data = pd.read_csv("data/raw/MasterDataExtend062025.csv")
else:
    print(
        "Ovitraps data from CSV doesn't exist. Loading from Excel and convert."
    )
    ovitraps_data = pd.read_excel("data/raw/MasterDataExtend062025.xlsx")
    ovitraps_data.to_csv(
        "data/raw/MasterDataExtend062025.csv", index=False
    )

# %% Health Centers data
# Load
print("Processing health centers data")
health_centers["CENTRO DE SAÚDE"].replace(
    {
        "BONSUO": "BONSUCESSO",
        "TARO": "TARCISIO",
        "DE IA": "DE CASSIA",
        "FRANO": "FRANCISCO",
        "DE RO": "DE CASTRO",
    },
    regex=True,
    inplace=True,
)

health_centers.rename(
    columns={
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "CENTRO DE SAÚDE": "health_center",
    },
    inplace=True,
)

health_centers.to_csv("data/processed/health_centers.csv", index=False)

# %% Dengue data
# Process
print("Processing dengue data")
dengue_data = project_utils.process_dengue(dengue_data)
dengue_data["closest_health_center"] = project_utils.closest_health_center(
    dengue_data, health_centers
)

dengue_data.to_csv("data/processed/dengue_data.csv", index=False)


# %% Ovitraps data
# Process
print("Processing ovitraps data")
ovitraps_data = project_utils.process_ovitraps(ovitraps_data)
ovitraps_data["closest_health_center"] = (
    project_utils.closest_health_center(ovitraps_data, health_centers)
)

ovitraps_data.to_csv("data/processed/ovitraps_data.csv", index=False)

# Get daily ovitraps
daily_ovitraps = project_utils.get_daily_ovitraps(ovitraps_data)
daily_ovitraps.to_csv(
    "data/processed/daily_ovitraps.csv",
    index=True,
    date_format="%Y-%m-%d",
)
