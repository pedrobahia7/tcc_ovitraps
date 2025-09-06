# %% Import libraries
import pandas as pd
import sys
import os

sys.path.append("utils")
import project_utils

# %% Load all data.
print("Loading data")
print("Loading health centers from CSV")
health_centers = pd.read_csv("data/raw/CENTRO_SAUDE_new.csv")
print("Loading dengue data from CSV")
dengue_data = pd.read_csv("data/raw/Dengue2007_2025.csv")
print("Loading ovitraps data from CSV")
ovitraps_data = pd.read_csv("data/raw/MasterDataExtend062025.csv")

# %% Prepare folders
os.makedirs("data/processed/slow/", exist_ok=True)


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

health_centers.to_csv(
    "data/processed/slow/health_centers_slow.csv", index=False
)

# %% Dengue data
# Process
print("Processing dengue data")

# Convert coordinates to latitude and longitude
dengue_data = project_utils.convert_qgis_to_latlon(dengue_data)


dengue_data["closest_health_center"] = project_utils.closest_health_center(
    dengue_data, health_centers
)

dengue_data.to_csv("data/processed/slow/dengue_data_slow.csv", index=False)


# %% Ovitraps data
# Process
print("Processing ovitraps data")
ovitraps_data = project_utils.convert_qgis_to_latlon(ovitraps_data)
ovitraps_data["closest_health_center"] = (
    project_utils.closest_health_center(ovitraps_data, health_centers)
)

ovitraps_data.to_csv(
    "data/processed/slow/ovitraps_data_slow.csv", index=False
)
