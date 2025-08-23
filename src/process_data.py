# %% Import libraries
import pandas as pd
import os
import sys

sys.path.append("utils")
import project_utils

# %% Health Centers data
# Load
if os.path.isfile("data/processed/health_centers.csv"):
    health_centers = pd.read_csv("data/processed/health_centers.csv")

else:
    health_centers = pd.read_excel("data/raw/HealthCenters.xlsx")
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
# Load the processed datasets if they exist, otherwise load the raw data
# and process them
if os.path.isfile("data/processed/dengue_data_with_coordinates.csv"):
    dengue_data = pd.read_csv(
        "data/processed/dengue_data_with_coordinates.csv"
    )

else:
    dengue_data = pd.read_excel("data/raw/Dengue2007_2025.xlsx")
    project_utils.process_dengue_data(dengue_data)
    dengue_data["closest_health_center"] = (
        project_utils.closest_health_center(dengue_data, health_centers)
    )


# %% Ovitraps data
# Load
if os.path.isfile("data/processed/ovitraps_data_with_coordinates.csv"):
    ovitraps_data = pd.read_csv(
        "data/processed/ovitraps_data_with_coordinates.csv"
    )
else:
    ovitraps_data = pd.read_excel("data/raw/MasterDataExtend062025.xlsx")
    project_utils.process_ovitraps(ovitraps_data)
    ovitraps_data["closest_health_center"] = (
        project_utils.closest_health_center(ovitraps_data, health_centers)
    )

# Process
daily_ovitraps = project_utils.get_daily_ovitraps(ovitraps_data)

# Save
daily_ovitraps.to_csv(
    "data/processed/daily_ovitraps.csv",
    index=True,
    date_format="%Y-%m-%d",
)
