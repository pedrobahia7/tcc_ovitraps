# %% Import libraries
import pandas as pd
import sys

sys.path.append("utils")
import project_utils

# %% Load all data
print("Loading data")
health_centers = pd.read_excel("data/raw/CENTRO_SAUDE_new.xlsx")
dengue_data = pd.read_excel("data/raw/Dengue2007_2025.xlsx")
ovitraps_data = pd.read_excel("data/raw/MasterDataExtend062025.xlsx")

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

# Save
health_centers.to_csv("data/processed/health_centers.csv", index=False)

# %% Dengue data
# Process
print("Processing dengue data")
project_utils.process_dengue_data(dengue_data)
dengue_data["closest_health_center"] = project_utils.closest_health_center(
    dengue_data, health_centers
)
dengue_data.to_csv("data/processed/dengue_data.csv", index=True)


# %% Ovitraps data
# Process
print("Processing ovitraps data")
project_utils.process_ovitraps(ovitraps_data)
ovitraps_data["closest_health_center"] = (
    project_utils.closest_health_center(ovitraps_data, health_centers)
)
daily_ovitraps = project_utils.get_daily_ovitraps(ovitraps_data)

# Save
daily_ovitraps.to_csv(
    "data/processed/daily_ovitraps.csv",
    index=True,
    date_format="%Y-%m-%d",
)
