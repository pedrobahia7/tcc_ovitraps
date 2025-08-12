# %% Import libraries
import pandas as pd
import os
import sys

sys.path.append("..")
import utils.project_utils as project_utils


# %% Dengue data
# Load the processed datasets if they exist, otherwise load the raw data
# and process them
if os.path.isfile("../data/processed/dengue_data_with_coordinates.csv"):
    dengue_data = pd.read_csv(
        "../data/processed/dengue_data_with_coordinates.csv"
    )

else:
    dengue_data = pd.read_excel("../data/raw/Dengue2007_2025.xlsx")
    print("PROCESS LATER")


# %% Ovitraps data
# Load
if os.path.isfile("../data/processed/ovitraps_data_with_coordinates.csv"):
    ovitraps_data = pd.read_csv(
        "../data/processed/ovitraps_data_with_coordinates.csv"
    )
else:
    ovitraps_data = pd.read_excel(
        "../data/raw/MasterDataExtend062025.xlsx"
    )
    print("PROCESS LATER")


# Process
daily_ovitraps = project_utils.get_daily_ovitraps(ovitraps_data)

# Save
daily_ovitraps.to_csv(
    "../data/processed/daily_ovitraps.csv",
    index=True,
    date_format="%Y-%m-%d",
)

# %% Health Centers data
# Load
