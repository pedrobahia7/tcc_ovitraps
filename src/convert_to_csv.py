import pandas as pd
import os

if not os.path.exists("data/raw/CENTRO_SAUDE_new.csv"):
    print("Loading health centers from Excel")
    health_centers = pd.read_excel("data/raw/CENTRO_SAUDE_new.xlsx")
    health_centers.to_csv("data/raw/CENTRO_SAUDE_new.csv", index=False)

if not os.path.exists("data/raw/Dengue2007_2025.csv"):
    print("Loading dengue data from Excel and convert to CSV")
    dengue_data = pd.read_excel("data/raw/Dengue2007_2025.xlsx")
    dengue_data.to_csv("data/raw/Dengue2007_2025.csv", index=False)

if not os.path.exists("data/raw/MasterDataExtend062025.csv"):
    print("Loading ovitraps data from Excel and convert to CSV")
    ovitraps_data = pd.read_excel("data/raw/MasterDataExtend062025.xlsx")
    ovitraps_data.to_csv(
        "data/raw/MasterDataExtend062025.csv", index=False
    )
