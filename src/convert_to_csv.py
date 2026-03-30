import pandas as pd
import os
import yaml
params = yaml.safe_load(open("params.yaml"))

if not os.path.exists(params["all"]["paths"]["data"]["raw"]["health_centers_csv"]):
    import ipdb; ipdb.set_trace()
    print("Loading health centers from Excel")
    health_centers = pd.read_excel(params["all"]["paths"]["data"]["raw"]["health_centers"])
    health_centers.to_csv(params["all"]["paths"]["data"]["raw"]["health_centers_csv"], index=False)
else:
    print("Health centers CSV already exists.")

if not os.path.exists(params["all"]["paths"]["data"]["raw"]["dengue_csv"]):
    print("Loading dengue data from Excel and convert to CSV")
    dengue_data = pd.read_excel(params["all"]["paths"]["data"]["raw"]["dengue"])
    dengue_data.to_csv(params["all"]["paths"]["data"]["raw"]["dengue_csv"], index=False)
else:
    print("Dengue data CSV already exists.")
if not os.path.exists(params["all"]["paths"]["data"]["raw"]["ovitraps_csv"]):
    print("Loading ovitraps data from Excel and convert to CSV")
    ovitraps_data = pd.read_excel(params["all"]["paths"]["data"]["raw"]["ovitraps"])
    ovitraps_data.to_csv(params["all"]["paths"]["data"]["raw"]["ovitraps_csv"], index=False)
else:
    print("Ovitraps data CSV already exists.")

# Convert IBGE 2010 population data from Excel to CSV
if not os.path.exists(params["all"]["paths"]["data"]["raw"]["ibge_2010_population_csv"]):
    print("Loading IBGE 2010 population data from Excel and convert to CSV")
    
    # Find the actual file with glob pattern to handle special characters
    import glob
    pattern = "data/raw/IBGE/2010/population/Base_informa*setores2010_sinopse_MG.xls"
    matching_files = glob.glob(pattern)
    
    if matching_files:
        file_path = matching_files[0]
        print(f"Found file: {file_path}")
        
        try:
            # Let pandas automatically detect the file format
            ibge_2010_data = pd.read_excel(file_path)
            ibge_2010_data.to_csv(params["all"]["paths"]["data"]["raw"]["ibge_2010_population_csv"], index=False, encoding='utf-8')
            print("IBGE 2010 population data converted successfully")
        except Exception as e:
            print(f"Error converting IBGE 2010 data: {e}")
            print("Unable to convert IBGE 2010 data - file may be corrupted or unsupported format")
    else:
        print(f"No matching files found for pattern: {pattern}")
else:
    print("IBGE 2010 population CSV already exists.")

# Copy IBGE 2022 population data to data/raw with simple name
if not os.path.exists(params["all"]["paths"]["data"]["raw"]["ibge_2022_population_csv"]):
    print("Copying IBGE 2022 population data to data/raw")
    try:
        ibge_2022_data = pd.read_csv(params["all"]["paths"]["data"]["raw"]["ibge_2022_population"], sep=';', encoding='latin-1')
        ibge_2022_data.to_csv(params["all"]["paths"]["data"]["raw"]["ibge_2022_population_csv"], index=False, encoding='utf-8')
        print("IBGE 2022 population data copied successfully")
    except Exception as e:
        print(f"Error copying IBGE 2022 data: {e}")
else:
    print("IBGE 2022 population CSV already exists.")

# Convert IBGE sector history data from Excel to CSV
if not os.path.exists(params["all"]["paths"]["data"]["raw"]["ibge_sector_history_csv"]):
    print("Loading IBGE sector history data from Excel and convert to CSV")
    ibge_history_data = pd.read_excel(params["all"]["paths"]["data"]["raw"]["ibge_sector_history"])
    ibge_history_data.to_csv(params["all"]["paths"]["data"]["raw"]["ibge_sector_history_csv"], index=False, encoding='utf-8')
    print("IBGE sector history data converted successfully")
else:
    print("IBGE sector history CSV already exists.")

# Note: IBGE 2022 population data is already in CSV format