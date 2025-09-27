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