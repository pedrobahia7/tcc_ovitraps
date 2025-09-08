import pandas as pd
import sys

sys.path.append("utils")
import project_utils

ovitraps_data = pd.read_csv("data/processed/ovitraps_data.csv")

# Get daily ovitraps
print("Daily ovitraps logic")
daily_ovitraps = project_utils.get_daily_ovitraps(ovitraps_data)
daily_ovitraps.to_csv(
    "data/processed/daily_ovitraps.csv",
    index=True,
    date_format="%Y-%m-%d",
)
