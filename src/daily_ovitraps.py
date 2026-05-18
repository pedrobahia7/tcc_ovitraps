"""
Stage: daily_ovitraps

Aggregates the fully-processed ovitrap records into a daily time series.
Delegates entirely to project_utils.get_daily_ovitraps, which groups records
by trap and date and fills gaps so every trap has a continuous daily index.

Input:  data/processed/ovitraps_data.csv (output of add_population_sectors)
Output: data/processed/daily_ovitraps.csv — one row per (trap, date)
"""

import pandas as pd
import sys
import yaml
sys.path.append("utils")
import project_utils

params = yaml.safe_load(open("params.yaml"))

# %% Load data
ovitraps_data = project_utils.load_ovitraps_data(params['all']['paths']['data']['processed']['ovitraps'])

# Get daily ovitraps
print("Daily ovitraps logic")
daily_ovitraps = project_utils.get_daily_ovitraps(ovitraps_data)
daily_ovitraps.to_csv(
    params['all']['paths']['data']['processed']['daily_ovitraps'],
    index=True,
    date_format="%Y-%m-%d",
)
