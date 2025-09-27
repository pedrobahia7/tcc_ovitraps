# %% Import libraries
import os
import pandas as pd
import sys
import yaml

sys.path.append("utils")
import project_utils

params = yaml.safe_load(open("params.yaml"))
# %% Load all data
print("Loading data")
dengue_data = pd.read_csv(params['all']['paths']['data']['processed']['slow']['dengue'])
ovitraps_data = pd.read_csv(params['all']['paths']['data']['processed']['slow']['ovitraps'])
health_centers = pd.read_csv(params['all']['paths']['data']['processed']['slow']['health_centers'])

# %% Prepare folders
os.makedirs(params['all']['paths']['data']['processed']['folder'], exist_ok=True)


# %% Health Centers data
# Load

# Save
health_centers.to_csv(params['all']['paths']['data']['processed']['health_centers'], index=False)

# %% Dengue data
# Process
print("Processing dengue data")

# Rename columns
dengue_data.rename(
    columns={
        "SemEpi": "semana",
        "Ano_Caso": "ano",
        "anoCepid": "anoepid",
    },
    inplace=True,
)

# Extract week from 'semepid' and convert to integer
dengue_data["semana"] = dengue_data["semana"].apply(
    lambda x: int(str(x)[-2:])
)

# Assign epidemic week and year based on notification date
dengue_data["semepid"] = project_utils.assign_epidemic_week(
    dengue_data, "dt_notific"
)

dengue_data["anoepid"] = project_utils.assign_epidemic_year(
    dengue_data, "dt_notific"
)

# Drop rows without confirmed Dengue cases
dengue_data.drop(
    dengue_data[dengue_data["Dengue"] == "N"].index,
    axis=0,
    inplace=True,
)

# Drop duplicated rows 
dengue_data.drop_duplicates(inplace=True)

dengue_data.reset_index(drop=True, inplace=True)

### Add useful columns ###
# Epidemic date
dengue_data["epidemic_date"] = project_utils.get_epidemic_date(dengue_data)

# Save
dengue_data.to_csv(params['all']['paths']['data']['processed']['dengue'], index=False)


# %% Ovitraps data
# Process
print("Processing ovitraps data")

# Rename columns for consistency
ovitraps_data.rename(
    columns={
        "semepi": "semana",
        "dtcol": "dt_col",
        "dtinstal": "dt_instal",
    },
    inplace=True,
)

# Convert semana into integer
ovitraps_data["semana"] = ovitraps_data["semana"].apply(
    lambda x: int(str(x)[-2:])
)

# Assign epidemic week based on installation date
ovitraps_data["semepid"] = project_utils.assign_epidemic_week(
    ovitraps_data, "dt_instal"
)

ovitraps_data["anoepid"] = project_utils.assign_epidemic_year(
    ovitraps_data, "dt_instal"
)


# Correct wrong dates
ovitraps_data.loc[ovitraps_data["dt_col"] == "2032-09-14", "dt_col"] = (
    "2023-09-14"
)

def correct_value(narmad, dt_col, column, new_value):
    assert not ovitraps_data.loc[
        (ovitraps_data["narmad"] == narmad) & (ovitraps_data["dt_col"] == dt_col),
        column].empty, f"Value for narmad={narmad}, dt_col={dt_col} does not exist in {column}"

    ovitraps_data.loc[
        (ovitraps_data["narmad"] == narmad) & (ovitraps_data["dt_col"] == dt_col),
        column,
    ] = new_value
    
correct_value(901011, "2017-04-20", "dt_col", "2016-03-08")
correct_value(901013, "2017-04-20", "dt_col", "2016-03-08")
correct_value(901199, "2021-01-27", "dt_col", "2020-04-13") 
correct_value(909027, "2025-05-08", "dt_col", "2024-05-08")
correct_value(906071, "2022-08-18", "novos", 50) # Correct wrong novos according to old data frame

# Convert date columns to datetime format and standardize to YYYY-MM-DD
ovitraps_data['dt_col'] = pd.to_datetime(ovitraps_data['dt_col'], format="mixed").dt.normalize()
ovitraps_data['dt_instal'] = pd.to_datetime(ovitraps_data['dt_instal'], format="mixed").dt.normalize()

# Same installation and collection date. Collection date is probably wrong
# so I'll set it to seven days after installation date
ovitraps_data.loc[ovitraps_data["dt_col"] == ovitraps_data["dt_instal"], "dt_col"] = (
    ovitraps_data["dt_instal"] + pd.Timedelta(days=7)
)

# Do the same for installation dates that are after collection dates
# And for NaN collection dates
ovitraps_data.loc[ovitraps_data["dt_instal"] > ovitraps_data["dt_col"], "dt_col"] = (
    ovitraps_data["dt_instal"] + pd.Timedelta(days=7)
)

ovitraps_data.loc[ovitraps_data["dt_col"].isna(), "dt_col"] = (
    ovitraps_data["dt_instal"] + pd.Timedelta(days=7)
)

# And for samples with overlapping dates. In this case, we'll set the collection date
# of the first sample to seven days after installation date. This happens because, after
# inspection, it was found that the exposition period of the first samples was longer than
# 7 days while the second sample has a valid exposition period.

overlapped_traps = project_utils.get_overlapped_samples(ovitraps_data, processed_name=True)
samples_to_fix = set([pair[0] for pair in overlapped_traps])
mask = ovitraps_data['nplaca'].isin(samples_to_fix)
ovitraps_data.loc[mask, 'dt_col'] = (
            ovitraps_data.loc[mask, 'dt_instal']
        ) + pd.Timedelta(days=7)


# Drop rows with missing critical information
ovitraps_data.drop(
    ovitraps_data[ovitraps_data["novos"].isna()].index,
    axis=0,
    inplace=True,
)

#### Add useful columns ####
# Days of exposition
ovitraps_data["days_expo"] = (ovitraps_data["dt_col"] - ovitraps_data["dt_instal"]).dt.days

# Correct dt_col in traps with exposition time > 21 days
# or < 4 days (limits established by Dilermando according to low frequency of
# such cases and probable data errors)
ovitraps_data.loc[ovitraps_data["days_expo"] > 21, "dt_col"] = (
    ovitraps_data["dt_instal"] + pd.Timedelta(days=7)
)
ovitraps_data.loc[ovitraps_data["days_expo"] < 4, "dt_col"] = (
    ovitraps_data["dt_instal"] + pd.Timedelta(days=7)
)

# Recalculate days of exposition
ovitraps_data["days_expo"] = (ovitraps_data["dt_col"] - ovitraps_data["dt_instal"]).dt.days

ovitraps_data['eggs_per_day'] = ovitraps_data['novos'] / ovitraps_data['days_expo']

# Epidemic date
ovitraps_data["epidemic_date"] = project_utils.get_epidemic_date(
    ovitraps_data
)
# Save Data
ovitraps_data.to_csv(params['all']['paths']['data']['processed']['ovitraps'], index=False)
