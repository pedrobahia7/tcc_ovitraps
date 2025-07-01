import pandas as pd
import numpy as np

################ Dengue Cases Functions ################


def process_dengue(dengue_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the dengue data DataFrame to prepare it for further analysis.
    This function renames the 'SemEpi' column to 'semepid', extracts the week
    from 'semepid', and converts it to an integer.
    
    Parameters
    ----------
    - dengue_data (pd.DataFrame): DataFrame containing dengue data with columns\
        'Ano_Caso', 'SemEpi', and 'novos'.
    
    Returns
    ----------
    - dengue_data (pd.DataFrame): Processed DataFrame with 'semepid' as an integer\
      column, ready for further analysis.
    
    """

    # Step 1: Rename and clean the DataFrame
    # Rename columns
    dengue_data.rename(
        columns={
            "SemEpi": "semepid",
            "Ano_Caso": "ano",
            "anoCepid": "anoepid",
        },
        inplace=True,
    )

    # Extract week from 'semepid' and convert to integer
    dengue_data["semepid"] = dengue_data["semepid"].apply(
        lambda x: int(str(x)[-2:])
    )

    return dengue_data


def get_weekly_dengue(
    dengue_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw dengue data to a pivoted format suitable for comparison
    with ovitrap data, with 'ano_Caso' and 'semepid' as index. The
    DataFrame will contain counts of 'novos' cases.

    Parameters
    ----------
    - data (pd.DataFrame): DataFrame containing dengue data with columns\
        'Ano_Caso', 'semepid', and 'novos'.

    Returns
    ----------
    - weekly_data (pd.DataFrame): A pivoted DataFrame with "'ano'W'semepid'"
      as index, where each cell contains the count of 'novos' cases in
      each week.

    """
    # Step 1: Rename and clean the DataFrame
    dengue_data = process_dengue(dengue_data)

    # Create a new datetime column from ano and semepid
    dengue_data["date"] = dengue_data.apply(
        lambda row: str(row["anoepid"]) + "W" + str(row["semepid"])
        if row["semepid"] > 9
        else str(row["anoepid"]) + "W0" + str(row["semepid"]),
        axis=1,
    )

    # Pivot the DataFrame to create a matrix with 'ano' and 'semepid' as index
    pivot_data = (
        dengue_data.groupby(["date"]).size().reset_index(name="count")
    )

    pivot_data.set_index(["date"], inplace=True)

    # Step 2: Generate new MultiIndex with all weeks
    new_tuples = generate_all_weeks(pivot_data)

    # Step 3: Combine with existing index and reindex the DataFrame
    weekly_data = pivot_data.reindex(new_tuples).sort_index()

    # Step 4: Fill NaN values with 0
    weekly_data.replace(np.nan, 0, inplace=True)

    return weekly_data


def get_daily_dengue(
    dengue_data: pd.DataFrame,
) -> pd.Series:
    """
    Convert raw dengue data to a daily time series format suitable for analysis.
    The DataFrame will contain counts of 'novos' cases per day.

    Parameters
    ----------
    - dengue_data (pd.DataFrame): DataFrame containing dengue data with columns\
        'dt_notific' and 'novos'.

    Returns
    ----------
    - dengue_cases_serie (pd.Series): A Series indexed by 'dt_notific', where each\
      value is the count of 'novos' cases on that date.

    """
    # Step 1: Rename and clean the DataFrame
    dengue_data = process_dengue(dengue_data)

    # Step 2: Get daily counts of dengue cases
    dengue_cases_serie = (
        dengue_data.groupby("dt_notific").size().reset_index(name="count")
    )

    # Step 3: Reset index
    dengue_cases_serie.set_index("dt_notific", inplace=True)

    return dengue_cases_serie


################ Ovitraps Eggs Functions ################


def process_ovitraps(ovitraps_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the raw ovitraps data by renaming columns and filtering relevant
    information.

    Parameters
    ----------
    - ovitraps_data (pd.DataFrame): DataFrame containing ovitraps data with columns\
        'ano', 'semepid', 'narmad', and 'novos'.

    Returns
    ----------
    - ovitraps_data (pd.DataFrame): Processed DataFrame ready for analysis.

    """
    # Step 1: Rename columns for consistency
    ovitraps_data.rename(
        columns={
            "semepi": "semepid",
        },
        inplace=True,
    )

    # At least two digits for semepid
    ovitraps_data["semepid"] = ovitraps_data["semepid"].apply(
        lambda x: f"{int(x):02d}" if pd.notnull(x) else x
    )

    return ovitraps_data


def get_biweekly_ovitraps(ovitraps_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the ovitraps_data DataFrame to create a matrix with
    "'anopid'W'semepid'" as index, 'narmad' as columns and 'novos' as
    values. Also handles the odd and even weeks by shifting odd weeks and
    combining them with even weeks and fill the missing values in index.


    Parameters
    ----------
    - ovitraps_data (pd.DataFrame): DataFrame containing ovitraps data with columns\
        'ano', 'semepid', 'narmad', and 'novos'.

    Returns
    ----------
    - pivot_data (pd.DataFrame): A pivoted DataFrame with 'ano' and 'semepid'
      as index, where each cell contains the count of 'novos' cases for each
      combination of 'ano', 'semepid', and 'narmad'.
    
    """
    # Rename and clean the DataFrame
    ovitraps_data = process_ovitraps(ovitraps_data)

    #  Convert week 53 to week 1 of the next year
    for row in ovitraps_data.itertuples():
        if row.semepid == "53":
            ovitraps_data.at[row.Index, "semepid"] = "01"
            year = int(row.anoepid[:4])
            ovitraps_data.at[row.Index, "anoepid"] = (
                f"{year + 1}_{year - 1999:02d}"
            )

    # Convert odd weeks to even weeks
    ovitraps_data["semepid"] = ovitraps_data["semepid"].apply(
        lambda x: str(int(x) + 1) if int(x) % 2 != 0 else x
    )

    # Create a new datetime column from ano and semepid
    ovitraps_data["date"] = ovitraps_data.apply(
        lambda row: str(row["anoepid"]) + "W" + str(row["semepid"]),
        axis=1,
    )

    # Pivot the DataFrame
    pivot_data = (
        ovitraps_data.groupby(["date", "narmad"])["novos"]
        .sum()
        .unstack()
        .sort_index(axis=1)
    )

    # Fill weeks
    all_weeks = generate_all_weeks(pivot_data)
    all_even_weeks = [
        week for week in all_weeks if int(week.split("W")[1]) % 2 == 0
    ]
    pivot_data = pivot_data.reindex(all_even_weeks).sort_index()

    return pivot_data


def get_weekly_ovitraps(ovitraps_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the ovitraps_data DataFrame to create a matrix with
    "'anopid'W'semepid'" as index, 'narmad' as columns and 'novos' as
    values. This function doesn't handle the odd and even weeks,
    it simply aggregates the data by week.


    Parameters
    ----------
    - ovitraps_data (pd.DataFrame): DataFrame containing ovitraps data with columns\
        'ano', 'semepid', 'narmad', and 'novos'.

    Returns
    ----------
    - pivot_data (pd.DataFrame): A pivoted DataFrame with 'ano' and 'semepid'
      as index, where each cell contains the count of 'novos' cases for each
      combination of 'ano', 'semepid', and 'narmad'.
    
    """
    # Rename and clean the DataFrame
    ovitraps_data = process_ovitraps(ovitraps_data)

    # Create a new datetime column from ano and semepid
    ovitraps_data["date"] = ovitraps_data.apply(
        lambda row: str(row["anoepid"]) + "W" + str(row["semepid"]),
        axis=1,
    )

    # Pivot the DataFrame
    pivot_data = (
        ovitraps_data.groupby(["date", "narmad"])["novos"]
        .sum()
        .unstack()
        .sort_index(axis=1)
    )

    # Fill weeks
    all_weeks = generate_all_weeks(pivot_data)
    pivot_data = pivot_data.reindex(all_weeks).sort_index()
    return pivot_data


################# Epidemiological Functions #################


def generate_all_weeks(pivot_data: pd.DataFrame) -> list:
    """
    Generate a list of week tuples based on the start and end dates
    in the pivoted DataFrame. This considers epidemic years and weeks
    between the range 1 and 53, formatted as '%YYYY_%YYW%WW'.

    Parameters
    ----------
    - pivot_data (pd.DataFrame): DataFrame with a datetime index.

    Returns
    ----------
    - new_tuples (list): List of week tuples in the format
      '%YYYY_%YYW%WW' for each week in the range from start to end
      date.

    """
    start_date = pivot_data.index.min()
    end_date = pivot_data.index.max()
    all_weeks_range = [f"{week:02d}" for week in range(1, 54)]
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    new_tuples = [
        f"{year}_{(year - 1999):02d}W{week}"
        for year in range(start_year, end_year + 1)
        for week in all_weeks_range
        if (f"{year}{week}" >= start_date and f"{year}{week}" <= end_date)
    ]

    return new_tuples
