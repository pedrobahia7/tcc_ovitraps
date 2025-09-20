import pandas as pd
import numpy as np
import generic
import matplotlib.pyplot as plt

from pyproj import Transformer


EPIDEMY_YEARS = ["2012_13", "2015_16", "2018_19", "2023_24"]


################ Dengue Cases Functions ################


def get_weekly_dengue(
    dengue_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert processed dengue data to a pivoted format suitable for comparison
    with ovitrap data, with 'epidemic_date' as index. The
    DataFrame will contain counts of 'novos' cases.

    Parameters
    ----------
    - data (pd.DataFrame): DataFrame containing dengue data with columns\
        'anoepid', 'semepid', and 'novos'.

    Returns
    ----------
    - weekly_data (pd.DataFrame): A pivoted DataFrame with "'ano'W'semepid'"
      as index, where each cell contains the count of 'novos' cases in
      each week.

    Doctest
    --------
    >>> week_data = get_weekly_dengue(dengue_data)
    >>> week_data['2007_08W01'] == ((dengue_data['anoepid'] == '2007_08') & (dengue_data['semepid'] == 1)).sum()
    """
    dengue_data = dengue_data.copy()

    # Create a new datetime column from ano and semepid
    dengue_data["epidemic_date"] = get_epidemic_date(dengue_data)

    # Pivot the DataFrame to create a matrix with 'epidemic_date' as index
    pivot_data = (
        dengue_data.groupby(["epidemic_date"]).size().reset_index()
    )

    pivot_data.set_index(["epidemic_date"], inplace=True)

    # Generate new MultiIndex with all weeks
    all_weeks_index = generate_all_weeks(pivot_data)

    # Combine with existing index and reindex the DataFrame
    weekly_data = pivot_data.reindex(all_weeks_index).sort_index()

    # Fill NaN values with 0
    weekly_data.replace(np.nan, 0, inplace=True)

    # Convert df to Series
    weekly_data = weekly_data[0]

    return weekly_data


def get_biweekly_dengue(
    dengue_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert processed dengue data to a pivoted format suitable for comparison
    with ovitrap data, with 'anoepid' and 'semepid' as index. The
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
    # Get weekly dengue data
    weekly_data = get_weekly_dengue(dengue_data)

    # Convert odd weeks to even weeks
    dengue_data["semepid"] = dengue_data["semepid"].apply(
        lambda x: int(int(x) + 1) if int(x) % 2 != 0 else x
    )
    raise NotImplementedError(
        "Biweekly dengue data generation not implemented"
    )
    biweekly_data = 1
    return biweekly_data


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
    dengue_data = dengue_data.copy()

    # Get daily counts of dengue cases
    dengue_cases_serie = (
        dengue_data.groupby("dt_notific").size().reset_index()
    )

    # Reset index and transform to Series
    dengue_cases_serie.set_index("dt_notific", inplace=True)
    dengue_cases_serie = dengue_cases_serie[0]
    dengue_cases_serie.index = pd.to_datetime(dengue_cases_serie.index)

    # Fill missing values with 0
    dengue_cases_serie = dengue_cases_serie.reindex(
        pd.date_range(
            dengue_cases_serie.index.min(),
            dengue_cases_serie.index.max(),
            freq="D",
        ),
        fill_value=0,
    )

    return dengue_cases_serie


################ Ovitraps Eggs Functions ################


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
    #  Copy dataframe
    ovitraps_data = ovitraps_data.copy()

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
        lambda x: int(int(x) + 1) if int(x) % 2 != 0 else x
    )

    # Create a new datetime column from ano and semepid
    ovitraps_data["date"] = ovitraps_data.apply(
        lambda row: str(row["anoepid"]) + "W" + str(row["semepid"])
        if row["semepid"] > 9
        else str(row["anoepid"]) + "W0" + str(row["semepid"]),
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
    ovitraps_data = ovitraps_data.copy()

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


def get_daily_ovitraps(
    ovitraps_data: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Convert raw ovitraps data to a daily samples by the mean of eggs in a
    sample over the whole time period the trap was installed. 

    Parameters
    ----------
    - ovitraps_data (pd.DataFrame): DataFrame containing ovitraps data with columns\
        'dt_instal', 'dt_col', 'narmad', and 'novos'.

    Returns
    ----------
    - daily_ovitraps (pd.DataFrame): A DataFrame indexed by date, with 'narmad'
      as columns and the mean of 'novos' cases per day.

    """
    # Input validation
    assert isinstance(ovitraps_data, pd.DataFrame), "Input must be a DataFrame"
    assert ovitraps_data.empty is False, "Input DataFrame must not be empty"
    assert all(
        col in ovitraps_data.columns
        for col in ["dt_instal", "dt_col", "narmad", "novos"]
    ), "DataFrame must contain 'dt_instal', 'dt_col', 'narmad', and 'novos' columns"
    
    assert pd.api.types.is_datetime64_any_dtype(
        ovitraps_data["dt_instal"]
    ), "'dt_instal' column must be of datetime type"
    
    assert pd.api.types.is_datetime64_any_dtype(
        ovitraps_data["dt_col"]
    ), "'dt_col' column must be of datetime type"
    
    assert pd.api.types.is_numeric_dtype(
        ovitraps_data["novos"].dropna(),
    ), "'novos' column must be numeric or NaN"
    
    assert (
        ovitraps_data["dt_col"] >= ovitraps_data["dt_instal"]
    ).all(), "'dt_col' must be greater than or equal to 'dt_instal'"
    
    assert ovitraps_data["narmad"].notnull().all(), "'narmad' must not contain null values"
    assert ovitraps_data["novos"].notnull().all(), "'novos' must not contain null values"
    assert (
        ovitraps_data["novos"] >= 0
    ).all(), "'novos' must be non-negative"


    ovitraps_data = ovitraps_data.copy()

    # Get daily counts of ovitraps cases
    expanded_rows = ovitraps_data.apply(
        lambda row: pd.DataFrame(
            {
                "date": pd.date_range(row["dt_instal"], row["dt_col"]),
                "narmad": row["narmad"],
                "novos": row["novos"]
                / (len(pd.date_range(row["dt_instal"], row["dt_col"])) - 1),
            }
        ),
        axis=1,
    )

    ovitraps_expanded = pd.concat(
        expanded_rows.tolist(), ignore_index=True
    )

    # Group by date and narmad, summing the 'novos' values
    daily_ovitraps = (
        ovitraps_expanded.groupby(["date", "narmad"])["novos"]
        .sum()
        .unstack()
        .sort_index(axis=1)
    )

    # Fill missing values with NaN
    daily_ovitraps = daily_ovitraps.reindex(
        pd.date_range(
            daily_ovitraps.index.min(),
            daily_ovitraps.index.max(),
            freq="D",
        ),
        fill_value=np.nan,
    )

    # Output Validation
    assert isinstance(daily_ovitraps, pd.DataFrame)
    assert daily_ovitraps.empty is False
    assert all(
        col in daily_ovitraps.columns for col in ovitraps_data["narmad"].unique()
    ), "Output DataFrame must contain all 'narmad' columns from input"
    
    assert pd.api.types.is_datetime64_any_dtype(
        daily_ovitraps.index
    ), "Output DataFrame index must be of datetime type"

    assert (all([pd.api.types.is_numeric_dtype(daily_ovitraps[col].dropna())
            for col in daily_ovitraps.columns]),
            "Output DataFrame must contain numeric values")

    assert (all([((daily_ovitraps[col].dropna() >= 0).all())
            for col in daily_ovitraps.columns]),
            "Output DataFrame must contain non-negative values")

    assert(daily_ovitraps.index.min() == ovitraps_data["dt_instal"].min(),
        "Output DataFrame index min must match input 'dt_instal' min")
    
    assert(daily_ovitraps.index.max() == ovitraps_data["dt_col"].max(),
        "Output DataFrame index max must match input 'dt_col' max")

    return daily_ovitraps

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


def get_epidemic_years_date_ranges_dengue(
    dengue_data: pd.DataFrame,
    ) -> dict:
    """
    Get the date ranges for each epidemic year based on the dengue data.

    Parameters
    ----------
    - dengue_data (pd.DataFrame): Data containing the 'anoepid' and 'dt_notific' columns.

    Returns
    -------
    - dict: A dictionary where the keys are epidemic years and the values are date ranges.
    """
    day_anoepid = {}
    for year in dengue_data.anoepid.unique():
        year_data = dengue_data[dengue_data["anoepid"] == year]
        day_anoepid[year] = pd.date_range(
            start=year_data["dt_notific"].min(),
            end=year_data["dt_notific"].max(),
        )
    return day_anoepid


def get_epidemic_years_date_ranges_ovitraps(  # DO NOT USE THIS. DATES IN OVITRAPS ARE NOT RELIABLE
    ovitrap_data: pd.DataFrame,
    ) -> dict:
    """
    Get the date ranges for each epidemic year based on ovitraps data.

    Parameters
    ----------
    - ovitrap_data (pd.DataFrame): Data containing the 'anoepid' and 'dt_col' columns.

    Returns
    -------
    - dict: A dictionary where the keys are epidemic years and the values are date ranges.
    """
    day_anoepid = {}
    for year in ovitrap_data.anoepid.unique():
        year_data = ovitrap_data[ovitrap_data["anoepid"] == year]
        day_anoepid[year] = pd.date_range(
            start=year_data["dt_col"].min(),
            end=year_data["dt_col"].max(),
        )
    return day_anoepid


def get_epidemic_date(data: pd.DataFrame) -> pd.Series:
    """
    Function to convert epidemic week and year into a single string of the
    format {anoepid}W{semepid} with two digits for the week.

    Parameters
    ----------
    - data (pd.DataFrame): DataFrame containing columns for the epidemic
      week and year.

    Returns
    -------
    epidemic_date (pd.Series): Series containing the formatted epidemic
    dates.
    """

    epidemic_date = data.apply(
        lambda row: str(row["anoepid"]) + "W" + str(row["semepid"])
        if int(row["semepid"]) > 9
        else str(row["anoepid"]) + "W0" + str(row["semepid"]),
        axis=1,
    )
    return epidemic_date


def assign_epidemic_year(df: pd.DataFrame, date_col: str) -> pd.Series:
    """
    Assign epidemic year to each date in the DataFrame column. The epidemic year
    starts on the first Sunday before June 1st.

    Parameters
    ----------
    df (pd.DataFrame) - DataFrame containing the date column.
    date_col (str) - Name of the date column in the DataFrame.

    Returns
    -------
    pd.Series
        Series containing the assigned epidemic years.
    """
    dates = pd.to_datetime(df[date_col])

    # Determine preliminary year
    year = np.where(dates.dt.month >= 6, dates.dt.year, dates.dt.year - 1)
    year = pd.Series(year, index=df.index)

    # Compute first Sunday on or before June 1
    june_first = pd.to_datetime(year.astype(str) + "-06-01")
    # weekday: Monday=0 ... Sunday=6
    offset = (june_first.dt.weekday + 1) % 7
    epi_year_start = june_first - pd.to_timedelta(offset, unit="D")

    # If date >= epi_year_start and < next year's epi_year_start → same epidemic year
    next_june_first = pd.to_datetime((year + 1).astype(str) + "-06-01")
    next_offset = (next_june_first.dt.weekday + 1) % 7
    next_epi_year_start = next_june_first - pd.to_timedelta(
        next_offset, unit="D"
    )

    # If date >= next year start, increment the epidemic year
    epi_year = year.copy()
    epi_year[dates >= next_epi_year_start] += 1

    # Create label "YYYY_YY"
    return epi_year.astype(str) + "_" + (epi_year + 1).astype(str).str[-2:]


def is_same_week_as_june1(dates: pd.Series) -> pd.Series:
    """
    Check if each date falls within the same week as June 1 of that year.

    Parameters
    ----------
    dates (pd.Series): Series of dates to check.

    Returns
    -------
    pd.Series: Boolean Series indicating if each date is in the same week as June 1.

    """
    # Convert to datetime
    dates = pd.to_datetime(dates)

    # Year corresponding to each date
    year = dates.dt.year

    # June 1 of each date's year
    june1 = pd.to_datetime(year.astype(str) + "-06-01")

    # First Sunday on or before June 1
    offset = (june1.dt.weekday + 1) % 7  # Monday=0 ... Sunday=6
    first_sunday = june1 - pd.to_timedelta(offset, unit="D")

    # Check if each date falls within that week (Sunday → Saturday)
    return (dates >= first_sunday) & (
        dates <= first_sunday + pd.Timedelta(days=6)
    )


def assign_epidemic_week(df: pd.DataFrame, date_col: str) -> pd.Series:
    """
    Assign epidemic week to each date in the DataFrame column. The epidemic
    week changes every Sunday, and the count starts every year on the first
    Sunday before June 1st. Some years may have 53 weeks.

    Parameters
    ----------
    df (pd.DataFrame) - DataFrame containing the date column.
    date_col (str) - Name of the date column in the DataFrame.

    Returns
    -------
    pd.Series - Series containing the assigned epidemic weeks (1-53).

    """

    dates = pd.to_datetime(df[date_col])

    # Determine epidemic year start: first Sunday on or before June 1 of that date's year
    year = np.where(dates.dt.month >= 6, dates.dt.year, dates.dt.year - 1)
    year = pd.Series(year, index=df.index)

    june_first = pd.to_datetime(year.astype(str) + "-06-01")
    offset = (june_first.dt.weekday + 1) % 7  # Sunday=6
    epi_year_start = june_first - pd.to_timedelta(offset, unit="D")

    # Epidemic week = number of Sundays since epidemic year start + 1, modulo 53
    week_num = ((dates - epi_year_start).dt.days // 7) + 1

    week_num[is_same_week_as_june1(dates)] = 1
    return week_num


def week_days_of_year(year: int) -> pd.DataFrame:
    """
    Generate a DataFrame containing all days of each week for a given year.

    Parameters
    ----------
    - year (int): The year for which to generate the week days.

    Returns
    -------
    - pd.DataFrame: A DataFrame with columns for each day of the week and rows
    representing each week of the year.
    """

    sundays = pd.date_range(
        start=f"{year}-01-01", end=f"{year}-12-31", freq="W-SUN"
    )
    data = {
        "Sunday": sundays,
        "Monday": sundays + pd.Timedelta(days=1),
        "Tuesday": sundays + pd.Timedelta(days=2),
        "Wednesday": sundays + pd.Timedelta(days=3),
        "Thursday": sundays + pd.Timedelta(days=4),
        "Friday": sundays + pd.Timedelta(days=5),
        "Saturday": sundays + pd.Timedelta(days=6),
    }
    return pd.DataFrame(data)


def convert_week_df_to_epidemic_week_and_year(
    df_week: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a DataFrame with week-days based columns to two DataFrames: one
    with epidemic weeks and another with epidemic years.

    Parameters
    ----------
    - df_week (pd.DataFrame): DataFrame with week-days based columns.

    Returns
    -------
    - df_epi_week (pd.DataFrame): DataFrame with epidemic weeks.
    - df_epi_year (pd.DataFrame): DataFrame with epidemic years.


    """
    df_epi_week = df_week.copy()
    for col in df_week.columns:
        df_epi_week[col] = assign_epidemic_week(df_week, col)
    df_epi_year = df_week.copy()
    for col in df_week.columns:
        df_epi_year[col] = assign_epidemic_year(df_week, col)
    return df_epi_week, df_epi_year


################# Geographical Functions ###################


def closest_health_center(df, health_centers, method="haversine"):
    """
    Find the closest health center to each point in the DataFrame.

    Parameters
    ----------
    - df (pd.DataFrame): DataFrame with the points.
    - health_centers (pd.DataFrame): DataFrame with health center locations and names.
    - method (str): Method to calculate the distance. Options are
        "haversine" and "planar".

    Returns
    -------
    - pd.DataFrame: DataFrame with the closest health center for each point.

    """
    # Calculate the distance between each point and each health center
    closest_health_center_list = []
    for _, row in df.iterrows():
        if row[["latitude", "longitude"]].notnull().all():
            closest_health_center_list.append(
                generic.smaller_distance_in_df(
                    row["latitude"],
                    row["longitude"],
                    health_centers,
                    method=method,
                )["health_center"]
            )
        else:
            closest_health_center_list.append(np.nan)

    return closest_health_center_list


def convert_qgis_to_latlon(df, x_col="coordx", y_col="coordy"):
    """
    Convert projected coordinates in a DataFrame to latitude/longitude.

    Parameters:
        df (pd.DataFrame): DataFrame containing coordx, coordy
        x_col (str): column name for easting
        y_col (str): column name for northing

    Returns:
        pd.DataFrame with extra columns 'latitude' and 'longitude'
    """
    # Define Belo Horizonte GTZ
    transformer = Transformer.from_crs(
        "EPSG:31983", "EPSG:4326", always_xy=True
    )

    lon, lat = transformer.transform(df[x_col].values, df[y_col].values)
    df = df.copy()
    df["latitude"] = lat
    df["longitude"] = lon
    df.loc[df["coordx"] == 0.0, "latitude"] = np.nan
    df.loc[df["coordy"] == 0.0, "longitude"] = np.nan
    return df


############### Visualization Functions #######################


def boxplot_filtered_data(
    df_to_plot,
    df_filter,
    lower_limit=range(0, 100, 10),
    upper_limit=range(10, 110, 10),
    title="Boxplot of Filtered Data",
    truncate_plot=True,
    truncation_limit=1000,
    ):
    """
    Function to create boxplots for separate groups of samples of
    df_to_plot, according to the values of df_filter. Both dataframes must
    have the same index, so the index of df_filter values inside the given
    range will be used to separate df_to_plot into different groups.

    Parameters
    ----------
    - df_to_plot (pd.DataFrame): DataFrame to plot be ploted. Each box will
      represent a group of values from df_filter.
    - df_filter (pd.Series): Series used to filter df_to_plot. It's values
      will define groups for the boxplots.
    - lower_limit (range): Lower limits for each boxplot group. Must be an
      iterable of the same length as upper_limit.
    - upper_limit (range): Upper limits for each boxplot group. Must be an
      iterable of the same length as lower_limit.
    - title (str): Title of the plot.
    - truncate_plot (bool): Whether plot a new graph with values cliped on
      the truncation_limit.
    - truncation_limit (int): The value at which to truncate the plot.

    Returns
    -------
    None
    """
    filtered_series = []
    # Filter df_to_plot according to df_filter values
    for down, up in zip(lower_limit, upper_limit):
        if up == list(upper_limit)[-1]:
            up = np.inf
        filter_index = (
            df_filter[(df_filter < up) & (df_filter >= down)]
            .dropna()
            .index
        )
        filtered_series.append(
            pd.Series(
                df_to_plot.loc[
                    df_to_plot.index.intersection(filter_index)
                ].dropna(),
                name=f"{down}-{up}",
            )
        )

    filtered_df_epidemy = pd.DataFrame(filtered_series).T
    cat_samples = filtered_df_epidemy.notna().sum()

    # Plot
    plt.figure(figsize=(12, 6))
    # Add the text inside the plot (adjust x, y as needed)
    ax = filtered_df_epidemy.boxplot()

    # Add text above each box
    for i, col in enumerate(filtered_df_epidemy.columns, start=1):
        if filtered_df_epidemy[col].isna().all():
            continue
        ax.text(
            i,  # x = box position
            filtered_df_epidemy[col].max(),  # y = max value of that box
            str(cat_samples[col]),  # text = count
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="darkred",
        )
    plt.ylabel("Dengue Cases")
    plt.xlabel("Ovitraps Mean")
    plt.title(title)
    plt.show()

    if truncate_plot:
        plt.figure(figsize=(12, 6))
        # Add the text inside the plot (adjust x, y as needed)
        ax = filtered_df_epidemy.clip(upper=truncation_limit).boxplot()
        # Add text above each box
        for i, col in enumerate(filtered_df_epidemy.columns, start=1):
            if filtered_df_epidemy[col].isna().all():
                continue
            ax.text(
                i,  # x = box position
                filtered_df_epidemy[col]
                .clip(upper=truncation_limit)
                .max(),  # y = max value of that box
                str(cat_samples[col]),  # text = count
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="darkred",
            )

        plt.title(f"{title} (truncated at {truncation_limit})")
        plt.ylabel("Dengue Cases")
        plt.xlabel("Ovitraps Mean")
        plt.show()
