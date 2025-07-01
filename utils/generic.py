import pygame
import time
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.feature_selection import mutual_info_regression
import folium
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

############# Music Functions :) #############


def play_ending_song(
    addr: str = "../data/Sinfonia To Cantata # 29.mp3",
) -> None:
    """
    Function to play a song when the program ends

    Parameters:
    addr (str): Address of the MP3 file

    Returns:
    None


    """
    # Initialize the mixer
    pygame.mixer.init()
    # Load the MP3 file
    pygame.mixer.music.load(addr)
    # Play the MP3 file
    pygame.mixer.music.play()


def stop_ending_song(seconds: int = 5) -> None:
    """
    Function to stop the song that is playing

    Parameters:
    seconds (int): Number of seconds to wait before stopping the song

    Returns:
    None
    """

    time.sleep(seconds)
    pygame.mixer.music.stop()
    # Optional: Clean up the mixer
    pygame.mixer.quit()


############# Correlation Funtions #############


def calculate_correlation(
    series_1: pd.Series | np.ndarray,
    series_2: pd.Series | np.ndarray,
    method: str = "pearson",
) -> dict:
    """
    Calculate the Pearson correlation between two time series with a given
    maximum lag. The function normalizes the series and calculate the
    correlation according to the specified method. This function expects
    them to be non-stationary series of the same length. The function
    returns the value of the correlation. Only one value of lag is
    considered.

    Parameters
    ----------
    - series_1 (pd.Series): First time series.
    - series_2 (pd.Series): Second time series.
    - method (str): Method to calculate the correlation. Options are \
      "pearson".

    Returns
    ----------
    - results (dict): Dictionary with the correlation values for each lag.

    """

    if len(series_1) != len(series_2):
        raise ValueError(
            "The two series must have the same length to calculate\
                  correlation."
        )

    # Check if the series are pandas Series or numpy arrays
    if (
        type(series_1) is not pd.Series
        and type(series_1) is not np.ndarray
    ) or (
        type(series_2) is not pd.Series
        and type(series_2) is not np.ndarray
    ):
        raise TypeError(
            "Both series must be pandas Series or numpy arrays to\
calculate correlation."
        )

    # Normalize the series
    series_1 = normalize_series(series_1, method="zscore")
    series_2 = normalize_series(series_2, method="zscore")

    # Drop NaN values from both series and align them
    combined = pd.concat([series_1, series_2], axis=1)
    cleaned = combined.dropna(axis=0, how="any")
    if cleaned.empty:
        print("Both series are empty after dropping NaN values.")
        return np.nan
    series_1 = cleaned.iloc[:, 0]
    series_2 = cleaned.iloc[:, 1]

    # Calculate the correlation based on the specified method
    if method == "pearson":
        x_corr = series_1.corr(series_2)
    elif method == "mutual_information":
        x_corr = mutual_info_regression(
            np.array(series_1).reshape(-1, 1),
            np.array(series_2).reshape(-1, 1),
        )[0]

    else:
        raise ValueError(
            f"Method {method} not supported. Only 'pearson' is available."
        )

    return x_corr


def windowed_correlation(
    series_1: pd.Series,
    series_2: pd.Series,
    max_lag: int,
    window_size: int = None,
    last_valid_index: int = None,
    method: str = "pearson",
) -> dict:
    """
    Calculate the maximum correlation between two time series and the lag
    at which it occurs. This function creates a window of size
    'window_size' and move this window from the 'last_valid_index' sample
    backwards 'max_lag' times. The series_1 window is shifted backward by
    one time step at each iteration. This process happens 'max_lag' times.
    The same is done for series_2. In this way, positive lags indicate that
    series_1 leads series_2, while negative lags indicate that series_2
    leads series_1. If any of the series has NaN values, they are dropped
    before calculating the correlation. 

    Parameters
    ----------
    - series_1 (pd.Series): First time series.
    - series_2 (pd.Series): Second time series.
    - max_lag (int): Maximum lag in days to consider in the correlation.\
      The lags are tested in both directions, leading to 2* max_lag + 1\
      lags in total.
    - window_size (int): Size in days of the window to calculate the\
      correlation. Default is None, which means the entire series is used.
    - last_valid_index (timestamp): The last valid index of the series to \
      consider for the correlation calculation. This is typically the last\
      index that has valid data for both series. This function is format\
      agnostic. Default is None, which means the last index with valid\
      values of both series is used.
    - method (str): Method to calculate the correlation. Options are \
      "pearson".

    Returns
    ----------
    - lag (int): The lag at which the  correlation occurs.
    - corr (float): The correlation value.

    """
    # Drop both indexes to ensure they are aligned
    series_1.reset_index(drop=True, inplace=True)
    series_2.reset_index(drop=True, inplace=True)

    # Get the last valid index if not provided
    if last_valid_index is None:
        last_valid_index = min(
            series_1.last_valid_index(), series_2.last_valid_index()
        )

    # Use the smaller series length if window_size is not provided
    if window_size is None:
        window_size = min(len(series_1), len(series_2), last_valid_index)

    results = {}
    for lag in range(-max_lag, max_lag + 1):
        # Roll the series to create a window of size 'window_size'
        if lag < 0:
            # series_2 is shifted backward (leading series)
            windowed_series_1 = series_1[
                last_valid_index - window_size : last_valid_index
            ]

            windowed_series_2 = series_2[
                max(
                    0, last_valid_index - window_size + lag
                ) : last_valid_index + lag
            ]

        else:
            # series_1 is shifted backward (leading series)
            windowed_series_1 = series_1[
                max(
                    last_valid_index - window_size - lag, 0
                ) : last_valid_index - lag
            ]

            windowed_series_2 = series_2[
                last_valid_index - window_size : last_valid_index
            ]

        # Ensure both series are of the same length. Since the window is
        # moving backward, it may reach the beginning of the series
        # before the other series, leading to different lengths.
        min_length = min(len(windowed_series_1), len(windowed_series_2))
        windowed_series_1 = windowed_series_1[:min_length]
        windowed_series_2 = windowed_series_2[:min_length]

        # Calculate the correlation
        corr = calculate_correlation(
            windowed_series_1,
            windowed_series_2,
            method=method,
        )

        # Store the result
        results[lag] = corr

    return results


def max_correlation(
    series_1: pd.Series,
    series_2: pd.Series,
    last_valid_index: pd.Timestamp,
    window_size: int,
    max_lag: int,
    method: str = "pearson",
) -> Tuple[int, float]:
    """
    Calculate the maximum correlation between two time series and the lag
    at which it occurs. This function creates a window of size
    'window_size' and move this window from the last sample 'max_lag'
    backward. The series_1 window is shifted backward by one time step at
    each iteration. This process happens 'max_lag' times. The same is done
    for series_2. In this way, positive lags indicate that series_1 leads
    series_2, while negative lags indicate that series_2 leads series_1. If
    any of the series has NaN values, they are dropped before calculating
    the correlation. The function returns the lag at which the maximum
    absolute correlation occurs.

    Parameters
    ----------
    - series_1 (pd.Series): First time series.
    - series_2 (pd.Series): Second time series.
    - last_valid_index (pd.Timestamp): The last valid index of the series\
      to consider for the correlation calculation. This is typically the\
      last index that has valid data for both series. Use the format\
      'YYYY-MM-DD'
    - window_size (int): Size in days of the window to calculate the\
      correlation.
    - max_lag (int): Maximum lag to consider for the correlation in days.
    - method (str): Method to calculate the correlation. Options are \
      "pearson".

    Returns
    ----------
    - best_lag (int): The lag at which the maximum absolute correlation\
      occurs.
    - max_corr (float): The maximum absolute correlation value at the best\
      lag.

    """
    results = windowed_correlation(
        series_1,
        series_2,
        last_valid_index,
        window_size,
        max_lag,
        method,
    )

    abs_results = {
        k: abs(v) for k, v in results.items() if not np.isnan(v)
    }

    best_lag = max(abs_results, key=abs_results.get)
    max_corr = results[best_lag]
    return best_lag, max_corr


def plot_cross_correlation(
    series_1: pd.Series,
    series_2: pd.Series,
    window_lengths: list,
    max_lag: int,
    last_valid_index: pd.Timestamp = None,
    method: str = "pearson",
    title: str = "Cross-Correlation",
    x_label: str = "Lag (days)",
    y_label: str = "Correlation",
) -> None:
    """
    Plot the cross-correlation between two time series.
    The function calculates the cross-correlation for lags from -max_lag
    to +max_lag and plots the results.

    Parameters
    ----------
    - series_1 (pd.Series): First time series.
    - series_2 (pd.Series): Second time series.
    - window_lengths (list): A list of integers representing all window\
        lengths to be analyzed.
    - max_lag (int): The maximum lag to consider for each analysis.
    - last_valid_index (pd.Timestamp): The last valid index of the series\
      to consider for the correlation calculation. This is typically the\
      last index that has valid data for both series. Use the format\
      'YYYY-MM-DD'
    - method (str): The correlation method to use. Options are "pearson"\
      and "mutual_info".
    - title (str): Title of the plot.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.

    Returns
    ----------
    - None

    """
    # If last_valid_index is not provided, use the last valid index of the
    # both series
    if last_valid_index is None:
        last_valid_index = min(series_1.index.max(), series_2.index.max())

    for size in window_lengths:
        # Calculate the cross-correlation
        results_dict = windowed_correlation(
            series_1=series_1,
            series_2=series_2,
            last_valid_index=last_valid_index,
            window_size=size,
            max_lag=max_lag,
            method=method,
        )

        # Extract lags and correlations
        lags = list(results_dict.keys())
        correlations = list(results_dict.values())

        plt.figure(figsize=(10, 5))
        plt.plot(lags, correlations, marker="o")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()
        if size != window_lengths[-1]:
            plt.show()


################### Map Functions ###################


def create_map(coordinates: np.array, title=None) -> folium.map:
    """
    Creates a folium map centered around the mean of the given coordinates.

    Parameters
    ----------
    - coordinates (np.array): A 2D numpy array with shape (n, 2) where n is\
      the number of points, and each row contains the latitude and\
      longitude of a point.
    - title (str, optional): Title to be displayed on the map. Defaults to None.

    Returns
    -------
    - mymap (folium.map): A folium map object centered at the mean of
        the coordinates with a zoom level of 11.

    """

    map_center = np.nanmean(coordinates, axis=0)
    mymap = folium.Map(location=map_center, zoom_start=11)
    if title is not None:
        title_html = f"""<h3 align="center" style="font-size:20px"><b>{title}</b></h3>"""
        mymap.get_root().html.add_child(folium.Element(title_html))
    return mymap


def add_points_to_map(
    mymap: folium.map,
    coordinates: np.array,
) -> folium.map:
    """
    Adds black points to a folium map given a list of coordinates.

    Parameters
    ----------
    - mymap (folium.map): The folium map to which the points will be added.
    - coordinates (np.array): A 2D numpy array with shape (n, 2) where n is the number of points,
      and each row contains the latitude and longitude of a point.

    Returns
    -------
    - mymap (folium.map): The folium map with the added points.
    """
    # Add points to the map
    for point in coordinates:
        # Add a custom marker for a very small dot
        folium.Marker(
            location=(point[0], point[1]),
            icon=folium.DivIcon(
                html='<div style="width: 3px; height: 3px; background-color: black; border-radius: 50%;"></div>'
            ),
        ).add_to(mymap)
    return mymap


def cluster_points(
    coordinates: np.array,
    eps_km: float = 0.05,
    min_samples: int = 5,
):
    """
    Clusters points using DBSCAN with haversine distance.

    Parameters
    ----------
    - coordinates (np.array): A 2D numpy array with shape (n, 2) where n is the number of points,
        and each row contains the latitude and longitude of a point.
    - eps_km (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns
    -------
    - cluster_data (pd.DataFrame): A DataFrame containing the cluster centers and the number of points in each cluster.

    """

    # DBSCAN with haversine expects radians
    coordinates_rad = np.radians(coordinates)
    eps_rad = eps_km / 6371.0
    db = DBSCAN(
        eps=eps_rad, min_samples=min_samples, metric="haversine"
    ).fit(coordinates_rad)

    df = pd.DataFrame(coordinates, columns=["latitude", "longitude"])
    df["cluster"] = db.labels_

    # # Get cluster centers (optional)
    # cluster_centers = (
    #     df.groupby("cluster")[["latitude", "longitude"]]
    #     .mean()
    #     .reset_index()
    # )

    # Count points per cluster
    cluster_counts = df.groupby("cluster").size().reset_index(name="count")

    # Get cluster centers
    cluster_centers = (
        df.groupby("cluster")[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )

    cluster_data = cluster_centers.merge(cluster_counts, on="cluster")

    return cluster_data


def add_clustered_markers_to_map(
    mymap: folium.map, cluster_data: pd.DataFrame
) -> folium.map:
    # Add clustered markers
    for _, row in cluster_data.iterrows():
        if row["cluster"] == -1:
            continue
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=row["count"] ** 0.3,  # scale size
            popup=f"Cluster {int(row['cluster'])}, Count: {row['count']}",
            color="blue",
            fill=True,
            fill_opacity=0.6,
        ).add_to(mymap)
    return mymap


##################### OS Functions #####################


def save_file(file: str, folder: str, file_name: str) -> None:
    """
    Saves a file to a specified folder with a given file name. If the folder
    does not exist, it creates the folder.

    Parameters
    ----------
    - file (str): The name of the file to be saved.
    - folder (str): The folder where the file will be saved.
    - file_name (str): The name of the file to be saved.

    Returns
    -------
    None

    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    file.save(os.path.join(folder, file_name))


################## Date Functions ##################


def convert_iso_year_week_to_string(date_iso: str) -> str:
    """
    Converts an ISO year-week string to a string in the format 'YYYYWW'.

    Parameters
    ----------
    - date_iso (str): The ISO year-week string in the format '%YYYYW%ww'\
      or '%YYYY-W%w'.

    Returns
    -------
    - str_date (str): The converted string in the format 'YYYYWW'.
    """
    year, week = date_iso.split("W")
    week = int(week)  # convert week to int
    str_date = f"{year}{week:02d}"
    return str_date


######## Series Manipulation Functions ########


def normalize_series(
    series: pd.Series,
    method: str = "zscore",
) -> pd.Series:
    """
    Normalize a pandas Series by subtracting the mean and dividing by the
    standard deviation. This is useful for standardizing data before
    analysis or modeling.

    Parameters
    ----------
    - series (pd.Series): The Series to be normalized.
    - method (str): The normalization method to be used. Options are \
        "zscore", "minmax", and "norm".

    Returns
    ----------
    - series (pd.Series): The normalized Series.

    """
    if method == "zscore":
        return (series - series.mean()) / series.std()
    elif method == "minmax":
        return (series - series.min()) / (series.max() - series.min())
    elif method == "norm":
        return series / np.linalg.norm(series.dropna())
    else:
        raise ValueError(f"Unknown normalization method: {method}")
