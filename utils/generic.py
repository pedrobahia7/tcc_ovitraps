# import pygame
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.feature_selection import mutual_info_regression
import folium
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math
from scipy.spatial import cKDTree
from geopy.distance import distance
from geopy import Point
############# Music Functions :) #############


# def play_ending_song(
#     addr: str = "../data/Sinfonia To Cantata # 29.mp3",
# ) -> None:
#     """
#     Function to play a song when the program ends

#     Parameters:
#     addr (str): Address of the MP3 file

#     Returns:
#     None


#     """
#     # Initialize the mixer
#     pygame.mixer.init()
#     # Load the MP3 file
#     pygame.mixer.music.load(addr)
#     # Play the MP3 file
#     pygame.mixer.music.play()


# def stop_ending_song(seconds: int = 5) -> None:
#     """
#     Function to stop the song that is playing

#     Parameters:
#     seconds (int): Number of seconds to wait before stopping the song

#     Returns:
#     None
#     """

#     time.sleep(seconds)
#     pygame.mixer.music.stop()
#     # Optional: Clean up the mixer
#     pygame.mixer.quit()


############# Correlation Funtions #############


def calculate_correlation(
    series_1: pd.Series | np.ndarray,
    series_2: pd.Series | np.ndarray,
    method: str = "pearson",
) -> dict:
    """
    Calculate the Pearson correlation between two time series. The function
    normalizes the series and calculate the correlation according to the
    specified method. This function expects them to be non-stationary
    series of the same length. The function returns the value of the
    correlation. Only one value of lag is considered.

    Parameters
    ----------
    - series_1 (pd.Series): First time series.
    - series_2 (pd.Series): Second time series.
    - method (str): Method to calculate the correlation. Options are \
      "pearson".

    Returns
    ----------
    - results (float): Correlation value.

    """
    if len(series_1) != len(series_2):
        raise ValueError("Both series must have the same length.")

    if not isinstance(series_1, (pd.Series, np.ndarray)) or not isinstance(
        series_2, (pd.Series, np.ndarray)
    ):
        raise TypeError("Inputs must be pandas Series or numpy arrays.")

    # Ensure Series for .corr()
    series_1 = pd.Series(series_1)
    series_2 = pd.Series(series_2)

    # Drop NaNs
    cleaned = pd.concat([series_1, series_2], axis=1).dropna()
    if cleaned.empty:
        print("Both series are empty after dropping NaN values.")
        return np.nan

    series_1 = cleaned.iloc[:, 0]
    series_2 = cleaned.iloc[:, 1]

    if method == "pearson":
        return series_1.corr(series_2)

    elif method == "mutual_information":
        # Normalize
        series_1 = normalize_series(series_1, method="zscore")
        series_2 = normalize_series(series_2, method="zscore")

        return mutual_info_regression(
            series_1.values.reshape(-1, 1), series_2.values.reshape(-1, 1)
        )[0]

    else:
        raise ValueError(f"Method '{method}' not supported.")


def windowed_correlation(
    series_1: pd.Series,
    series_2: pd.Series,
    max_lag: int,
    window_size: int = None,
    last_valid_index: int = None,
    method: str = "pearson",
) -> dict:
    """
    Calculate the correlation between two time series and the lag
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

    # Ensure both series have the same index
    if not series_1.index.equals(series_2.index):
        if series_1.index.size > series_2.index.size:
            series_2 = series_2.reindex(series_1.index)
        else:
            series_1 = series_1.reindex(series_2.index)

    # Check if both series have at least one valid value
    if series_1.isnull().all() or series_2.isnull().all():
        raise ValueError(
            "Both series must have at least one valid value to calculate correlation."
        )

    # Get the last valid index if not provided
    if last_valid_index is None:
        last_valid_index = min(
            series_1.index.get_loc(series_1.last_valid_index()),
            series_2.index.get_loc(series_2.last_valid_index()),
        )

    # Use the smaller series length if window_size is not provided
    if window_size is None:
        window_size = min(len(series_1), len(series_2), last_valid_index)

    first_index = last_valid_index - window_size
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        # Roll the series to create a window of size 'window_size'
        if lag <= 0:
            # series_2 is shifted backward (leading series)
            windowed_series_1 = series_1[first_index:last_valid_index]
            windowed_series_2 = series_2.shift(lag)[
                first_index:last_valid_index
            ]

        elif lag > 0:
            # series_1 is shifted backward (leading series)
            windowed_series_1 = series_1.shift(-lag)[
                first_index:last_valid_index
            ]

            windowed_series_2 = series_2[first_index:last_valid_index]

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
    max_lag: int,
    last_valid_index: pd.Timestamp = None,
    window_size: int = None,
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
        series_1=series_1,
        series_2=series_2,
        max_lag=max_lag,
        last_valid_index=last_valid_index,
        window_size=window_size,
        method=method,
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
    max_lag: int,
    window_lengths: list = [None],
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
        lengths to be analyzed. Use the smaller series length if\
        window_size is not provided.
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

        # Confidence interval
        if size is None:
            size = min(len(series_1), len(series_2))
        ci = 1.96 / np.sqrt(size)

        plt.figure(figsize=(10, 5))
        plt.plot(lags, correlations, marker="o")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.fill_between(
            lags,
            -ci,
            ci,
            color="gray",
            alpha=0.2,
        )

        plt.grid()

        if size != window_lengths[-1]:
            plt.show()


def two_series_scatter_plot(
    series_1: pd.Series,
    series_2: pd.Series,
    series_2_shift: int,
    title: str = "",
    xlabel: str = "Series 1",
    ylabel: str = "Series 2 Shifted",
    xlim: tuple = None,
    ylim: tuple = None,
):
    series_2_shifted = series_2.shift(series_2_shift)
    if len(series_1) < len(series_2_shifted):
        series_2_shifted = series_2_shifted.reindex(series_1.index)
    if len(series_1) > len(series_2_shifted):
        series_1 = series_1.reindex(series_2_shifted.index)

    if series_1.isnull().all() or series_2_shifted.isnull().all():
        raise ValueError(
            "Both series must have at least one valid value to plot."
        )
    plt.figure(figsize=(10, 5))
    plt.scatter(series_1, series_2_shifted, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid()
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
    text: np.array = None,
    size: int = 3,
    color: str = "black",
) -> folium.map:
    """
    Adds black points to a folium map given a list of coordinates.

    Parameters
    ----------
    - mymap (folium.map): The folium map to which the points will be added.
    - coordinates (np.array): A 2D numpy array with shape (n, 2) where n is the number of points,
      and each row contains the latitude and longitude of a point.
    - size (int, optional): The size of the points to be added. Defaults to 3.
    - text (np.array | None, optional): An array of text labels for each point. Defaults to None.

    Returns
    -------
    - mymap (folium.map): The folium map with the added points.
    """
    # Add points to the map
    for n in range(coordinates.shape[0]):
        # Get the popup text for the marker
        if text is not None:
            popup_text = text[n]
        else:
            popup_text = ""

        # Add a custom marker for a very small dot
        folium.Marker(
            location=(coordinates[n, 0], coordinates[n, 1]),
            icon=folium.DivIcon(
                html=f'<div style="width: {size}px; height: {size}px; background-color: {color}; border-radius: 50%;"></div>'
            ),
            popup=popup_text,
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
    mymap: folium.map,
    cluster_data: pd.DataFrame,
    scale: float = 0.1,
) -> folium.map:
    """
    Add clustered markers to the folium map. It requires a DataFrame
    containing cluster name in column 'cluster', its coordinates in columns
    'latitude' and 'longitude', and the number of points in each cluster
    in column 'count'.

    Parameters
    ----------
    - mymap (folium.map): The folium map to which the clustered markers will be added.
    - cluster_data (pd.DataFrame): A DataFrame containing cluster information.

    Returns
    -------
    - mymap (folium.map): The folium map with the added clustered markers.

    """
    for _, row in cluster_data.iterrows():
        if row["cluster"] == -1:
            continue
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=np.sqrt(row["count"]) * scale,
            popup=f"Cluster {int(row['cluster'])}, Count: {row['count']}",
            color="blue",
            fill=True,
            fill_opacity=0.6,
        ).add_to(mymap)
    return mymap


##################### Geographical Functions #####################


def haversine_distance(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> float:
    """
    Function to calculate the distance between two points in the globe. It
    calculates the great-circle distance between two points on the Earth's
    surface using the Haversine formula, which assumes a perfectly
    spherical Earth. It introduces an error of about 0.5% in the distance
    calculation, which is negligible for most applications. This function
    is more computationally intensive than Planar Approximation
    (Equirectangular Projection with Pythagorean Theorem). For better
    performance, consider using the Vincenty's Formula, which is the most
    accurate method for calculating but also the slowest one.

    Parameters
    ----------
        latitude1 (float): Latitude of the first point in degrees.
        longitude1 (float): Longitude of the first point in degrees.
        latitude2 (float): Latitude of the second point in degrees.
        longitude2 (float): Longitude of the second point in degrees.

    Returns
        distance (float): Distance between the two points in kilometers.
    """

    EARTH_RADIUS = 6371  # Earth's mean radius in kilometers
    # Convert degrees to radians
    lat1_radians = math.radians(latitude1)
    lon1_radians = math.radians(longitude1)
    lat2_radians = math.radians(latitude2)
    lon2_radians = math.radians(longitude2)

    # Differences in coordinates
    dlat = lat2_radians - lat1_radians
    dlon = lon2_radians - lon1_radians

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_radians)
        * math.cos(lat2_radians)
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = EARTH_RADIUS * c
    return distance


def planar_approximation_distance(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> float:
    """
    Function to calculate the distance between two points in the globe. It
    calculates the distance between two points on the Earth's surface using
    the Equirectangular Projection with Pythagorean Theorem, which assumes
    the Earth to be locally flat. This method is generally the least
    accurate formula, with the error being proportional to the longitude
    and the distance between the two points. For small distances, the error
    is approximately 0.1%, what makes the method more precise than the
    Haversine formula. Furthermore, it is also the least computationally
    expensive method to calculate distances.

    Parameters
    ----------
        latitude1 (float): Latitude of the first point in degrees.
        longitude1 (float): Longitude of the first point in degrees.
        latitude2 (float): Latitude of the second point in degrees.
        longitude2 (float): Longitude of the second point in degrees.
    Returns
        distance (float): Distance between the two points in kilometers.
    """
    EARTH_RADIUS = 6371  # Earth's mean radius in kilometers
    # Convert degrees to radians
    lat1_radians = math.radians(latitude1)
    lon1_radians = math.radians(longitude1)
    lat2_radians = math.radians(latitude2)
    lon2_radians = math.radians(longitude2)

    # Differences in coordinates
    dlat = lat2_radians - lat1_radians
    dlon = lon2_radians - lon1_radians

    # Equirectangular projection
    x = dlon * math.cos((lat1_radians + lat2_radians) / 2)
    y = dlat

    distance = EARTH_RADIUS * math.sqrt(x**2 + y**2)
    return distance


def distance_between_points(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
    method: str,
) -> float:
    """
    Calculate the distance between two points on the Earth's surface using
    the chosen method.

    Parameters
    ----------
        - latitude1 (float): Latitude of the first point in degrees.
        - longitude1 (float): Longitude of the first point in degrees.
        - latitude2 (float): Latitude of the second point in degrees.
        - longitude2 (float): Longitude of the second point in degrees.
        - method (str): Method to calculate the distance. Options are
            "haversine" and "planar".

    Returns
    -------
        distance (float): Distance between the two points in kilometers.
    """
    if method == "haversine":
        return haversine_distance(
            latitude1,
            longitude1,
            latitude2,
            longitude2,
        )
    elif method == "planar":
        return planar_approximation_distance(
            latitude1,
            longitude1,
            latitude2,
            longitude2,
        )
    else:
        raise ValueError(f"Method '{method}' not supported.")


def smaller_distance_in_df(
    latitude: float,
    longitude: float,
    df: pd.DataFrame,
    method: str,
) -> pd.Series:
    """
    Find the row in the DataFrame with the smallest distance to the given point.

    Parameters
    ----------
        - latitude (float): Latitude of the point.
        - longitude (float): Longitude of the point.
        - df (pd.DataFrame): DataFrame with the points.
        - method (str): Method to calculate the distance. Options are
            "haversine" and "planar".

    Returns
    -------
        pd.Series: Row with the smallest distance.
    """
    distances = df.apply(
        lambda row: distance_between_points(
            latitude,
            longitude,
            row["latitude"],
            row["longitude"],
            method=method,
        ),
        axis=1,
    )
    return df.loc[distances.idxmin()]


def nearest_neighbors(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    For each point in points1, find the closest point in points2.
    Returns indices of nearest points in points2. This function uses
    KDTree for efficient nearest neighbor search and is suitable for
    large datasets and latitude/longitude coordinates for small distances 
    (within a city).

    Parameters
    ----------
    - points1 (np.ndarray): Array of shape (n, d) with n points (latitude, longitude).
    - points2 (np.ndarray): Array of shape (m, d) with m points (latitude, longitude).

    Returns
    -------
    - indices (np.ndarray): Array of shape (n,) with indices of nearest points in points2.
    
    """
    # Input validation
    def check_array(arr,name):
        
        assert isinstance(arr, np.ndarray), f"{name} must be a numpy array"
        assert not arr.size == 0, f"{name} must not be empty"
        assert arr.ndim == 2, f"{name} must be 2D arrays"
        assert np.issubdtype(arr.dtype, np.number), f"{name} must contain numeric values"
        assert not np.issubdtype(arr.dtype, np.bool_), f"{name} must not be boolean"
        assert not np.isnan(arr).any(), f"{name} must not contain NaN values"
        assert not np.isinf(arr).any(), f"{name} must not contain infinity values"
        
    check_array(points1,"points1")
    check_array(points2,"points2")
    assert points1.shape[1] == points2.shape[1], "points1 and points2 must have the same number of dimensions"

    # Build KDTree for points2 and query for nearest neighbors
    tree = cKDTree(points2)
    _, indices = tree.query(points1, k=1)

    # Ouptut validation
    assert isinstance(indices, np.ndarray), "Output must be a numpy array"
    assert indices.ndim == 1, "Output must be a 1D array"
    assert indices.shape[0] == points1.shape[0], "Output length must match points1 length"
    assert np.issubdtype(indices.dtype, np.integer), "Output must contain integer values"
    assert not np.isnan(indices).any(), "Output must not contain NaN values"
    assert not np.isinf(indices).any(), "Output must not contain infinity values"

    return indices


def create_grid(lat_min: float, lat_max: float, lon_min: float, lon_max: float, spacing_m: float) -> pd.DataFrame:
    """
    Create a grid of points in lat/lon space with approximately equal spacing in meters.
    
    Parameters:
    - lat_min, lat_max: latitude bounds
    - lon_min, lon_max: longitude bounds
    - spacing_m: distance between points in meters
    
    Returns:
    - DataFrame with columns ['latitude', 'longitude']
    """
    # Input validation
    def check_lat_lon(lat, lon, name):
        assert isinstance(lat, (int, float)), f"{name} latitude must be a number"
        assert isinstance(lon, (int, float)), f"{name} longitude must be a number"
        assert not isinstance(lat, bool), f"{name} latitude must not be boolean"
        assert not isinstance(lon, bool), f"{name} longitude must not be boolean"
        assert -90 <= lat <= 90, f"{name} latitude must be between -90 and 90"
        assert -180 <= lon <= 180, f"{name} longitude must be between -180 and 180"
        assert not np.isnan(lat), f"{name} latitude must not be NaN"
        assert not np.isnan(lon), f"{name} longitude must not be NaN"
        assert not np.isinf(lat), f"{name} latitude must not be infinity"
        assert not np.isinf(lon), f"{name} longitude must not be infinity"

    check_lat_lon(lat_min, lon_min, "Minimum")
    check_lat_lon(lat_max, lon_max, "Maximum")

    assert lat_min < lat_max, "lat_min must be less than lat_max"
    assert lon_min < lon_max, "lon_min must be less than lon_max"
    assert isinstance(spacing_m, (int, float)), "spacing_m must be a number"
    assert spacing_m > 0, "spacing_m must be positive"
    assert not isinstance(spacing_m, bool), "spacing_m must not be boolean"
    assert not np.isnan(spacing_m), "spacing_m must not be NaN"
    assert not np.isinf(spacing_m), "spacing_m must not be infinity"
    

    # Create grid points
    points = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            points.append((lat, lon))
            # Move east by spacing_m
            origin = Point(lat, lon)
            lon = distance(meters=spacing_m).destination(origin, bearing=90).longitude
        # Move north by spacing_m
        origin = Point(lat, lon_min)
        lat = distance(meters=spacing_m).destination(origin, bearing=0).latitude
    df = pd.DataFrame(points, columns=["latitude", "longitude"])

    # Output validation
    assert not df.empty, "Output DataFrame must not be empty"
    assert list(df.columns) == ["latitude", "longitude"], "Output DataFrame must have columns ['latitude', 'longitude']"
    assert np.issubdtype(df['latitude'].dtype, np.number), "Latitude column must contain numeric values"
    assert np.issubdtype(df['longitude'].dtype, np.number), "Longitude column must contain numeric values"
    assert not df.isnull().values.any(), "Output DataFrame must not contain NaN values"
    assert all(df['latitude'].between(lat_min, lat_max)), "Latitude values must be between -90 and 90"
    assert all(df['longitude'].between(lon_min, lon_max)), "Longitude values must be between -180 and 180"
    assert df['latitude'].diff().abs().max() <= spacing_m, "Latitude spacing must be consistent with spacing_m"
    assert df['longitude'].diff().abs().max() <= spacing_m, "Longitude spacing must be consistent with spacing_m"

    return df

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


######## Series Manipulation Functions ################


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


def group_series(
    series: pd.Series,
    offset: int,
    group_size: int,
    operation: str,
) -> pd.Series:
    """
    Group series with a specified group size. An offset is applied to
    alignment reasons. The type of operation to be performed on each group
    (e.g., "sum", "mean") is specified by the user.

    Parameters
    ----------
    - series (pd.Series): The Series to be grouped.
    - offset (int): The offset to be applied for alignment.
    - group_size (int): The size of each group.
    - operation (str): The operation to be performed on each group.

    Returns
    -------
    - pd.Series: The grouped Series.

    """
    group_ids = (pd.RangeIndex(len(series)) - offset) // group_size
    if operation == "sum":
        grouped = series.groupby(group_ids).sum()
    elif operation == "mean":
        grouped = series.groupby(group_ids).mean()
    else:
        raise ValueError(f"Invalid operation: {operation}")
    grouped.index = series.groupby(group_ids).apply(lambda x: x.index[0])
    return grouped


################# Plot Functions #################
def text_above_plot(plot_df, text_df):
    if isinstance(plot_df, pd.Series):
        plot_df = plot_df.to_frame().T

    # Add text above each box
    for i, col in enumerate(plot_df.columns):
        if plot_df[col].isna().all():
            continue
        plt.text(
            i,  # x = box position
            plot_df[col].max(),  # y = max value of that box
            text_df[col],
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="darkred",
        )
