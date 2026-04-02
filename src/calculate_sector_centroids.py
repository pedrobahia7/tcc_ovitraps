"""
Script to calculate centroids of all census sectors in Belo Horizonte.

This script reads the census sectors GeoJSON file and calculates the geometric
centroid for each sector. The centroids can be used for visualization and
spatial analysis purposes. It also calculates IDW interpolated ovitrap egg
values for each sector based on the 6 nearest ovitraps.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist


def calculate_sector_centroids(
    geojson_path: str,
    output_path: str = None,
    ) -> pd.DataFrame:
    """
    Calculate the centroids of all regions in a GeoJSON file.

    Parameters
    ----------
    geojson_path : str
        Path to the input GeoJSON file containing sector geometries.
    output_path : str, optional
        Path to save the output CSV file with centroids. If None, file is
        saved in the same directory as the input with '_centroids.csv' suffix.

    Returns
    -------
    pd.DataFrame
        DataFrame containing sector codes, centroid coordinates (latitude,
        longitude), and other relevant attributes.

    Examples
    --------
    >>> centroids_df = calculate_sector_centroids(
    ...     'data/processed/bh_sectors_2022_with_populations.geojson'
    ... )
    >>> centroids_df.head()
    """
    # Input validation
    assert isinstance(geojson_path, str), "geojson_path must be a string"
    assert Path(geojson_path).exists(), f"File not found: {geojson_path}"
    assert geojson_path.endswith('.geojson'), "Input must be a GeoJSON file"

    # Load the GeoJSON file
    gdf = gpd.read_file(geojson_path)

    # Store original CRS
    original_crs = gdf.crs

    # Project to UTM (SIRGAS 2000 / UTM zone 23S - EPSG:31983) for accurate centroid calculation
    gdf_projected = gdf.to_crs(epsg=31983)

    # Calculate centroids in projected CRS
    centroids_projected = gdf_projected.geometry.centroid

    # Project centroids back to geographic coordinates (WGS84)
    centroids_geo = centroids_projected.to_crs(original_crs)

    # Extract latitude and longitude from centroids
    gdf['centroid_longitude'] = centroids_geo.x
    gdf['centroid_latitude'] = centroids_geo.y

    # Create output DataFrame with relevant columns
    output_columns = [
        'CD_SETOR',
        'SITUACAO',
        'NM_BAIRRO',
        'NM_SUBDIST',
        'AREA_KM2',
        'pop_2010',
        'pop_2022',
        'centroid_latitude',
        'centroid_longitude'
    ]

    # Filter to only include columns that exist in the GeoDataFrame
    available_columns = [col for col in output_columns if col in gdf.columns]
    centroids_df = gdf[available_columns].copy()

    # Sort by sector code
    centroids_df = centroids_df.sort_values('CD_SETOR').reset_index(drop=True)

    # Determine output path
    if output_path is None:
        input_path = Path(geojson_path)
        output_path = input_path.parent / f"{input_path.stem}_centroids.csv"

    # Save to CSV
    centroids_df.to_csv(output_path, index=False)

    # Output validation
    assert isinstance(centroids_df, pd.DataFrame), "Output must be a DataFrame"
    assert len(centroids_df) > 0, "Output DataFrame must not be empty"
    assert 'centroid_latitude' in centroids_df.columns, "Missing centroid_latitude"
    assert 'centroid_longitude' in centroids_df.columns, "Missing centroid_longitude"
    assert centroids_df['centroid_latitude'].notnull().all(), "Latitude contains null values"
    assert centroids_df['centroid_longitude'].notnull().all(), "Longitude contains null values"

    return centroids_df


def calculate_idw_for_centroids(
    centroids_df: pd.DataFrame,
    ovitraps_path: str,
    n_neighbors: int = 6,
    power: float = 2.0,
    output_path: str = None,
    ) -> pd.DataFrame:
    """
    Calculate IDW (Inverse Distance Weighting) interpolated egg values for
    each centroid based on the nearest ovitraps for each epidemic date.

    Parameters
    ----------
    centroids_df : pd.DataFrame
        DataFrame containing centroids with 'centroid_latitude' and
        'centroid_longitude' columns.
    ovitraps_path : str
        Path to the ovitraps data CSV file.
    n_neighbors : int, optional
        Number of nearest neighbors to use for IDW interpolation (default: 6).
    power : float, optional
        Power parameter for IDW (default: 2.0).
    output_path : str, optional
        Path to save the output CSV file with IDW values. If None, file is
        saved as 'sector_centroids_with_idw.csv' in the processed directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: CD_SETOR, epidemic_date, idw_egg_value,
        along with centroid coordinates.

    Examples
    --------
    >>> idw_df = calculate_idw_for_centroids(
    ...     centroids_df,
    ...     'data/processed/ovitraps_data.csv'
    ... )
    """
    # Input validation
    assert isinstance(centroids_df, pd.DataFrame), "centroids_df must be a DataFrame"
    assert 'centroid_latitude' in centroids_df.columns, "Missing centroid_latitude"
    assert 'centroid_longitude' in centroids_df.columns, "Missing centroid_longitude"
    assert isinstance(ovitraps_path, str), "ovitraps_path must be a string"
    assert Path(ovitraps_path).exists(), f"File not found: {ovitraps_path}"
    assert n_neighbors > 0, "n_neighbors must be positive"
    assert power > 0, "power must be positive"

    # Load ovitraps data
    print(f"\nLoading ovitraps data: {ovitraps_path}")
    eggs = pd.read_csv(ovitraps_path)

    # Create biweek grouping: pair odd weeks with following even weeks
    # Extract year and week from epidemic_date format (e.g., "2024_25W43")
    eggs['epi_year'] = eggs['epidemic_date'].str.split('W').str[0]
    eggs['week_num'] = eggs['epidemic_date'].str.split('W').str[1].astype(int)
    
    # Calculate biweek: pair odd weeks with following even weeks
    # Week 1-2 → biweek 2, Week 3-4 → biweek 4, etc.
    eggs['biweek_num'] = ((eggs['week_num'] + 1) // 2) * 2
    eggs['biweek'] = eggs['epi_year'] + 'W' + eggs['biweek_num'].astype(str).str.zfill(2)

    # Get centroid coordinates and sector codes
    centroid_coords = centroids_df[['centroid_latitude', 'centroid_longitude']].values
    sector_codes = centroids_df['CD_SETOR'].values
    
    # Get unique biweeks
    biweeks = sorted(eggs['biweek'].unique())
    print(f"Calculating IDW for {len(centroids_df)} centroids across {len(biweeks)} biweeks...")

    # Prepare result list
    results = []

    # Process each biweek
    for idx, biweek in enumerate(biweeks):
        if (idx + 1) % 10 == 0:
            print(f"  Processing biweek {idx + 1}/{len(biweeks)}...")
            
        # Get egg data for this biweek (includes both odd and even weeks)
        week_data = eggs[eggs['biweek'] == biweek]
        
        if len(week_data) == 0:
            continue

        # Check for duplicate traps and sum their values
        week_data = week_data.groupby(['narmad', 'latitude', 'longitude'], as_index=False)['novos'].sum()


        # Get trap coordinates and values (vectorized)
        trap_coords = week_data[['latitude', 'longitude']].values
        trap_egg_values = week_data['novos'].values
        trap_narmads = week_data['narmad'].values

        # Calculate distances from all centroids to all traps (single call)
        distances = cdist(centroid_coords, trap_coords, metric='euclidean')

        # Vectorized IDW calculation for all centroids
        if len(trap_coords) <= n_neighbors:
            # Use all traps if we have fewer than n_neighbors
            nearest_indices_all = np.tile(np.arange(len(trap_coords)), (len(centroids_df), 1))
        else:
            # Get k nearest neighbors for each centroid (vectorized)
            nearest_indices_all = np.argpartition(distances, n_neighbors, axis=1)[:, :n_neighbors]

        # Process each centroid (still need loop for narmads string building)
        for i in range(len(centroids_df)):
            nearest_indices = nearest_indices_all[i]
            nearest_distances = distances[i, nearest_indices]
            nearest_values = trap_egg_values[nearest_indices]
            nearest_narmads = trap_narmads[nearest_indices]

            # Calculate IDW weights
            if np.any(nearest_distances == 0):
                zero_idx = np.where(nearest_distances == 0)[0][0]
                idw_value = nearest_values[zero_idx]
            else:
                weights = 1 / (nearest_distances ** power)
                idw_value = np.sum(weights * nearest_values) / np.sum(weights)

            # Store result
            results.append({
                'CD_SETOR': sector_codes[i],
                'biweek': biweek,
                'idw_egg_value': idw_value,
                'centroid_latitude': centroid_coords[i, 0],
                'centroid_longitude': centroid_coords[i, 1],
                'n_traps_used': len(nearest_indices),
                'min_distance_km': np.min(nearest_distances) * 111.32,
                'narmads_used': ','.join(str(n) for n in nearest_narmads)
            })

    # Create output DataFrame
    idw_df = pd.DataFrame(results)

    # Sort by sector code and biweek
    idw_df = idw_df.sort_values(['CD_SETOR', 'biweek']).reset_index(drop=True)

    # Determine output path
    if output_path is None:
        output_path = Path("data/processed/sector_centroids_with_idw.csv")

    # Save to CSV
    print(f"\nSaving IDW results to: {output_path}")
    idw_df.to_csv(output_path, index=False)

    # Display summary statistics
    print(f"\nSummary:")
    print(f"Total records: {len(idw_df)}")
    print(f"Unique sectors: {idw_df['CD_SETOR'].nunique()}")
    print(f"Unique biweeks: {idw_df['biweek'].nunique()}")
    print(f"IDW egg value range: [{idw_df['idw_egg_value'].min():.2f}, "
          f"{idw_df['idw_egg_value'].max():.2f}]")
    print(f"Mean IDW egg value: {idw_df['idw_egg_value'].mean():.2f}")
    print(f"\nFirst few records:")
    print(idw_df.head(10))

    # Output validation
    assert isinstance(idw_df, pd.DataFrame), "Output must be a DataFrame"
    assert len(idw_df) > 0, "Output DataFrame must not be empty"
    assert 'idw_egg_value' in idw_df.columns, "Missing idw_egg_value"
    assert idw_df['idw_egg_value'].notnull().all(), "IDW values contain nulls"

    return idw_df


if __name__ == "__main__":
    # Define paths
    GEOJSON_PATH = "data/processed/bh_sectors_2022_with_populations.geojson"
    CENTROIDS_OUTPUT_PATH = "data/processed/bh_sectors_2022_centroids.csv"
    OVITRAPS_PATH = "data/processed/ovitraps_data.csv"
    IDW_OUTPUT_PATH = "data/processed/sector_centroids_with_idw.csv"

    # Calculate centroids
    print("Step 1: Calculating sector centroids...")
    centroids_df = calculate_sector_centroids(
        geojson_path=GEOJSON_PATH,
        output_path=CENTROIDS_OUTPUT_PATH
    )

    # Calculate IDW values
    print("\nStep 2: Calculating IDW egg values for centroids...")
    idw_df = calculate_idw_for_centroids(
        centroids_df=centroids_df,
        ovitraps_path=OVITRAPS_PATH,
        n_neighbors=6,
        power=2.0,
        output_path=IDW_OUTPUT_PATH
    )

    print("\n✓ All processing completed successfully!")

