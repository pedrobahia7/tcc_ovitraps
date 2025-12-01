#!/usr/bin/env python3
"""
Add Population Sector IDs to Ovitraps and Dengue Data

This script reads the processed ovitraps and dengue data files, performs spatial joins
with the census sectors shapefile to identify which census sector each data point
belongs to, and adds a 'population_sector' column containing the sector ID.

Input files:
- data/processed/ovitraps_data.csv
- data/processed/dengue_data.csv  
- data/processed/bh_sectors_2022_with_populations.geojson

Output files:
- data/processed/ovitraps_data.csv (updated with population_sector column)
- data/processed/dengue_data.csv (updated with population_sector column)

Author: Census Equivalence Processing System
Date: December 2025
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import yaml
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def add_sector_ids_to_dataset(df: pd.DataFrame, sectors_gdf: gpd.GeoDataFrame, 
                             dataset_name: str) -> pd.DataFrame:
    """
    Add population_sector column to a dataset using spatial join.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with latitude and longitude columns
    sectors_gdf : gpd.GeoDataFrame
        Census sectors with geometry
    dataset_name : str
        Name of dataset for logging
        
    Returns:
    --------
    pd.DataFrame
        Dataset with population_sector column added
    """
    print(f"\n🔄 Processing {dataset_name} data...")
    
    # Check if required columns exist
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print(f"   ⚠️ Warning: {dataset_name} data missing latitude/longitude columns")
        df['population_sector'] = None
        return df
    
    # Filter records with valid coordinates
    valid_coords_mask = (
        df['latitude'].notna() & 
        df['longitude'].notna() &
        (df['latitude'] != 0) & 
        (df['longitude'] != 0)
    )
    
    valid_count = valid_coords_mask.sum()
    total_count = len(df)
    
    print(f"   📊 Records with valid coordinates: {valid_count:,} / {total_count:,} ({valid_count/total_count*100:.1f}%)")
    
    if valid_count == 0:
        print(f"   ⚠️ No valid coordinates found in {dataset_name}")
        df['population_sector'] = None
        return df
    
    # Create points for valid coordinates
    valid_df = df[valid_coords_mask].copy()
    geometry = [Point(lon, lat) for lon, lat in zip(valid_df['longitude'], valid_df['latitude'])]
    
    # Create GeoDataFrame
    points_gdf = gpd.GeoDataFrame(valid_df, geometry=geometry, crs='EPSG:4326')
    
    # Ensure both have same CRS
    if sectors_gdf.crs != points_gdf.crs:
        print(f"   🔄 Reprojecting sectors from {sectors_gdf.crs} to {points_gdf.crs}")
        sectors_gdf = sectors_gdf.to_crs(points_gdf.crs)
    
    # Perform spatial join
    print(f"   🎯 Performing spatial join...")
    joined = gpd.sjoin(points_gdf, sectors_gdf[['sector_id', 'geometry']], 
                      how='left', predicate='within')
    
    # Count matches
    matched_count = joined['sector_id'].notna().sum()
    print(f"   ✅ Spatial matches: {matched_count:,} / {valid_count:,} ({matched_count/valid_count*100:.1f}%)")
    
    # Initialize population_sector column for all records
    df['population_sector'] = None
    
    # Update matched records
    if matched_count > 0:
        # Get the sector_id from the joined result
        sector_mapping = joined['sector_id'].dropna()
        df.loc[valid_coords_mask, 'population_sector'] = joined['sector_id'].values
        
        # Show sample matches
        sample_matches = df[df['population_sector'].notna()]['population_sector'].head(5)
        print(f"   📋 Sample sector IDs: {list(sample_matches)}")
    
    return df


def main():
    """Main processing function"""
    print("🏥 ADDING POPULATION SECTOR IDS")
    print("=" * 50)
    
    # Load parameters
    params = load_params()
    
    # Define file paths - read from intermediate location
    ovitraps_input_path = Path(params['all']['paths']['data']['processed']['add_population_sectors']['ovitraps'])
    dengue_input_path = Path(params['all']['paths']['data']['processed']['add_population_sectors']['dengue'])
    
    # Output to final location
    ovitraps_output_path = Path(params['all']['paths']['data']['processed']['ovitraps'])
    dengue_output_path = Path(params['all']['paths']['data']['processed']['dengue'])
    
    sectors_path = Path("data/processed/bh_sectors_2022_with_populations.geojson")
    
    print(f"📂 Input files:")
    print(f"   • Ovitraps: {ovitraps_input_path}")
    print(f"   • Dengue: {dengue_input_path}")
    print(f"   • Sectors: {sectors_path}")
    
    print(f"📂 Output files:")
    print(f"   • Ovitraps: {ovitraps_output_path}")
    print(f"   • Dengue: {dengue_output_path}")
    
    # Check files exist
    missing_files = []
    for name, path in [("Ovitraps", ovitraps_input_path), ("Dengue", dengue_input_path), ("Sectors", sectors_path)]:
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file in missing_files:
            print(f"   • {file}")
        print("   Please ensure all input files exist before running this stage.")
        return
    
    print(f"\n✅ All input files found")
    
    # Load census sectors
    print(f"\n🗺️ Loading census sectors...")
    try:
        sectors_gdf = gpd.read_file(sectors_path)
        print(f"   ✅ Loaded {len(sectors_gdf):,} census sectors")
        print(f"   📋 CRS: {sectors_gdf.crs}")
        
        # Show available columns
        print(f"   📋 Available columns: {list(sectors_gdf.columns)}")
        
        # Check for sector_id column
        if 'sector_id' not in sectors_gdf.columns:
            print(f"   ⚠️ 'sector_id' column not found, checking alternatives...")
            
            # Look for sector ID columns
            id_columns = [col for col in sectors_gdf.columns 
                         if any(x in col.lower() for x in ['sector', 'setor', 'cd_', 'geocod'])]
            
            if id_columns:
                # Use the first likely column
                id_col = id_columns[0]
                sectors_gdf['sector_id'] = sectors_gdf[id_col]
                print(f"   🔄 Using '{id_col}' as sector_id")
            else:
                print(f"   ❌ No sector ID column found in shapefile")
                return
                
    except Exception as e:
        print(f"   ❌ Error loading sectors: {e}")
        return
    
    # Load and process ovitraps data
    print(f"\n📊 Loading ovitraps data...")
    try:
        ovitraps_df = pd.read_csv(ovitraps_input_path)
        print(f"   ✅ Loaded {len(ovitraps_df):,} ovitrap records")
        
        # Add population sectors
        ovitraps_df = add_sector_ids_to_dataset(ovitraps_df, sectors_gdf, "ovitraps")
        
        # Save updated file to final location
        ovitraps_df.to_csv(ovitraps_output_path, index=False)
        print(f"   💾 Updated ovitraps data saved to {ovitraps_output_path}")
        
    except Exception as e:
        print(f"   ❌ Error processing ovitraps data: {e}")
        return
    
    # Load and process dengue data
    print(f"\n🦠 Loading dengue data...")
    try:
        dengue_df = pd.read_csv(dengue_input_path)
        print(f"   ✅ Loaded {len(dengue_df):,} dengue records")
        
        # Add population sectors
        dengue_df = add_sector_ids_to_dataset(dengue_df, sectors_gdf, "dengue")
        
        # Save updated file to final location
        dengue_df.to_csv(dengue_output_path, index=False)
        print(f"   💾 Updated dengue data saved to {dengue_output_path}")
        
    except Exception as e:
        print(f"   ❌ Error processing dengue data: {e}")
        return
    
    # Summary
    print(f"\n📈 PROCESSING SUMMARY")
    print(f"=" * 30)
    
    ovitraps_with_sectors = ovitraps_df['population_sector'].notna().sum()
    dengue_with_sectors = dengue_df['population_sector'].notna().sum()
    
    print(f"✅ Ovitraps with sector IDs: {ovitraps_with_sectors:,} / {len(ovitraps_df):,} ({ovitraps_with_sectors/len(ovitraps_df)*100:.1f}%)")
    print(f"✅ Dengue cases with sector IDs: {dengue_with_sectors:,} / {len(dengue_df):,} ({dengue_with_sectors/len(dengue_df)*100:.1f}%)")
    
    if ovitraps_with_sectors > 0 or dengue_with_sectors > 0:
        print(f"\n🎉 Population sector assignment completed successfully!")
        print(f"📊 Both datasets now include 'population_sector' column")
        print(f"🔗 Use this column to link with population data from sector_population_comparison.csv")
    else:
        print(f"\n⚠️ No spatial matches found. Please check:")
        print(f"   • Coordinate systems compatibility")
        print(f"   • Geographic coverage overlap")
        print(f"   • Data quality of coordinates")


if __name__ == "__main__":
    main()