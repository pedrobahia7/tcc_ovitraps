"""
Dengue Cases Dashboard - Belo Horizonte

This dashboard visualizes dengue cases on a map of Belo Horizonte with:
- Interactive map showing up to 2000 dengue cases as dots
- Hover information displaying date and population sector
- Clickable census sectors showing sector IDs
- Sector boundaries overlay

Author: Dashboard System
Date: December 2025
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, callback
from dash.exceptions import PreventUpdate

# =============================================================================
# CONSTANTS
# =============================================================================

# File paths
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DENGUE_CSV = PROCESSED_DIR / "dengue_data.csv"
GEOJSON_2022 = PROCESSED_DIR / "bh_sectors_2022_with_populations.geojson"

# Map center (Belo Horizonte)
MAP_CENTER_LAT = -19.9167
MAP_CENTER_LON = -43.9345
MAP_ZOOM = 11

# Map style configuration
MAPBOX_STYLE = "carto-positron"

# Maximum number of cases to display
MAX_CASES_DISPLAY = 2000

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


def load_dengue_data() -> pd.DataFrame:
    """
    Load dengue cases data from CSV file.

    Returns:
        DataFrame with dengue cases including latitude, longitude, date, population_sector
    """
    print("🦠 Loading dengue cases data...")
    df = pd.read_csv(DENGUE_CSV)
    
    # Filter records with valid coordinates
    valid_coords_mask = (
        df['latitude'].notna() & 
        df['longitude'].notna() &
        (df['latitude'] != 0) & 
        (df['longitude'] != 0)
    )
    
    df_valid = df[valid_coords_mask].copy()
    print(f"   ✅ Loaded {len(df_valid):,} dengue cases with valid coordinates (from {len(df):,} total)")
    
    # Limit to MAX_CASES_DISPLAY for performance
    if len(df_valid) > MAX_CASES_DISPLAY:
        df_valid = df_valid.sample(n=MAX_CASES_DISPLAY, random_state=42)
        print(f"   📊 Sampled {MAX_CASES_DISPLAY:,} cases for display")
    
    return df_valid


def load_sectors_geojson() -> gpd.GeoDataFrame:
    """
    Load 2022 census sectors GeoJSON.

    Returns:
        GeoDataFrame with 2022 sector geometries
    """
    print("🗺️ Loading census sectors GeoJSON...")
    gdf = gpd.read_file(GEOJSON_2022)
    
    # Identify sector ID column
    if 'sector_id' in gdf.columns:
        gdf['CD_SETOR'] = gdf['sector_id'].astype(str)
    elif 'CD_SETOR' not in gdf.columns:
        # Find likely ID column
        id_columns = [col for col in gdf.columns 
                     if any(x in col.lower() for x in ['sector', 'setor', 'cd_', 'geocod'])]
        if id_columns:
            gdf['CD_SETOR'] = gdf[id_columns[0]].astype(str)
    else:
        gdf['CD_SETOR'] = gdf['CD_SETOR'].astype(str)
    
    print(f"   ✅ Loaded {len(gdf):,} sectors")
    return gdf


# =============================================================================
# MAP HELPER FUNCTIONS
# =============================================================================


def create_sectors_trace(gdf: gpd.GeoDataFrame) -> go.Choroplethmapbox:
    """
    Create a trace for census sectors with clickable areas.

    Args:
        gdf: GeoDataFrame with geometries and sector IDs

    Returns:
        Choroplethmapbox trace with clickable sectors
    """
    if gdf.empty:
        return None
    
    # Convert to GeoJSON
    geojson = json.loads(gdf.to_json())
    
    locations = []
    hover_texts = []

    for _, row in gdf.iterrows():
        sector_id = row['CD_SETOR']
        locations.append(sector_id)
        hover_texts.append(f"Sector: {sector_id}<br>Click to see sector ID")
    
    # Use uniform z values for uniform coloring
    z_values = [0] * len(locations)

    trace = go.Choroplethmapbox(
        geojson=geojson,
        locations=locations,
        z=z_values,
        featureidkey="properties.CD_SETOR",
        colorscale=[[0, 'rgba(200, 200, 200, 0.2)'], [1, 'rgba(200, 200, 200, 0.2)']],
        zmin=0,
        zmax=1,
        marker={'opacity': 0.3, 'line': {'width': 0.8, 'color': 'gray'}},
        hovertext=hover_texts,
        hoverinfo='text',
        showscale=False,
        name='Census Sectors'
    )
    
    return trace


def create_dengue_cases_trace(df: pd.DataFrame) -> go.Scattermapbox:
    """
    Create a trace for dengue cases as dots.

    Args:
        df: DataFrame with dengue cases

    Returns:
        Scattermapbox trace with dengue case dots
    """
    if df.empty:
        return None
    
    # Prepare hover text
    hover_texts = []
    for _, row in df.iterrows():
        date_str = row.get('dt_notific', 'Unknown')
        sector = row.get('population_sector', 'Unknown')
        hover_text = f"Date: {date_str}<br>Sector: {sector}"
        hover_texts.append(hover_text)
    
    trace = go.Scattermapbox(
        lon=df['longitude'].tolist(),
        lat=df['latitude'].tolist(),
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            opacity=0.6
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        name='Dengue Cases'
    )
    
    return trace


# =============================================================================
# MAIN DATA LOADING
# =============================================================================

print("\n" + "="*80)
print("🦠 LOADING DATA FOR DENGUE CASES DASHBOARD")
print("="*80)

# Load all data
dengue_df = load_dengue_data()
sectors_gdf = load_sectors_geojson()

print("\n" + "="*80)
print("✅ DATA LOADING COMPLETE")
print("="*80)
print(f"🦠 Dengue cases: {len(dengue_df):,}")
print(f"🗺️ Census sectors: {len(sectors_gdf):,}")
print("="*80 + "\n")

# =============================================================================
# DASH APP SETUP
# =============================================================================

app = Dash(__name__)
app.title = "Dengue Cases Dashboard - Belo Horizonte"

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = html.Div([
    html.H1("Dengue Cases Dashboard - Belo Horizonte",
            style={'textAlign': 'center', 'color': '#E74C3C', 'marginBottom': 20}),
    
    html.Div([
        html.P(f"Displaying {len(dengue_df):,} dengue cases. Hover over dots to see date and sector. Click on sectors to see sector ID.",
               style={'textAlign': 'center', 'color': '#666', 'fontSize': 14, 'marginBottom': 10}),
    ]),
    
    # Main map
    html.Div([
        dcc.Graph(id='dengue-map', style={'height': '800px'}),
    ]),
    
    # Selected sector info
    html.Div(id='sector-info', style={
        'textAlign': 'center',
        'padding': '20px',
        'fontSize': '18px',
        'fontWeight': 'bold',
        'color': '#2E86C1',
        'minHeight': '60px'
    })
    
], style={'padding': 20})

# =============================================================================
# CALLBACKS
# =============================================================================


@app.callback(
    Output('dengue-map', 'figure'),
    Input('dengue-map', 'id'),
    prevent_initial_call=False
)
def create_map(trigger):
    """
    Create the dengue cases map.

    Returns:
        Plotly figure with sectors and dengue cases
    """
    fig = go.Figure()
    
    # Add sectors layer
    sectors_trace = create_sectors_trace(sectors_gdf)
    if sectors_trace:
        fig.add_trace(sectors_trace)
    
    # Add dengue cases layer
    dengue_trace = create_dengue_cases_trace(dengue_df)
    if dengue_trace:
        fig.add_trace(dengue_trace)
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style=MAPBOX_STYLE,
            center=dict(lat=MAP_CENTER_LAT, lon=MAP_CENTER_LON),
            zoom=MAP_ZOOM,
            pitch=0,
            bearing=0
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig


@app.callback(
    Output('sector-info', 'children'),
    Input('dengue-map', 'clickData'),
    prevent_initial_call=True
)
def display_sector_info(clickData):
    """
    Display sector ID when a sector is clicked.

    Args:
        clickData: Click event data from the map

    Returns:
        HTML displaying the clicked sector ID
    """
    if not clickData:
        raise PreventUpdate
    
    # Check if the click was on a sector (not a dengue case dot)
    point = clickData['points'][0]
    
    # Sectors are from the choropleth trace (trace 0), cases are from scatter (trace 1)
    if point.get('curveNumber') == 0:  # Clicked on sector
        sector_id = point.get('location', 'Unknown')
        return f"Selected Sector ID: {sector_id}"
    else:
        return "Click on a sector (not a case dot) to see its ID"


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 STARTING DENGUE CASES DASHBOARD")
    print("="*80)
    print("📍 Open your browser and navigate to: http://127.0.0.1:8051/")
    print("="*80 + "\n")

    app.run(debug=True, port=8051)
