"""
Interactive dashboard to visualize census sector centroids on a map.

This dashboard displays the census sectors of Belo Horizonte with their
centroids marked as small dots on an interactive map using Dash and Plotly.
Includes ovitraps locations with a slider to filter by epidemic date.
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import geopandas as gpd
import pandas as pd

# Map configuration
MAP_CENTER_LAT = -19.9167
MAP_CENTER_LON = -43.9345
MAP_ZOOM = 11

# Load data
print("Loading census sectors data...")
sectors_gdf = gpd.read_file(
    "data/processed/bh_sectors_2022_with_populations.geojson"
)

print("Loading centroids data...")
centroids_df = pd.read_csv("data/processed/bh_sectors_2022_centroids.csv")

print("Loading ovitraps data...")
ovitraps_df = pd.read_csv("data/processed/ovitraps_data.csv")

# Remove rows with missing coordinates
ovitraps_df = ovitraps_df[
    ovitraps_df["latitude"].notna() & ovitraps_df["longitude"].notna()
]

# Create biweek grouping to match the IDW calculation
ovitraps_df["epi_year"] = (
    ovitraps_df["epidemic_date"].str.split("W").str[0]
)
ovitraps_df["week_num"] = (
    ovitraps_df["epidemic_date"].str.split("W").str[1].astype(int)
)
ovitraps_df["biweek_num"] = ((ovitraps_df["week_num"] + 1) // 2) * 2
ovitraps_df["biweek"] = (
    ovitraps_df["epi_year"]
    + "W"
    + ovitraps_df["biweek_num"].astype(str).str.zfill(2)
)

# Get unique biweeks and sort them
biweeks = sorted(ovitraps_df["biweek"].unique())

print("Loading IDW data with trap associations...")
idw_df = pd.read_csv("data/processed/sector_centroids_with_idw.csv")

print(
    f"Loaded {len(sectors_gdf)} sectors, {len(centroids_df)} centroids, "
    f"and {len(ovitraps_df)} ovitrap samples across {len(biweeks)} biweeks"
)


def create_sectors_trace():
    """Create trace for census sector boundaries."""
    # Convert to GeoJSON format for Plotly
    import json

    geojson_data = json.loads(sectors_gdf.to_json())

    # Extract sector codes for hover
    sector_codes = sectors_gdf["CD_SETOR"].tolist()
    populations_2022 = sectors_gdf["pop_2022"].tolist()
    neighborhoods = sectors_gdf["NM_BAIRRO"].tolist()

    # Create hover text
    hover_text = [
        f"Sector: {code}<br>Neighborhood: {neighborhood}<br>Population (2022): {pop}"
        for code, pop, neighborhood in zip(
            sector_codes, populations_2022, neighborhoods
        )
    ]

    trace = go.Choroplethmapbox(
        geojson=geojson_data,
        locations=sector_codes,
        featureidkey="properties.CD_SETOR",
        z=[1] * len(sector_codes),  # Uniform color
        colorscale=[
            [0, "rgba(211, 211, 211, 0.3)"],
            [1, "rgba(211, 211, 211, 0.3)"],
        ],
        marker=dict(line=dict(width=0.5, color="gray")),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        showscale=False,
        name="Sectors",
    )

    return trace


def create_centroids_trace(biweek, selected_sector=None):
    """Create trace for sector centroids with clickable functionality."""
    # Determine colors based on selection
    colors = [
        "red" if code == selected_sector else "darkred"
        for code in centroids_df["CD_SETOR"]
    ]
    sizes = [
        8 if code == selected_sector else 3
        for code in centroids_df["CD_SETOR"]
    ]

    # Get IDW values for this biweek
    idw_for_date = idw_df[idw_df["biweek"] == biweek]

    # Create hover text with sector ID and egg count
    hover_text = []
    for code in centroids_df["CD_SETOR"]:
        idw_row = idw_for_date[idw_for_date["CD_SETOR"] == code]
        if not idw_row.empty:
            egg_value = idw_row.iloc[0]["idw_egg_value"]
            hover_text.append(
                f"Sector: {code}<br>Eggs (IDW): {egg_value:.1f}"
            )
        else:
            hover_text.append(f"Sector: {code}<br>Eggs (IDW): N/A")

    trace = go.Scattermapbox(
        lat=centroids_df["centroid_latitude"],
        lon=centroids_df["centroid_longitude"],
        mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.8),
        text=hover_text,
        customdata=centroids_df["CD_SETOR"],
        hovertemplate="%{text}<extra></extra>",
        name="Centroids",
    )

    return trace


def create_ovitraps_trace(biweek, selected_sector=None):
    """Create trace for ovitraps with hollow circles, highlighting those used in IDW."""
    # Filter data for the selected biweek
    ovitraps_filtered = ovitraps_df[ovitraps_df["biweek"] == biweek].copy()

    if len(ovitraps_filtered) == 0:
        return go.Scattermapbox(
            lat=[], lon=[], mode="markers", name="Ovitraps"
        )

    # Get narmads used for selected sector if any
    used_narmads = set()
    if selected_sector:
        idw_row = idw_df[
            (idw_df["CD_SETOR"] == selected_sector)
            & (idw_df["biweek"] == biweek)
        ]
        if not idw_row.empty:
            narmads_str = idw_row.iloc[0]["narmads_used"]
            used_narmads = set(narmads_str.split(","))

    # Determine colors and sizes: green/larger for used traps, black/smaller for others
    colors = []
    sizes = []
    for narmad in ovitraps_filtered["narmad"]:
        if str(narmad) in used_narmads:
            colors.append("green")
            sizes.append(10)
        else:
            colors.append("black")
            sizes.append(5)

    # Create hover text
    hover_text = [
        f"Trap: {narmad}<br>Eggs: {novos}"
        for narmad, novos in zip(
            ovitraps_filtered["narmad"], ovitraps_filtered["novos"]
        )
    ]

    trace = go.Scattermapbox(
        lat=ovitraps_filtered["latitude"],
        lon=ovitraps_filtered["longitude"],
        mode="markers",
        marker=dict(size=sizes, color=colors, opacity=1.0),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        name="Ovitraps",
    )

    return trace


# Create the Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Census Sectors, Centroids, and Ovitraps Map",
                    style={"textAlign": "center", "marginBottom": "10px"},
                ),
                html.P(
                    f"Total sectors: {len(sectors_gdf)} | Total centroids: {len(centroids_df)} | "
                    f"Total ovitrap samples: {len(ovitraps_df)}",
                    style={
                        "textAlign": "center",
                        "color": "gray",
                        "marginBottom": "10px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                html.Label(
                    "Select Biweek:",
                    style={"fontWeight": "bold", "marginBottom": "5px"},
                ),
                dcc.Slider(
                    id="biweek-slider",
                    min=0,
                    max=len(biweeks) - 1,
                    value=0,
                    marks={i: biweek for i, biweek in enumerate(biweeks)},
                    step=1,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ],
            style={"margin": "20px 50px"},
        ),
        dcc.Graph(
            id="sectors-map",
            config={"displayModeBar": True, "scrollZoom": True},
            style={"height": "700px"},
        ),
        html.Div(
            [
                html.P(
                    "Red dots: sector centroids (click to highlight) | "
                    "Hollow circles: ovitraps (green = used in IDW, gray = not used) | "
                    "Gray boundaries: sector limits",
                    style={
                        "textAlign": "center",
                        "color": "gray",
                        "marginTop": "10px",
                        "fontSize": "12px",
                    },
                )
            ]
        ),
        # Store for selected sector
        dcc.Store(id="selected-sector", data=None),
    ]
)


# Callback to update the map based on slider selection and centroid clicks
@app.callback(
    Output("sectors-map", "figure"),
    Output("selected-sector", "data"),
    Input("biweek-slider", "value"),
    Input("sectors-map", "clickData"),
    Input("selected-sector", "data"),
)
def update_map(slider_value, click_data, current_selected):
    """Update the map when the slider changes or a centroid is clicked."""
    selected_biweek = biweeks[slider_value]

    # Determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger = None
    else:
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # Update selected sector based on click
    selected_sector = current_selected
    if trigger == "sectors-map" and click_data:
        # Check if click was on a centroid (has customdata)
        if "customdata" in click_data["points"][0]:
            clicked_sector = click_data["points"][0]["customdata"]
            # Toggle selection
            if clicked_sector == current_selected:
                selected_sector = None
            else:
                selected_sector = clicked_sector
    elif trigger == "biweek-slider":
        # Keep current selection when slider changes
        selected_sector = current_selected

    # Create figure
    fig = go.Figure()

    # Add sectors trace
    sectors_trace = create_sectors_trace()
    fig.add_trace(sectors_trace)

    # Add centroids trace with selection
    centroids_trace = create_centroids_trace(
        selected_biweek, selected_sector
    )
    fig.add_trace(centroids_trace)

    # Add ovitraps trace for selected biweek with highlighting
    ovitraps_trace = create_ovitraps_trace(
        selected_biweek, selected_sector
    )
    fig.add_trace(ovitraps_trace)

    # Update layout
    title_text = f"Biweek: {selected_biweek}"
    if selected_sector:
        title_text += f" | Selected Sector: {selected_sector}"

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=MAP_CENTER_LAT, lon=MAP_CENTER_LON),
            zoom=MAP_ZOOM,
        ),
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text=title_text, x=0.5, xanchor="center", font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        uirevision="constant",  # Preserve zoom/pan state
    )

    return fig, selected_sector


if __name__ == "__main__":
    print("\nStarting dashboard server...")
    print("Open your browser and navigate to: http://127.0.0.1:8052/")
    app.run(debug=True, port=8052)
