import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils")))

import generic 
import project_utils
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update
from sklearn.linear_model import LinearRegression



spacing_m = 200

# Function to cluster cases by nearest grid point
def cluster_cases_by_year(dengue_data, grid_points, year):
    """Cluster dengue cases to nearest grid points for a specific year"""
    year_data = dengue_data[dengue_data['anoepid'] == year]
    if len(year_data) == 0:
        return pd.DataFrame(columns=['latitude', 'longitude', 'cases', 'cluster_id'])
    

    # Find nearest grid point for each case
    cluster_cases = generic.nearest_neighbors(year_data[['latitude', 'longitude']].values, grid_points.values)
    refpoint_idx, cases_per_refpoint = np.unique(cluster_cases, return_counts=True)
        
    cluster_data = pd.DataFrame({
            'latitude': grid_points.iloc[refpoint_idx,0],
            'longitude': grid_points.iloc[refpoint_idx,1],
            'cases': cases_per_refpoint,
            'cluster_id': refpoint_idx
        })
    
    return cluster_data


# Load dengue data
dengue_data = pd.read_csv('../../data/processed/dengue_data.csv')

# Filter data with coordinates and within Belo Horizonte bounds
dengue_filtered = dengue_data[['latitude', 'longitude', 'anoepid']].dropna()
dengue_filtered = dengue_filtered[
    (dengue_filtered['longitude'] > -44.1) & 
    (dengue_filtered['longitude'] < -43.87) & 
    (dengue_filtered['latitude'] > -20.03) & 
    (dengue_filtered['latitude'] < -19.8)
]


# Get available years
available_years = sorted(dengue_filtered['anoepid'].unique())
marks = ["*" if year in project_utils.EPIDEMY_YEARS else "" for year in available_years]


grid_points = generic.create_grid(
        lat_min=dengue_filtered['latitude'].min(),
        lat_max=dengue_filtered['latitude'].max(),
        lon_min=dengue_filtered['longitude'].min(),
        lon_max=dengue_filtered['longitude'].max(),
        spacing_m=spacing_m
    )


# Create dataset with all year combinations
def create_year_comparison_data():
    """Create dataset for year comparisons with missing data filled as 0"""
    all_data = []
        # Create grid for Belo Horizonte
    
    for year in available_years:
        year_clusters = cluster_cases_by_year(dengue_filtered, grid_points, year)
        year_clusters[f'cases_{year}'] = year_clusters['cases']
        all_data.append(year_clusters[['latitude', 'longitude', 'cluster_id', f'cases_{year}']])
    
    # Merge all years on cluster_id
    merged_df = grid_points.copy()
    merged_df['cluster_id'] = merged_df.index
    
    for i, year_data in enumerate(all_data):
        merged_df = merged_df.merge(
            year_data, 
            on=['latitude', 'longitude', 'cluster_id'], 
            how='left'
        )
    
    # Fill missing values with 0
    for year in available_years:
        merged_df[f'cases_{year}'] = merged_df[f'cases_{year}'].fillna(0)
    
    # Filter out grid points with no cases in any year
    case_columns = [f'cases_{year}' for year in available_years]
    merged_df = merged_df[merged_df[case_columns].sum(axis=1) > 0]
    
    return merged_df

df = create_year_comparison_data()
assert grid_points.iloc[df.index].equals(df[['latitude', 'longitude']])


app = Dash(__name__)

def base_map_figure(center_lat=None, center_lon=None, zoom=10):
    if center_lat is None:
        center_lat = -19.922778  # 19°55′21″S
    if center_lon is None:
        center_lon = -43.945556  # 43°56′42″W
    
    fig = go.Figure()
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={"lat": center_lat, "lon": center_lon},
            zoom=zoom
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=400
    )
    # Add an empty scatter trace to properly initialize the mapbox
    fig.add_trace(go.Scattermapbox(
        lat=[],
        lon=[],
        mode='markers',
        showlegend=False
    ))
    return fig

app.layout = html.Div([
    html.H3("Interactive Analysis Dashboard"),
    
    # Controls section at the top
    html.Div([
        html.Div([
            html.Label("First Epidemic Year"),
            dcc.Dropdown(
                id="series-x",
                options=[{"label": str(year) + mark, "value": f"cases_{year}"} for year, mark in zip(available_years, marks)],
                value=f"cases_2015_16"
            ),
        ], style={"width": "20%", "display": "inline-block", "paddingRight": "20px"}),
        
        html.Div([
            html.Label("Second Epidemic Year"),
            dcc.Dropdown(
                id="series-y", 
                options=[{"label": str(year) + mark, "value": f"cases_{year}"} for year, mark in zip(available_years, marks)],
                value=f"cases_2018_19"
            ),
        ], style={"width": "20%", "display": "inline-block", "paddingRight": "20px"}),
        
        html.Div([
            html.Label("Grid Spacing (meters):", style={"fontWeight": "bold", "fontSize": "12px"}),
            dcc.Input(
                id="grid-spacing-input",
                type="number",
                value=200, # default to 200m grid
                min=100,
                max=5000,
                step=50,
                style={"width": "100px", "marginBottom": "10px", "fontSize": "11px"}
            ),
        ], style={"width": "15%", "display": "inline-block", "paddingRight": "20px"}),
        
        html.Div([
            html.Button("Update Grid", id="update-grid-btn", 
                       style={"backgroundColor": "#28a745", "color": "white", "border": "none", 
                             "padding": "6px 12px", "borderRadius": "4px", "fontSize": "12px", "marginBottom": "8px"}),
            html.Div(f"Current: {spacing_m}m", id="current-spacing-display", 
                    style={"fontSize": "11px", "color": "#666"})
        ], style={"width": "15%", "display": "inline-block", "paddingRight": "20px"}),
    ], style={"marginBottom": "30px", "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"}),

    # Main content area - full width
    html.Div([
        # Two maps - full width
        html.H3("Geographical distribution of dengue cases by year"),
        html.Div([
            html.Div([dcc.Graph(id="map-series1", style={"height": "500px"})], 
                    style={"width": "49%", "display": "inline-block"}),
            html.Div([dcc.Graph(id="map-series2", style={"height": "500px"})], 
                    style={"width": "49%", "display": "inline-block", "marginLeft": "2%"})
        ]),
        
        # Ratio map
        html.Div([
            html.H3("Ratio Comparison Map (First Year / Second Year)"),
            html.Div([
                html.Label("Min Cases for Ratio:", style={"fontWeight": "bold", "fontSize": "14px", "marginRight": "10px"}),
                dcc.Input(
                    id="min-cases-input",
                    type="number",
                    value=1, # default minimum cases
                    min=0,
                    step=1,
                    style={"width": "80px", "fontSize": "12px"}
                ),
                html.Span("Filter low case areas", style={"fontSize": "12px", "color": "#666", "marginLeft": "10px"})
            ], style={"marginBottom": "10px", "display": "flex", "alignItems": "center"}),
            html.Div([
                dcc.Checklist(
                    id="relative-values-checkbox",
                    options=[{"label": " Use normalized values (z-score)", "value": "relative"}],
                    value=[],  # Default: use absolute values
                    style={"fontSize": "14px"}
                )
            ], style={"marginBottom": "15px"}),
            dcc.Graph(id="map-ratio", style={"height": "500px"})
        ], style={"marginTop": "30px"}),
        
        # Separator
        html.Hr(style={"marginTop": "30px", "marginBottom": "30px"}),
        
        # Scatter plot and selected points map below
        html.H3("Interactive Point Selection"),
        dcc.Graph(id="scatter-plot", style={"height": "500px", "marginBottom": "20px"}),
        
        # Buttons for map interaction
        html.Div([
            html.Button("Update map with selected points", id="update-map-btn", 
                       style={"marginRight": "10px", "backgroundColor": "#007bff", "color": "white", "border": "none", "padding": "8px 16px", "borderRadius": "4px"}),
            html.Button("Clear selection", id="clear-btn", 
                       style={"backgroundColor": "#ff6b6b", "color": "white", "border": "none", "padding": "8px 16px", "borderRadius": "4px"})
        ], style={"marginBottom": "20px", "textAlign": "left"}),
        
        dcc.Graph(id="map-selected", figure=base_map_figure(), 
                 style={"height": "800px", "marginBottom": "50px"}),
        
        # Hidden storage
        dcc.Store(id="selected-points-store"),
        dcc.Store(id="grid-data-store", data=df.to_dict('records')),
        dcc.Store(id="current-spacing-store", data=spacing_m)
    ], style={"width": "100%"}),

    html.Div(style={"height": "50px"})
])

# Callback to update grid data when button is pressed
@app.callback(
    Output("grid-data-store", "data"),
    Output("current-spacing-store", "data"),
    Output("current-spacing-display", "children"),
    Input("update-grid-btn", "n_clicks"),
    State("grid-spacing-input", "value"),
    State("current-spacing-store", "data"),
    prevent_initial_call=True
)
def update_grid_data(n_clicks, new_spacing, current_spacing):
    if new_spacing == current_spacing:
        return no_update, no_update, no_update
    
    # Create new grid with updated spacing
    new_grid_points = generic.create_grid(
        lat_min=dengue_filtered['latitude'].min(),
        lat_max=dengue_filtered['latitude'].max(),
        lon_min=dengue_filtered['longitude'].min(),
        lon_max=dengue_filtered['longitude'].max(),
        spacing_m=new_spacing
    )
    
    # Recreate dataset with new grid
    def create_year_comparison_data_new_grid():
        all_data = []
        
        for year in available_years:
            year_clusters = cluster_cases_by_year(dengue_filtered, new_grid_points, year)
            year_clusters[f'cases_{year}'] = year_clusters['cases']
            all_data.append(year_clusters[['latitude', 'longitude', 'cluster_id', f'cases_{year}']])
        
        merged_df = new_grid_points.copy()
        merged_df['cluster_id'] = merged_df.index
        
        for i, year_data in enumerate(all_data):
            merged_df = merged_df.merge(
                year_data, 
                on=['latitude', 'longitude', 'cluster_id'], 
                how='left'
            )
        
        for year in available_years:
            merged_df[f'cases_{year}'] = merged_df[f'cases_{year}'].fillna(0)
        
        case_columns = [f'cases_{year}' for year in available_years]
        merged_df = merged_df[merged_df[case_columns].sum(axis=1) > 0]
        
        return merged_df
    
    new_df = create_year_comparison_data_new_grid()
    return new_df.to_dict('records'), new_spacing, f"Current: {new_spacing}m"

@app.callback(
    Output("scatter-plot", "figure"),
    Input("series-x", "value"),
    Input("series-y", "value"),
    Input("grid-data-store", "data")
)
def update_scatter(xcol, ycol, grid_data):
    current_df = pd.DataFrame(grid_data)
    
    # Create scatter plot
    fig = px.scatter(
        current_df,
        x=xcol,
        y=ycol,
        custom_data=["latitude", "longitude"], 
        title=f"Dengue Cases Correlation: {xcol.replace('cases_', '')} vs {ycol.replace('cases_', '')}"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    
    # Add linear regression line
    x_vals = current_df[xcol].values
    y_vals = current_df[ycol].values
    
    # Remove points where either x or y is 0 for better regression
    mask = (x_vals > 0) & (y_vals > 0)
    if np.sum(mask) > 1:  # Need at least 2 points for regression
        x_reg = x_vals[mask].reshape(-1, 1)
        y_reg = y_vals[mask]
        
        # Fit linear regression
        reg_model = LinearRegression()
        reg_model.fit(x_reg, y_reg)
        
        # Create regression line points
        x_min, x_max = x_vals.min(), x_vals.max()
        x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_line = reg_model.predict(x_line)
        
        # Add regression line to plot
        fig.add_trace(go.Scatter(
            x=x_line.flatten(),
            y=y_line,
            mode='lines',
            name=f'Linear Regression (R² = {reg_model.score(x_reg, y_reg):.3f})',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(dragmode="lasso")
    return fig

@app.callback(
    Output("selected-points-store", "data"),
    Input("scatter-plot", "selectedData"),
    State("selected-points-store", "data"),
    State("grid-data-store", "data")
)
def store_selected_points(selectedData, current_selected, grid_data):
    current_df = pd.DataFrame(grid_data)
    if not current_selected:
        current_selected = []
    
    if not selectedData or "points" not in selectedData:
        return current_selected
    
    # Create a set of current selected points for easy lookup
    current_set = set()
    for point in current_selected:
        current_set.add((round(point["lat"], 6), round(point["lon"], 6)))
    
    # Process newly selected points
    for p in selectedData["points"]:
        # Get lat/lon from the point
        if "customdata" in p and len(p["customdata"]) >= 2:
            lat, lon = p["customdata"][:2]
        else:
            idx = p.get("pointIndex")
            lat, lon = current_df.iloc[idx][["latitude", "longitude"]]
        
        # Round for consistent comparison
        lat_round = round(lat, 6)
        lon_round = round(lon, 6)
        point_key = (lat_round, lon_round)
        
        # Toggle logic: if point already selected, remove it; otherwise add it
        if point_key in current_set:
            # Remove the point
            current_selected = [pt for pt in current_selected 
                              if not (round(pt["lat"], 6) == lat_round and round(pt["lon"], 6) == lon_round)]
            current_set.remove(point_key)
        else:
            # Add the point
            current_selected.append({"lat": lat, "lon": lon})
            current_set.add(point_key)
    
    return current_selected

@app.callback(
    Output("selected-points-store", "data", allow_duplicate=True),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_selection(_):
    return []

@app.callback(
    Output("map-selected", "figure"),
    Input("update-map-btn", "n_clicks"),
    State("selected-points-store", "data"),
    State("grid-data-store", "data"),
    State("series-x", "value"),
    State("series-y", "value"),
    prevent_initial_call=True
)
def update_map(_, selected_points, grid_data, series_x, series_y):
    # return base map if nothing selected
    if not selected_points:
        return base_map_figure()
    
    # Get the grid data to find case counts for selected points
    current_df = pd.DataFrame(grid_data)
    
    # Create dataframe with selected points and their case counts
    selected_data = []
    for point in selected_points:
        lat, lon = point["lat"], point["lon"]
        # Find matching row in grid data (with small tolerance for floating point comparison)
        matching_row = current_df[
            (abs(current_df['latitude'] - lat) < 0.0001) & 
            (abs(current_df['longitude'] - lon) < 0.0001)
        ]
        if not matching_row.empty:
            row = matching_row.iloc[0]
            selected_data.append({
                "lat": lat,
                "lon": lon,
                "cases_x": int(row[series_x]),
                "cases_y": int(row[series_y])
            })
    
    if not selected_data:
        return base_map_figure()
    
    dff = pd.DataFrame(selected_data)
    lats = dff["lat"].tolist()
    lons = dff["lon"].tolist()
    
    fig = px.scatter_mapbox(
        dff, 
        lat="lat", 
        lon="lon",
        custom_data=["cases_x", "cases_y"]
    )
    
    # Update hover template to show both year counts
    year_x = series_x.replace('cases_', '')
    year_y = series_y.replace('cases_', '')
    fig.update_traces(
        hovertemplate=f"<b>{year_x}: %{{customdata[0]}}</b><br><b>{year_y}: %{{customdata[1]}}</b><extra></extra>"
    )
    
    # center on selected points
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": float(np.mean(lats)), "lon": float(np.mean(lons))},
        mapbox_zoom=10,
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    return fig

@app.callback(
    Output("map-series1", "figure"),
    Output("map-series2", "figure"),
    Output("map-ratio", "figure"),
    Input("series-x", "value"),
    Input("series-y", "value"),
    Input("grid-data-store", "data"),
    Input("current-spacing-store", "data"),
    Input("min-cases-input", "value"),
    Input("relative-values-checkbox", "value")
)
def update_series_maps(series_x, series_y, grid_data, current_spacing, min_cases, use_relative):
    # Always show all data points - static maps not affected by selection
    data_to_plot = pd.DataFrame(grid_data)
    center_lat = -19.922778
    center_lon = -43.945556
    zoom_level = 10
    grid_scale = 0.5
    # Filter out points with zero cases for better visualization
    data_series1 = data_to_plot[data_to_plot[series_x] > 0].copy()
    data_series2 = data_to_plot[data_to_plot[series_y] > 0].copy()
    
    # Create color scale from blue (small) to red (large) based on case counts
    max_cases_1 = data_series1[series_x].max() if len(data_series1) > 0 else 1
    max_cases_2 = data_series2[series_y].max() if len(data_series2) > 0 else 1
    overall_max = max(max_cases_1, max_cases_2)
    
    # Normalize case counts to 0-1 for color mapping
    data_series1['color_intensity'] = data_series1[series_x] / overall_max
    data_series2['color_intensity'] = data_series2[series_y] / overall_max

    # Create maps with uniform small circles, only color varies
    fig1 = px.scatter_mapbox(
        data_series1, 
        lat="latitude",
        lon="longitude",
        color='color_intensity',
        color_continuous_scale=[(0, 'blue'), (1, 'red')],  # Blue to red gradient
        title=f"Dengue Cases - {series_x.replace('cases_', '')} (Grid: {current_spacing}m)",
        custom_data=[series_x]
    )
    
    fig2 = px.scatter_mapbox(
        data_series2, 
        lat="latitude",
        lon="longitude",
        color='color_intensity',
        color_continuous_scale=[(0, 'blue'), (1, 'red')],  # Blue to red gradient
        title=f"Dengue Cases - {series_y.replace('cases_', '')} (Grid: {current_spacing}m)",
        custom_data=[series_y]
    )
    
    # Update hover templates to show "Counts: X" format
    fig1.update_traces(hovertemplate="<b>Counts: %{customdata[0]}</b><extra></extra>")
    fig2.update_traces(hovertemplate="<b>Counts: %{customdata[0]}</b><extra></extra>")
    
    # Apply consistent styling to both maps with variable circle sizes
    fig1.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=zoom_level,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_showscale=False  # Hide the color bar
    )
    # Make circle size proportional to case counts for first map
    fig1.update_traces(
        marker=dict(
            size=np.sqrt(data_series1[series_x]) * grid_scale,  # Size varies with case count
            opacity=0.8
        )
    )
    
    fig2.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=zoom_level,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_showscale=False  # Hide the color bar
    )
    # Make circle size proportional to case counts for second map
    fig2.update_traces(
        marker=dict(
            size=np.sqrt(data_series2[series_y]) * grid_scale,  # Size varies with case count
            opacity=0.8
        )
    )
    
    # Create ratio map
    # Calculate ratio with protection against division by zero
    data_ratio = data_to_plot.copy()
    
    # Check if relative values should be used
    is_relative = 'relative' in use_relative if use_relative else False
    
    if is_relative:
        # Calculate relative values using z-score normalization
        # Normalize each year's values by their own mean and standard deviation
        
        # Z-score normalization for series_x (first year)
        x_raw = data_ratio[series_x]
        x_mean = x_raw.mean()
        x_std = x_raw.std()
        if x_std == 0:  # Handle case where all values are the same
            x_normalized = np.zeros_like(x_raw)
        else:
            x_normalized = (x_raw - x_mean) / x_std
        
        # Z-score normalization for series_y (second year)
        y_raw = data_ratio[series_y]
        y_mean = y_raw.mean()
        y_std = y_raw.std()
        if y_std == 0:  # Handle case where all values are the same
            y_normalized = np.zeros_like(y_raw)
        else:
            y_normalized = (y_raw - y_mean) / y_std
        
        # Shift z-scores to positive values for ratio calculation (add 3 to ensure positive values)
        # Z-scores are typically between -3 and +3, so adding 3 makes them 0 to 6
        x_values = x_normalized + 3
        y_values = y_normalized + 3
        
        # Replace any remaining zero/negative values to avoid division issues
        x_values = x_values.replace(0, 0.1)
        y_values = y_values.replace(0, 0.1)
        x_values = np.maximum(x_values, 0.1)  # Ensure all values are positive
        y_values = np.maximum(y_values, 0.1)
        
        comparison_type = "normalized"
    else:
        # Use absolute values
        # Transform zeros to ones for both series to avoid division issues
        x_values = data_ratio[series_x].replace(0, 1)
        y_values = data_ratio[series_y].replace(0, 1)
        comparison_type = "absolute"
    
    # Calculate ratio: first year / second year
    data_ratio['ratio'] = x_values / y_values
    
    # Store z-scores for hover display (if using normalized values)
    if is_relative:
        data_ratio['zscore_x'] = x_normalized
        data_ratio['zscore_y'] = y_normalized
    else:
        data_ratio['zscore_x'] = np.nan  # Not applicable for absolute values
        data_ratio['zscore_y'] = np.nan
    
    # Filter out points where either year has fewer cases than the minimum threshold
    # This helps focus on meaningful comparisons and avoids noise from areas with very few cases
    if min_cases is None:
        min_cases = 1
    data_ratio = data_ratio[(data_to_plot[series_x] >= min_cases) & (data_to_plot[series_y] >= min_cases)].copy()
    
    # Create the ratio map
    # Use direct ratio values (all will be >= 0)
    # Color scale: low ratios (second year higher) = blue, high ratios (first year higher) = red
    fig_ratio = px.scatter_mapbox(
        data_ratio,
        lat="latitude", 
        lon="longitude",
        color='ratio',
        color_continuous_scale='Viridis',  # Better for ratios starting from 0
        title=f"Ratio Map: {series_x.replace('cases_', '')} / {series_y.replace('cases_', '')} ({comparison_type.title()} Values - Grid: {current_spacing}m, Min: {min_cases} cases)",
        custom_data=['ratio', series_x, series_y, 'zscore_x', 'zscore_y'],
        range_color=[0, data_ratio['ratio'].quantile(0.95)]  # Scale from 0 to 95th percentile to handle outliers
    )
    
    # Update hover template to show ratio and actual counts
    year_x = series_x.replace('cases_', '')
    year_y = series_y.replace('cases_', '')
    
    if is_relative:
        hover_template = f"<b>{comparison_type.title()} Ratio: %{{customdata[0]:.2f}}</b><br><b>{year_x}: %{{customdata[1]}} cases</b><br><b>{year_y}: %{{customdata[2]}} cases</b><br><i>(Comparing z-score normalized values)</i><extra></extra>"
    else:
        hover_template = f"<b>{comparison_type.title()} Ratio: %{{customdata[0]:.2f}}</b><br><b>{year_x}: %{{customdata[1]}} cases</b><br><b>{year_y}: %{{customdata[2]}} cases</b><extra></extra>"
    
    fig_ratio.update_traces(hovertemplate=hover_template)
    
    # Update layout and styling
    fig_ratio.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=zoom_level,
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    
    # Make circle size proportional to total cases for better visibility
    total_cases = data_ratio[series_x] + data_ratio[series_y]
    fig_ratio.update_traces(
        marker=dict(
            size=np.sqrt(total_cases) * grid_scale,
            opacity=0.8
        )
    )
    
    return fig1, fig2, fig_ratio

if __name__ == '__main__':
    app.run(debug=True)