"""
Interactive Sector Dashboard: Ovitraps IDW & Cases Per Capita

Click on census sectors to view time series of:
- IDW ovitrap egg values (biweekly)
- Dengue cases per 1000 people (biweekly)
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GEOJSON_PATH = PROCESSED_DIR / "bh_sectors_2022_with_populations.geojson"
IDW_CSV = PROCESSED_DIR / "sector_centroids_with_idw.csv"
PER_CAPITA_CSV = PROCESSED_DIR / "dengue_per_capita.csv"

MAP_CENTER_LAT = -19.9167
MAP_CENTER_LON = -43.9345
MAP_ZOOM = 11
MAPBOX_STYLE = "carto-positron"

CASES_COLORSCALE = [
    [0.0, "rgb(5,48,97)"],
    [0.15, "rgb(33,102,172)"],
    [0.3, "rgb(67,147,195)"],
    [0.45, "rgb(146,197,222)"],
    [0.5, "rgb(247,247,247)"],
    [0.6, "rgb(244,165,130)"],
    [0.75, "rgb(214,96,77)"],
    [0.9, "rgb(178,24,43)"],
    [1.0, "rgb(103,0,31)"],
]

EGGS_COLORSCALE = [
    [0.0, "rgb(255,255,229)"],
    [0.2, "rgb(247,252,185)"],
    [0.4, "rgb(217,240,163)"],
    [0.6, "rgb(173,221,142)"],
    [0.8, "rgb(120,198,121)"],
    [1.0, "rgb(49,163,84)"],
]

print("Loading data...")
sectors_gdf = gpd.read_file(GEOJSON_PATH)
sectors_gdf["CD_SETOR"] = sectors_gdf["CD_SETOR"].astype(str)
GEOJSON_DATA = json.loads(sectors_gdf.to_json())

idw_df = pd.read_csv(IDW_CSV)
idw_df["CD_SETOR"] = idw_df["CD_SETOR"].astype(str)
idw_df["biweek"] = idw_df["biweek"].astype(str)

pc_df = pd.read_csv(PER_CAPITA_CSV)
pc_df["sector_id"] = pc_df["sector_id"].astype(str)
pc_df["biweek"] = pc_df["biweek"].astype(str)

ALL_BIWEEKS = sorted(
    set(idw_df["biweek"].unique()) & set(pc_df["biweek"].unique())
)
BIWEEK_TO_IDX = {w: i for i, w in enumerate(ALL_BIWEEKS)}

print(f"Loaded {len(sectors_gdf)} sectors, {len(ALL_BIWEEKS)} biweeks")
print(f"IDW data: {len(idw_df):,} rows | Per capita: {len(pc_df):,} rows")

app = Dash(__name__)
app.title = "Sector Time Series Dashboard"

app.layout = html.Div(
    [
        html.H1(
            "Sector Time Series Dashboard — Ovitraps & Dengue Cases",
            style={
                "textAlign": "center",
                "color": "#2C3E50",
                "marginBottom": 5,
            },
        ),
        html.P(
            f"{len(ALL_BIWEEKS)} biweeks  ·  {len(sectors_gdf)} sectors  ·  "
            "Click a sector to view time series",
            style={
                "textAlign": "center",
                "color": "#888",
                "fontSize": 13,
                "marginBottom": 15,
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Select Biweek:", style={"fontWeight": "bold"}
                        ),
                        dcc.Slider(
                            id="week-slider",
                            min=0,
                            max=len(ALL_BIWEEKS) - 1,
                            value=0,
                            marks={
                                i: w
                                for i, w in enumerate(ALL_BIWEEKS)
                                if i % max(1, len(ALL_BIWEEKS) // 15) == 0
                            },
                            step=1,
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                    ],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "verticalAlign": "middle",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Map Metric:",
                            style={
                                "fontWeight": "bold",
                                "marginRight": 10,
                            },
                        ),
                        dcc.RadioItems(
                            id="metric-selector",
                            options=[
                                {"label": " IDW Eggs", "value": "eggs"},
                                {"label": " Cases/1000", "value": "cases"},
                            ],
                            value="eggs",
                            inline=True,
                            style={"fontSize": 14},
                        ),
                    ],
                    style={
                        "width": "25%",
                        "display": "inline-block",
                        "verticalAlign": "middle",
                        "textAlign": "right",
                    },
                ),
            ],
            style={"margin": "0 40px 10px 40px"},
        ),
        html.Div(
            id="sector-info",
            style={
                "margin": "0 40px 10px 40px",
                "fontSize": 13,
                "minHeight": 20,
                "color": "#555",
            },
        ),
        html.Div(
            dcc.Graph(id="choropleth-map", style={"height": "550px"}),
            style={"margin": "0 20px"},
        ),
        html.Hr(style={"margin": "20px 40px"}),
        html.H2(
            "Time Series for Selected Sector",
            style={
                "textAlign": "center",
                "marginTop": 10,
                "color": "#2C3E50",
                "fontSize": 20,
            },
        ),
        html.Div(
            id="timeseries-container",
            children=[
                html.P(
                    "Click on a sector in the map above to view time series",
                    style={
                        "textAlign": "center",
                        "color": "#999",
                        "fontSize": 14,
                        "marginTop": 30,
                    },
                )
            ],
        ),
        dcc.Store(id="selected-sector", data=None),
    ],
    style={"fontFamily": "Arial, sans-serif", "padding": 10},
)


@app.callback(
    Output("choropleth-map", "figure"),
    Output("sector-info", "children"),
    Input("week-slider", "value"),
    Input("metric-selector", "value"),
    Input("selected-sector", "data"),
)
def update_map(slider_idx, metric, selected_sector):
    biweek = ALL_BIWEEKS[slider_idx]

    if metric == "eggs":
        week_data = idw_df[idw_df["biweek"] == biweek][
            ["CD_SETOR", "idw_egg_value"]
        ].copy()
        week_data.rename(columns={"idw_egg_value": "value"}, inplace=True)
        colorscale = EGGS_COLORSCALE
        colorbar_title = "IDW Eggs"
        metric_label = "Eggs"
    else:
        week_data = pc_df[pc_df["biweek"] == biweek][
            ["sector_id", "cases_per_1000"]
        ].copy()
        week_data.rename(
            columns={"sector_id": "CD_SETOR", "cases_per_1000": "value"},
            inplace=True,
        )
        colorscale = CASES_COLORSCALE
        colorbar_title = "Cases/1k"
        metric_label = "Cases/1000"

    merged = sectors_gdf[["CD_SETOR", "NM_BAIRRO", "pop_2022"]].merge(
        week_data,
        on="CD_SETOR",
        how="left",
    )
    merged["value"] = merged["value"].fillna(0)

    hover = [
        f"Sector: {s}<br>Neighborhood: {n}<br>Pop: {p}<br>{metric_label}: {v:.2f}"
        for s, n, p, v in zip(
            merged["CD_SETOR"],
            merged["NM_BAIRRO"],
            merged["pop_2022"],
            merged["value"],
        )
    ]

    max_val = max(merged["value"].quantile(0.95), 1)

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=GEOJSON_DATA,
            locations=merged["CD_SETOR"].tolist(),
            z=merged["value"].tolist(),
            featureidkey="properties.CD_SETOR",
            colorscale=colorscale,
            zmin=0,
            zmax=max_val,
            marker={
                "opacity": 0.8,
                "line": {"width": 0.5, "color": "#444"},
            },
            hovertext=hover,
            hoverinfo="text",
            colorbar=dict(title=colorbar_title, thickness=15, len=0.5),
            name="Metric",
        )
    )

    if selected_sector:
        sector_geo = sectors_gdf[
            sectors_gdf["CD_SETOR"] == selected_sector
        ]
        if not sector_geo.empty:
            centroid = sector_geo.geometry.centroid.iloc[0]
            fig.add_trace(
                go.Scattermapbox(
                    lat=[centroid.y],
                    lon=[centroid.x],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color="red",
                        opacity=1.0,
                        symbol="circle",
                    ),
                    hovertext=[f"Selected: {selected_sector}"],
                    hoverinfo="text",
                    name="Selected",
                )
            )

    fig.update_layout(
        mapbox=dict(
            style=MAPBOX_STYLE,
            center=dict(lat=MAP_CENTER_LAT, lon=MAP_CENTER_LON),
            zoom=MAP_ZOOM,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Biweek: {biweek}", x=0.5, font=dict(size=15)),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        uirevision="constant",
    )

    info_text = f"Biweek: {biweek}"
    if selected_sector:
        sector_info = sectors_gdf[
            sectors_gdf["CD_SETOR"] == selected_sector
        ]
        if not sector_info.empty:
            neighborhood = sector_info.iloc[0]["NM_BAIRRO"]
            pop = sector_info.iloc[0]["pop_2022"]
            info_text += f"  |  Selected Sector: {selected_sector} ({neighborhood}, Pop: {pop:,})"

    return fig, info_text


@app.callback(
    Output("selected-sector", "data"),
    Input("choropleth-map", "clickData"),
    State("selected-sector", "data"),
)
def update_selected_sector(click_data, current_selected):
    if not click_data:
        return current_selected

    clicked_sector = click_data["points"][0]["location"]

    if clicked_sector == current_selected:
        return None

    return clicked_sector


@app.callback(
    Output("timeseries-container", "children"),
    Input("selected-sector", "data"),
    Input("week-slider", "value"),
)
def update_timeseries(selected_sector, slider_idx):
    if not selected_sector:
        return html.P(
            "Click on a sector in the map above to view time series",
            style={
                "textAlign": "center",
                "color": "#999",
                "fontSize": 14,
                "marginTop": 30,
            },
        )

    current_biweek = ALL_BIWEEKS[slider_idx]

    sector_idw = idw_df[idw_df["CD_SETOR"] == selected_sector].copy()
    sector_idw = sector_idw.sort_values("biweek")

    sector_pc = pc_df[pc_df["sector_id"] == selected_sector].copy()
    sector_pc = sector_pc.sort_values("biweek")

    fig_eggs = go.Figure()
    fig_eggs.add_trace(
        go.Scatter(
            x=sector_idw["biweek"],
            y=sector_idw["idw_egg_value"],
            mode="lines+markers",
            name="IDW Eggs",
            line=dict(color="#2ECC71", width=2),
            marker=dict(size=4),
        )
    )

    if current_biweek in sector_idw["biweek"].values:
        current_val = sector_idw[sector_idw["biweek"] == current_biweek][
            "idw_egg_value"
        ].iloc[0]
        fig_eggs.add_trace(
            go.Scatter(
                x=[current_biweek],
                y=[current_val],
                mode="markers",
                name="Current",
                marker=dict(size=12, color="red", symbol="circle"),
            )
        )

    fig_eggs.update_layout(
        title=f"IDW Ovitrap Eggs — Sector {selected_sector}",
        xaxis_title="Biweek",
        yaxis_title="IDW Egg Value",
        height=350,
        margin=dict(l=50, r=20, t=40, b=80),
        xaxis=dict(tickangle=-45),
        hovermode="x unified",
    )

    fig_cases = go.Figure()
    fig_cases.add_trace(
        go.Scatter(
            x=sector_pc["biweek"],
            y=sector_pc["cases_per_1000"],
            mode="lines+markers",
            name="Crude Rate",
            line=dict(color="#E74C3C", width=2),
            marker=dict(size=4),
        )
    )

    if "eb_rate_per_1000" in sector_pc.columns:
        fig_cases.add_trace(
            go.Scatter(
                x=sector_pc["biweek"],
                y=sector_pc["eb_rate_per_1000"],
                mode="lines+markers",
                name="EB Smoothed",
                line=dict(color="#3498DB", width=2, dash="dash"),
                marker=dict(size=4),
            )
        )

    if current_biweek in sector_pc["biweek"].values:
        current_val = sector_pc[sector_pc["biweek"] == current_biweek][
            "cases_per_1000"
        ].iloc[0]
        fig_cases.add_trace(
            go.Scatter(
                x=[current_biweek],
                y=[current_val],
                mode="markers",
                name="Current",
                marker=dict(size=12, color="red", symbol="circle"),
            )
        )

    fig_cases.update_layout(
        title=f"Dengue Cases per 1000 — Sector {selected_sector}",
        xaxis_title="Biweek",
        yaxis_title="Cases per 1000",
        height=350,
        margin=dict(l=50, r=20, t=40, b=80),
        xaxis=dict(tickangle=-45),
        hovermode="x unified",
    )

    return html.Div(
        [
            html.Div(
                dcc.Graph(figure=fig_eggs),
                style={
                    "width": "50%",
                    "display": "inline-block",
                    "verticalAlign": "top",
                },
            ),
            html.Div(
                dcc.Graph(figure=fig_cases),
                style={
                    "width": "50%",
                    "display": "inline-block",
                    "verticalAlign": "top",
                },
            ),
        ]
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting Sector Time Series Dashboard")
    print("Open: http://127.0.0.1:8059/")
    print("=" * 70 + "\n")
    app.run(debug=True, port=8059)
