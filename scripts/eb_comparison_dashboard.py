"""
Empirical Bayes Comparison Dashboard

Side-by-side comparison of crude vs. EB-smoothed dengue incidence rates.
Visualises the shrinkage effect of Marshall (1991) Empirical Bayes estimation:
  - Dual choropleth maps (crude vs. EB)
  - Scatter: crude vs. EB coloured by population (shrinkage funnel)
  - |crude − EB| vs. population (correction magnitude)
  - Histogram overlay of both distributions
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

# =====================================================================
# PATHS & CONSTANTS
# =====================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PER_CAPITA_CSV = PROCESSED_DIR / "dengue_per_capita.csv"
GEOJSON_PATH = PROCESSED_DIR / "bh_sectors_2022_with_populations.geojson"

MAP_CENTER_LAT = -19.9167
MAP_CENTER_LON = -43.9345
MAP_ZOOM = 11
MAPBOX_STYLE = "carto-positron"

RATE_COLORSCALE = [
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

# =====================================================================
# DATA
# =====================================================================

print("Loading data …")
sectors_gdf = gpd.read_file(GEOJSON_PATH)
sectors_gdf["CD_SETOR"] = sectors_gdf["CD_SETOR"].astype(str)
GEOJSON_DATA = json.loads(sectors_gdf.to_json())

pc_df = pd.read_csv(PER_CAPITA_CSV)
pc_df["sector_id"] = pc_df["sector_id"].astype(str)
pc_df["biweek"] = pc_df["biweek"].astype(str)

ALL_BIWEEKS = sorted(pc_df["biweek"].unique())
print(
    f"  {len(pc_df):,} rows | {len(ALL_BIWEEKS)} biweeks | "
    f"{sectors_gdf.shape[0]} sectors"
)

# =====================================================================
# DASH APP
# =====================================================================

app = Dash(__name__)
app.title = "EB Comparison — Crude vs Smoothed"

app.layout = html.Div(
    [
        html.H1(
            "Empirical Bayes Comparison — Crude vs Smoothed Rates",
            style={
                "textAlign": "center",
                "color": "#2C3E50",
                "marginBottom": 5,
            },
        ),
        html.P(
            "Marshall (1991) Poisson-Gamma EB smoothing.  "
            "Small-population sectors are shrunk toward the global mean; "
            "large-population sectors are nearly unchanged.",
            style={
                "textAlign": "center",
                "color": "#888",
                "fontSize": 13,
                "marginBottom": 15,
            },
        ),
        # Biweek slider
        html.Div(
            [
                html.Label("Select Biweek:", style={"fontWeight": "bold"}),
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
            style={"margin": "0 40px 20px 40px"},
        ),
        # --- Dual choropleth maps ---
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="map-crude", style={"height": "500px"}),
                    style={"width": "50%", "display": "inline-block"},
                ),
                html.Div(
                    dcc.Graph(id="map-eb", style={"height": "500px"}),
                    style={"width": "50%", "display": "inline-block"},
                ),
            ]
        ),
        html.Hr(style={"margin": "20px 40px"}),
        # --- Scatter + shrinkage + histogram ---
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id="scatter-crude-eb", style={"height": "420px"}
                    ),
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    dcc.Graph(
                        id="shrinkage-pop", style={"height": "420px"}
                    ),
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    dcc.Graph(
                        id="histogram-overlay", style={"height": "420px"}
                    ),
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={"margin": "0 20px"},
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "padding": 10},
)


# =====================================================================
# CALLBACKS
# =====================================================================


def _make_map(biweek_data, col, title, geojson):
    """Build a choropleth figure for a single rate column."""
    merged = sectors_gdf[["CD_SETOR"]].merge(
        biweek_data[["sector_id", "case_count", "population", col]],
        left_on="CD_SETOR",
        right_on="sector_id",
        how="left",
    )
    merged[col] = merged[col].fillna(0)
    merged["case_count"] = merged["case_count"].fillna(0).astype(int)
    merged["population"] = merged["population"].fillna(0).astype(int)

    hover = [
        f"Sector: {s}<br>Cases: {c}<br>Pop: {p}<br>Rate: {r:.3f}"
        for s, c, p, r in zip(
            merged["CD_SETOR"],
            merged["case_count"],
            merged["population"],
            merged[col],
        )
    ]
    max_rate = max(merged[col].quantile(0.95), 1)

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=merged["CD_SETOR"].tolist(),
            z=merged[col].tolist(),
            featureidkey="properties.CD_SETOR",
            colorscale=RATE_COLORSCALE,
            zmin=0,
            zmax=max_rate,
            marker={
                "opacity": 0.8,
                "line": {"width": 0.3, "color": "#444"},
            },
            hovertext=hover,
            hoverinfo="text",
            colorbar=dict(title="Rate", thickness=12, len=0.5),
        )
    )
    fig.update_layout(
        mapbox=dict(
            style=MAPBOX_STYLE,
            center=dict(lat=MAP_CENTER_LAT, lon=MAP_CENTER_LON),
            zoom=MAP_ZOOM,
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        title=dict(text=title, x=0.5, font=dict(size=14)),
        uirevision="constant",
    )
    return fig


@app.callback(
    Output("map-crude", "figure"),
    Output("map-eb", "figure"),
    Output("scatter-crude-eb", "figure"),
    Output("shrinkage-pop", "figure"),
    Output("histogram-overlay", "figure"),
    Input("week-slider", "value"),
)
def update_all(slider_idx):
    biweek = ALL_BIWEEKS[slider_idx]
    wdf = pc_df[pc_df["biweek"] == biweek].copy()

    # --- Dual maps ---
    fig_crude = _make_map(
        wdf, "cases_per_1000", f"Crude Rate — {biweek}", GEOJSON_DATA
    )
    fig_eb = _make_map(
        wdf,
        "eb_rate_per_1000",
        f"EB Smoothed Rate — {biweek}",
        GEOJSON_DATA,
    )

    # --- Scatter: crude vs EB coloured by log(population) ---
    valid = wdf[
        (wdf["population"] > 0) & (wdf["cases_per_1000"] > 0)
    ].copy()
    valid["log_pop"] = np.log10(valid["population"].clip(lower=1))

    fig_scatter = go.Figure()
    if not valid.empty:
        fig_scatter.add_trace(
            go.Scatter(
                x=valid["cases_per_1000"],
                y=valid["eb_rate_per_1000"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=valid["log_pop"],
                    colorscale="Viridis",
                    colorbar=dict(
                        title="log₁₀(Pop)", thickness=12, len=0.5
                    ),
                    opacity=0.7,
                ),
                text=[
                    f"Sector: {s}<br>Pop: {p}<br>Crude: {c:.3f}<br>EB: {e:.3f}"
                    for s, p, c, e in zip(
                        valid["sector_id"],
                        valid["population"],
                        valid["cases_per_1000"],
                        valid["eb_rate_per_1000"],
                    )
                ],
                hoverinfo="text",
            )
        )
        max_val = max(
            valid["cases_per_1000"].max(), valid["eb_rate_per_1000"].max()
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="grey", dash="dash", width=1),
                showlegend=False,
            )
        )
    fig_scatter.update_layout(
        title=dict(
            text="Crude vs EB Rate (shrinkage funnel)", font=dict(size=13)
        ),
        xaxis_title="Crude Rate (per 1k)",
        yaxis_title="EB Rate (per 1k)",
        margin=dict(l=50, r=20, t=40, b=50),
        height=420,
    )

    # --- Shrinkage magnitude vs population ---
    fig_shrink = go.Figure()
    if not valid.empty:
        valid["shrinkage"] = (
            valid["cases_per_1000"] - valid["eb_rate_per_1000"]
        ).abs()
        fig_shrink.add_trace(
            go.Scatter(
                x=valid["population"],
                y=valid["shrinkage"],
                mode="markers",
                marker=dict(size=4, color="#E74C3C", opacity=0.5),
                text=[
                    f"Sector: {s}<br>Pop: {p}<br>|Δ|: {d:.3f}"
                    for s, p, d in zip(
                        valid["sector_id"],
                        valid["population"],
                        valid["shrinkage"],
                    )
                ],
                hoverinfo="text",
            )
        )
    fig_shrink.update_layout(
        title=dict(text="|Crude − EB| vs Population", font=dict(size=13)),
        xaxis_title="Population",
        yaxis_title="|Crude − EB| (per 1k)",
        margin=dict(l=50, r=20, t=40, b=50),
        height=420,
        xaxis_type="log",
    )

    # --- Histogram overlay ---
    fig_hist = go.Figure()
    pos = wdf[wdf["cases_per_1000"] > 0]
    if not pos.empty:
        fig_hist.add_trace(
            go.Histogram(
                x=pos["cases_per_1000"],
                nbinsx=60,
                name="Crude",
                opacity=0.6,
                marker_color="#3498DB",
            )
        )
        fig_hist.add_trace(
            go.Histogram(
                x=pos["eb_rate_per_1000"],
                nbinsx=60,
                name="EB Smoothed",
                opacity=0.6,
                marker_color="#E74C3C",
            )
        )
    fig_hist.update_layout(
        barmode="overlay",
        title=dict(
            text="Rate Distribution (positive only)", font=dict(size=13)
        ),
        xaxis_title="Rate (per 1k)",
        yaxis_title="Count",
        margin=dict(l=50, r=20, t=40, b=50),
        height=420,
        legend=dict(x=0.7, y=0.95),
    )

    return fig_crude, fig_eb, fig_scatter, fig_shrink, fig_hist


# =====================================================================
# RUN
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting EB Comparison Dashboard")
    print("Open: http://127.0.0.1:8060/")
    print("=" * 70 + "\n")
    app.run(debug=True, port=8060)
