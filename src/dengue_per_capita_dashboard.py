"""
Dengue Per Capita Dashboard - Belo Horizonte

Interactive dashboard to visualise dengue incidence (cases per 1,000 pop.)
by census sector and epidemic week.  Includes:

- Choropleth map with blue-to-red gradient
- Week selector (slider)
- Distribution explorer: histogram, box plot by year, scatter (cases vs pop.)
- Edge-cases table with top/bottom sectors
- Customisable threshold controls to highlight sectors on the map
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table

# =============================================================================
# CONSTANTS
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PER_CAPITA_CSV = PROCESSED_DIR / "dengue_per_capita.csv"
GEOJSON_PATH = PROCESSED_DIR / "bh_sectors_2022_with_populations.geojson"

MAP_CENTER_LAT = -19.9167
MAP_CENTER_LON = -43.9345
MAP_ZOOM = 11
MAPBOX_STYLE = "carto-positron"

BLUE_RED_COLORSCALE = [
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

# =============================================================================
# DATA LOADING
# =============================================================================

print("Loading dengue per-capita data ...")
pc_df = pd.read_csv(PER_CAPITA_CSV)
pc_df["sector_id"] = pc_df["sector_id"].astype(str)
pc_df["epidemic_date"] = pc_df["epidemic_date"].astype(str)

# Extract year from epidemic_date for box-plot grouping
pc_df["epi_year"] = pc_df["epidemic_date"].str.split("W").str[0]

# Sorted unique weeks for the slider
ALL_WEEKS = sorted(pc_df["epidemic_date"].unique())
WEEK_TO_IDX = {w: i for i, w in enumerate(ALL_WEEKS)}

print("Loading GeoJSON sectors ...")
sectors_gdf = gpd.read_file(GEOJSON_PATH)
sectors_gdf["CD_SETOR"] = sectors_gdf["CD_SETOR"].astype(str)
GEOJSON_DATA = json.loads(sectors_gdf.to_json())

print(
    f"  {len(pc_df):,} per-capita rows  |  {len(ALL_WEEKS)} weeks  |  {len(sectors_gdf)} sectors"
)

# =============================================================================
# DASH APP
# =============================================================================

app = Dash(__name__)
app.title = "Dengue Per Capita - Belo Horizonte"

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = html.Div(
    [
        # --- Header ---
        html.H1(
            "Dengue Incidence Dashboard — Belo Horizonte",
            style={
                "textAlign": "center",
                "color": "#2C3E50",
                "marginBottom": 5,
            },
        ),
        html.P(
            f"{len(ALL_WEEKS)} epidemic weeks  ·  {sectors_gdf.shape[0]} census sectors  ·  "
            "rate = cases per 1,000 population",
            style={
                "textAlign": "center",
                "color": "#888",
                "fontSize": 13,
                "marginBottom": 15,
            },
        ),
        # --- Week selector ---
        html.Div(
            [
                html.Label(
                    "Select Epidemic Week:", style={"fontWeight": "bold"}
                ),
                dcc.Slider(
                    id="week-slider",
                    min=0,
                    max=len(ALL_WEEKS) - 1,
                    value=0,
                    marks={
                        i: w
                        for i, w in enumerate(ALL_WEEKS)
                        if i % max(1, len(ALL_WEEKS) // 15) == 0
                    },
                    step=1,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ],
            style={"margin": "0 40px 10px 40px"},
        ),
        # --- Threshold controls ---
        html.Div(
            [
                html.Details(
                    [
                        html.Summary(
                            "Threshold Controls",
                            style={
                                "fontWeight": "bold",
                                "cursor": "pointer",
                                "fontSize": 14,
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Min per-capita threshold (highlight ≥):"
                                        ),
                                        dcc.Input(
                                            id="threshold-min",
                                            type="number",
                                            value=5,
                                            min=0,
                                            step=0.5,
                                            style={
                                                "width": 80,
                                                "marginLeft": 8,
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "marginRight": 30,
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Max per-capita threshold (highlight ≤):"
                                        ),
                                        dcc.Input(
                                            id="threshold-max",
                                            type="number",
                                            value=0,
                                            min=0,
                                            step=0.1,
                                            style={
                                                "width": 80,
                                                "marginLeft": 8,
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "marginRight": 30,
                                    },
                                ),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="threshold-checks",
                                            options=[
                                                {
                                                    "label": " Highlight zero-case sectors",
                                                    "value": "zero_cases",
                                                },
                                                {
                                                    "label": " Highlight low-population sectors (<100)",
                                                    "value": "low_pop",
                                                },
                                            ],
                                            value=[],
                                            inline=True,
                                            style={"marginTop": 6},
                                        ),
                                    ]
                                ),
                            ],
                            style={"padding": "10px 0"},
                        ),
                    ],
                    open=False,
                ),
            ],
            style={
                "margin": "0 40px 10px 40px",
                "background": "#f8f9fa",
                "padding": "8px 16px",
                "borderRadius": 6,
            },
        ),
        # --- Summary stats ---
        html.Div(
            id="summary-stats",
            style={"margin": "0 40px 10px 40px", "fontSize": 13},
        ),
        # --- Map ---
        html.Div(
            dcc.Graph(id="choropleth-map", style={"height": "600px"}),
            style={"margin": "0 20px"},
        ),
        # --- Distribution explorer ---
        html.H2(
            "Distribution Explorer",
            style={
                "textAlign": "center",
                "marginTop": 30,
                "color": "#2C3E50",
            },
        ),
        html.Div(
            [
                # Histogram + Box plot side by side
                html.Div(
                    dcc.Graph(id="histogram"),
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    dcc.Graph(id="boxplot-year"),
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ]
        ),
        html.Div(
            [
                # Scatter + Edge-cases table side by side
                html.Div(
                    dcc.Graph(id="scatter-cases-pop"),
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    [
                        html.H4(
                            "Top 20 Sectors by Incidence Rate",
                            style={"textAlign": "center"},
                        ),
                        dash_table.DataTable(
                            id="edge-table",
                            style_table={
                                "overflowX": "auto",
                                "maxHeight": "380px",
                                "overflowY": "auto",
                            },
                            style_cell={
                                "textAlign": "left",
                                "fontSize": 12,
                                "padding": "4px 8px",
                            },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#f0f0f0",
                            },
                            page_size=20,
                        ),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "0 20px",
                    },
                ),
            ]
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "padding": 10},
)

# =============================================================================
# CALLBACKS
# =============================================================================


def _week_data(week: str) -> pd.DataFrame:
    """Return per-capita rows for a single epidemic week."""
    return pc_df[pc_df["epidemic_date"] == week].copy()


def _threshold_mask(
    df: pd.DataFrame,
    min_thresh: float,
    max_thresh: float,
    checks: list[str],
) -> pd.Series:
    """Build a boolean mask for sectors meeting ANY threshold criterion."""
    mask = pd.Series(False, index=df.index)
    if min_thresh is not None and min_thresh > 0:
        mask |= df["cases_per_1000"].fillna(0) >= min_thresh
    if max_thresh is not None and max_thresh > 0:
        mask |= (df["cases_per_1000"].fillna(0) <= max_thresh) & (
            df["cases_per_1000"].notna()
        )
    if "zero_cases" in (checks or []):
        mask |= df["case_count"] == 0
    if "low_pop" in (checks or []):
        mask |= df["population"].fillna(0) < 100
    return mask


@app.callback(
    Output("choropleth-map", "figure"),
    Output("summary-stats", "children"),
    Output("histogram", "figure"),
    Output("scatter-cases-pop", "figure"),
    Output("edge-table", "data"),
    Output("edge-table", "columns"),
    Input("week-slider", "value"),
    Input("threshold-min", "value"),
    Input("threshold-max", "value"),
    Input("threshold-checks", "value"),
)
def update_dashboard(slider_idx, thresh_min, thresh_max, thresh_checks):
    week = ALL_WEEKS[slider_idx]
    wdf = _week_data(week)

    # ---- Choropleth map ----
    merged = sectors_gdf[["CD_SETOR"]].merge(
        wdf[["sector_id", "case_count", "population", "cases_per_1000"]],
        left_on="CD_SETOR",
        right_on="sector_id",
        how="left",
    )
    merged["cases_per_1000"] = merged["cases_per_1000"].fillna(0)
    merged["case_count"] = merged["case_count"].fillna(0).astype(int)
    merged["population"] = merged["population"].fillna(0).astype(int)

    hover = [
        f"Sector: {s}<br>Cases: {c}<br>Pop: {p}<br>Rate: {r:.2f} ‰"
        for s, c, p, r in zip(
            merged["CD_SETOR"],
            merged["case_count"],
            merged["population"],
            merged["cases_per_1000"],
        )
    ]

    max_rate = max(merged["cases_per_1000"].quantile(0.95), 1)

    fig_map = go.Figure(
        go.Choroplethmapbox(
            geojson=GEOJSON_DATA,
            locations=merged["CD_SETOR"].tolist(),
            z=merged["cases_per_1000"].tolist(),
            featureidkey="properties.CD_SETOR",
            colorscale=BLUE_RED_COLORSCALE,
            zmin=0,
            zmax=max_rate,
            marker={
                "opacity": 0.8,
                "line": {"width": 0.3, "color": "#444"},
            },
            hovertext=hover,
            hoverinfo="text",
            colorbar=dict(title="Cases/1k", thickness=15, len=0.6),
            name="Incidence",
        )
    )

    # Highlight threshold sectors
    tmask = _threshold_mask(wdf, thresh_min, thresh_max, thresh_checks)
    highlighted = wdf[tmask]
    if not highlighted.empty:
        hl_geo = sectors_gdf[
            sectors_gdf["CD_SETOR"].isin(highlighted["sector_id"])
        ]
        if not hl_geo.empty:
            centroids = hl_geo.geometry.centroid
            fig_map.add_trace(
                go.Scattermapbox(
                    lat=centroids.y.tolist(),
                    lon=centroids.x.tolist(),
                    mode="markers",
                    marker=dict(
                        size=9,
                        color="yellow",
                        opacity=0.9,
                        symbol="circle",
                    ),
                    hovertext=[
                        f"Threshold hit — {s}" for s in hl_geo["CD_SETOR"]
                    ],
                    hoverinfo="text",
                    name="Threshold",
                )
            )

    fig_map.update_layout(
        mapbox=dict(
            style=MAPBOX_STYLE,
            center=dict(lat=MAP_CENTER_LAT, lon=MAP_CENTER_LON),
            zoom=MAP_ZOOM,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Week: {week}", x=0.5, font=dict(size=15)),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        uirevision="constant",
    )

    # ---- Summary stats ----
    valid = wdf[wdf["cases_per_1000"].notna()]
    total_cases = int(wdf["case_count"].sum())
    mean_rate = valid["cases_per_1000"].mean() if len(valid) else 0
    median_rate = valid["cases_per_1000"].median() if len(valid) else 0
    n_threshold = int(tmask.sum())
    summary = html.Div(
        [
            html.Span(
                f"Week: {week}",
                style={"fontWeight": "bold", "marginRight": 20},
            ),
            html.Span(
                f"Total cases: {total_cases:,}", style={"marginRight": 20}
            ),
            html.Span(
                f"Mean rate: {mean_rate:.3f}", style={"marginRight": 20}
            ),
            html.Span(
                f"Median rate: {median_rate:.3f}",
                style={"marginRight": 20},
            ),
            html.Span(
                f"Sectors meeting threshold: {n_threshold}",
                style={"color": "#E74C3C", "fontWeight": "bold"},
            ),
        ]
    )

    # ---- Histogram ----
    fig_hist = px.histogram(
        wdf[wdf["cases_per_1000"].notna()],
        x="cases_per_1000",
        nbins=50,
        title=f"Rate Distribution — {week}",
        labels={"cases_per_1000": "Cases per 1,000"},
        color_discrete_sequence=["#2980B9"],
    )
    fig_hist.update_layout(
        yaxis_title="Sector count",
        margin=dict(l=40, r=20, t=40, b=40),
        height=380,
    )

    # ---- Scatter: cases vs population ----
    scatter_df = wdf[
        (wdf["population"] > 0) & (wdf["case_count"] > 0)
    ].copy()
    fig_scatter = px.scatter(
        scatter_df,
        x="population",
        y="case_count",
        color="cases_per_1000",
        color_continuous_scale="RdBu_r",
        hover_data=["sector_id", "cases_per_1000"],
        title=f"Cases vs Population — {week}",
        labels={
            "population": "Population",
            "case_count": "Cases",
            "cases_per_1000": "Rate",
        },
    )
    fig_scatter.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        height=380,
    )

    # ---- Edge cases table (top sectors only) ----
    edge = wdf[wdf["cases_per_1000"].notna()].sort_values(
        "cases_per_1000", ascending=False
    )
    top = edge.head(20).copy()
    top["cases_per_1000"] = top["cases_per_1000"].round(3)
    top = top[["sector_id", "case_count", "population", "cases_per_1000"]]
    table_data = top.to_dict("records")
    table_cols = [{"name": c, "id": c} for c in top.columns]

    return fig_map, summary, fig_hist, fig_scatter, table_data, table_cols


# ---- Box plot by year (independent of week slider) ----
@app.callback(
    Output("boxplot-year", "figure"),
    Input("week-slider", "value"),
)
def update_boxplot(_):
    yearly = pc_df[
        pc_df["cases_per_1000"].notna() & (pc_df["cases_per_1000"] > 0)
    ].copy()
    fig_box = px.box(
        yearly,
        x="epi_year",
        y="cases_per_1000",
        hover_data=["epidemic_date", "sector_id"],
        title="Per-Capita Rate Distribution by Year",
        labels={
            "epi_year": "Epidemic Year",
            "cases_per_1000": "Cases per 1,000",
            "epidemic_date": "Week",
        },
        color_discrete_sequence=["#E74C3C"],
    )
    fig_box.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        height=380,
        xaxis_tickangle=-45,
    )
    return fig_box


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting Dengue Per Capita Dashboard")
    print("Open: http://127.0.0.1:8053/")
    print("=" * 70 + "\n")
    app.run(debug=True, port=8053)
