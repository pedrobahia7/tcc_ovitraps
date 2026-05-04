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
pc_df["biweek"] = pc_df["biweek"].astype(str)

# Extract year from biweek for box-plot grouping
pc_df["epi_year"] = pc_df["biweek"].str.split("W").str[0]

# Sorted unique biweeks for the slider
ALL_BIWEEKS = sorted(pc_df["biweek"].unique())
BIWEEK_TO_IDX = {w: i for i, w in enumerate(ALL_BIWEEKS)}

print("Loading GeoJSON sectors ...")
sectors_gdf = gpd.read_file(GEOJSON_PATH)
sectors_gdf["CD_SETOR"] = sectors_gdf["CD_SETOR"].astype(str)
GEOJSON_DATA = json.loads(sectors_gdf.to_json())

print(
    f"  {len(pc_df):,} per-capita rows  |  {len(ALL_BIWEEKS)} biweeks  |  {len(sectors_gdf)} sectors"
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
            f"{len(ALL_BIWEEKS)} biweeks  ·  {sectors_gdf.shape[0]} census sectors  ·  "
            "rate = cases per 1,000 population",
            style={
                "textAlign": "center",
                "color": "#888",
                "fontSize": 13,
                "marginBottom": 15,
            },
        ),
        # --- Biweek selector + rate toggle ---
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
                        "width": "75%",
                        "display": "inline-block",
                        "verticalAlign": "middle",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Rate Metric:",
                            style={"fontWeight": "bold", "marginRight": 8},
                        ),
                        dcc.RadioItems(
                            id="rate-metric",
                            options=[
                                {
                                    "label": " Crude",
                                    "value": "cases_per_1000",
                                },
                                {
                                    "label": " EB Smoothed",
                                    "value": "eb_rate_per_1000",
                                },
                            ],
                            value="cases_per_1000",
                            inline=True,
                            style={"fontSize": 14},
                        ),
                    ],
                    style={
                        "width": "20%",
                        "display": "inline-block",
                        "verticalAlign": "middle",
                        "textAlign": "right",
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
        # --- Distribution explorer (paired rows: Crude | EB) ---
        html.H2(
            "Distribution Explorer",
            style={
                "textAlign": "center",
                "marginTop": 30,
                "color": "#2C3E50",
            },
        ),
        # Column headers
        html.Div(
            [
                html.H3(
                    "Crude Rate",
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "textAlign": "center",
                        "color": "#E74C3C",
                        "margin": 0,
                    },
                ),
                html.H3(
                    "EB Smoothed Rate",
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "textAlign": "center",
                        "color": "#2980B9",
                        "margin": 0,
                    },
                ),
            ],
        ),
        # Row 1: Histograms
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="histogram-crude"),
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    dcc.Graph(id="histogram-eb"),
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
        ),
        # Row 2: Box plots by year
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="boxplot-year-crude"),
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    dcc.Graph(id="boxplot-year-eb"),
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
        ),
        # Row 3: Scatter plots
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="scatter-crude"),
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    dcc.Graph(id="scatter-eb"),
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
        ),
        # Row 4: Edge-case tables (flexbox for reliable side-by-side)
        html.Div(
            [
                html.Div(
                    [
                        html.H4(
                            "Top 20 Sectors — Crude Rate",
                            style={"textAlign": "center", "marginTop": 10},
                        ),
                        dash_table.DataTable(
                            id="edge-table-crude",
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
                        "flex": "1",
                        "minWidth": 0,
                        "padding": "0 10px",
                    },
                ),
                html.Div(
                    [
                        html.H4(
                            "Top 20 Sectors — EB Rate",
                            style={"textAlign": "center", "marginTop": 10},
                        ),
                        dash_table.DataTable(
                            id="edge-table-eb",
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
                        "flex": "1",
                        "minWidth": 0,
                        "padding": "0 10px",
                    },
                ),
            ],
            style={"display": "flex", "gap": "10px"},
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "padding": 10},
)

# =============================================================================
# CALLBACKS
# =============================================================================


def _biweek_data(biweek: str) -> pd.DataFrame:
    """Return per-capita rows for a single biweek."""
    return pc_df[pc_df["biweek"] == biweek].copy()


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


def _build_histogram(wdf, col, label, color, biweek):
    """Build a histogram figure for a single rate column."""
    fig = px.histogram(
        wdf[wdf[col].notna()],
        x=col,
        nbins=50,
        title=f"{label} Distribution — {biweek}",
        labels={col: f"{label} per 1,000"},
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        yaxis_title="Sector count",
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
    )
    return fig


def _build_scatter(wdf, col, label, biweek):
    """Build a cases-vs-population scatter for a single rate column."""
    sdf = wdf[(wdf["population"] > 0) & (wdf["case_count"] > 0)].copy()
    fig = px.scatter(
        sdf,
        x="population",
        y="case_count",
        color=col,
        color_continuous_scale="RdBu_r",
        hover_data=["sector_id", col],
        title=f"Cases vs Population — {biweek}",
        labels={
            "population": "Population",
            "case_count": "Cases",
            col: label,
        },
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
    )
    return fig


def _build_edge_table(wdf, col):
    """Build top-20 edge-cases table data + columns for a rate column."""
    edge = wdf[wdf[col].notna()].sort_values(col, ascending=False)
    top = edge.head(20).copy()
    top[col] = top[col].round(3)
    top = top[["sector_id", "case_count", "population", col]]
    return top.to_dict("records"), [
        {"name": c, "id": c} for c in top.columns
    ]


def _build_boxplot(col, label, color):
    """Build a box plot by year for a rate column."""
    yearly = pc_df[pc_df[col].notna() & (pc_df[col] > 0)].copy()
    yearly = yearly.sort_values("epi_year")
    fig = px.box(
        yearly,
        x="epi_year",
        y=col,
        hover_data=["biweek", "sector_id"],
        title=f"{label} Distribution by Year",
        labels={
            "epi_year": "Epidemic Year",
            col: f"{label} per 1,000",
            "biweek": "Biweek",
        },
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
        xaxis_tickangle=-45,
    )
    return fig


@app.callback(
    Output("choropleth-map", "figure"),
    Output("summary-stats", "children"),
    Output("histogram-crude", "figure"),
    Output("histogram-eb", "figure"),
    Output("scatter-crude", "figure"),
    Output("scatter-eb", "figure"),
    Output("edge-table-crude", "data"),
    Output("edge-table-crude", "columns"),
    Output("edge-table-eb", "data"),
    Output("edge-table-eb", "columns"),
    Input("week-slider", "value"),
    Input("rate-metric", "value"),
    Input("threshold-min", "value"),
    Input("threshold-max", "value"),
    Input("threshold-checks", "value"),
)
def update_dashboard(
    slider_idx, rate_col, thresh_min, thresh_max, thresh_checks
):
    biweek = ALL_BIWEEKS[slider_idx]
    wdf = _biweek_data(biweek)

    # ---- Choropleth map (uses selected metric) ----
    rate_label = (
        "EB Rate" if rate_col == "eb_rate_per_1000" else "Crude Rate"
    )
    merged = sectors_gdf[["CD_SETOR"]].merge(
        wdf[["sector_id", "case_count", "population", rate_col]],
        left_on="CD_SETOR",
        right_on="sector_id",
        how="left",
    )
    merged[rate_col] = merged[rate_col].fillna(0)
    merged["case_count"] = merged["case_count"].fillna(0).astype(int)
    merged["population"] = merged["population"].fillna(0).astype(int)

    hover = [
        f"Sector: {s}<br>Cases: {c}<br>Pop: {p}<br>{rate_label}: {r:.2f} ‰"
        for s, c, p, r in zip(
            merged["CD_SETOR"],
            merged["case_count"],
            merged["population"],
            merged[rate_col],
        )
    ]

    max_rate = max(merged[rate_col].quantile(0.95), 1)

    fig_map = go.Figure(
        go.Choroplethmapbox(
            geojson=GEOJSON_DATA,
            locations=merged["CD_SETOR"].tolist(),
            z=merged[rate_col].tolist(),
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
            colorbar=dict(title=rate_label, thickness=15, len=0.6),
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
        title=dict(text=f"Biweek: {biweek}", x=0.5, font=dict(size=15)),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        uirevision="constant",
    )

    # ---- Summary stats ----
    total_cases = int(wdf["case_count"].sum())
    crude_valid = wdf[wdf["cases_per_1000"].notna()]
    eb_valid = wdf[wdf["eb_rate_per_1000"].notna()]
    n_threshold = int(tmask.sum())
    summary = html.Div(
        [
            html.Span(
                f"Biweek: {biweek}",
                style={"fontWeight": "bold", "marginRight": 20},
            ),
            html.Span(
                f"Total cases: {total_cases:,}", style={"marginRight": 20}
            ),
            html.Span(
                f"Crude mean: {crude_valid['cases_per_1000'].mean():.3f}",
                style={"marginRight": 20},
            ),
            html.Span(
                f"EB mean: {eb_valid['eb_rate_per_1000'].mean():.3f}",
                style={"marginRight": 20},
            ),
            html.Span(
                f"Sectors meeting threshold: {n_threshold}",
                style={"color": "#E74C3C", "fontWeight": "bold"},
            ),
        ]
    )

    # ---- Distribution charts (both columns) ----
    fig_hist_crude = _build_histogram(
        wdf, "cases_per_1000", "Crude Rate", "#E74C3C", biweek
    )
    fig_hist_eb = _build_histogram(
        wdf, "eb_rate_per_1000", "EB Rate", "#2980B9", biweek
    )

    fig_scatter_crude = _build_scatter(
        wdf, "cases_per_1000", "Crude Rate", biweek
    )
    fig_scatter_eb = _build_scatter(
        wdf, "eb_rate_per_1000", "EB Rate", biweek
    )

    table_crude_data, table_crude_cols = _build_edge_table(
        wdf, "cases_per_1000"
    )
    table_eb_data, table_eb_cols = _build_edge_table(
        wdf, "eb_rate_per_1000"
    )

    return (
        fig_map,
        summary,
        fig_hist_crude,
        fig_hist_eb,
        fig_scatter_crude,
        fig_scatter_eb,
        table_crude_data,
        table_crude_cols,
        table_eb_data,
        table_eb_cols,
    )


# ---- Box plots by year (independent of week slider) ----
@app.callback(
    Output("boxplot-year-crude", "figure"),
    Output("boxplot-year-eb", "figure"),
    Input("week-slider", "value"),
)
def update_boxplots(_):
    fig_crude = _build_boxplot("cases_per_1000", "Crude Rate", "#E74C3C")
    fig_eb = _build_boxplot("eb_rate_per_1000", "EB Rate", "#2980B9")
    return fig_crude, fig_eb


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting Dengue Per Capita Dashboard")
    print("Open: http://127.0.0.1:8054/")
    print("=" * 70 + "\n")
    app.run(debug=True, port=8054)
