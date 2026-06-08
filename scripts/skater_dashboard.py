"""SKATER results dashboard — two interactive HTML files.

dashboard_map.html
  Choropleth map of BH census sectors coloured by cluster assignment.
  A slider lets you sweep through C=2..C_max to see how the partition
  evolves.  Uses a single Choroplethmap trace + slider 'restyle' (updates
  only the z array, not the full GeoJSON) to keep file size to ~14 MB.

dashboard_analysis.html
  Two-panel view:
    Top:    Q-vs-C trajectory — shows how the global objective improves
            as more clusters are added.
    Bottom: Per-cluster normalised eggs (solid) and dengue rate (dotted)
            time series over epidemic biweeks.  Eggs and dengue are each
            min-max normalised to [0,1] so they can be overlaid on the
            same axis to visualise the lagged relationship.  A dropdown
            selects which C value to inspect.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────
RESULTS = Path("results/skater")
GEOJSON_PATH = Path(
    "data/dvc/process_population_data/"
    "bh_sectors_2022_with_populations.geojson"
)
EGGS_PATH = Path(
    "data/dvc/add_population_info/sector_centroids_with_idw.csv"
)
DENGUE_PATH = Path(
    "data/dvc/add_population_info/dengue_per_capita.csv"
)
BH_CENTER = {"lat": -19.917, "lon": -43.934}

# Biweek year prefixes that count as epidemic years in BH
EPIDEMIC_YEARS = ["2012_13", "2015_16", "2018_19", "2023_24"]

# 30 visually distinct colours — cycled for cluster IDs beyond 30
CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d", "#666666", "#1f78b4", "#b2df8a",
    "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
]


# ── Data loaders ──────────────────────────────────────────────────────

def _load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSV outputs produced by src/skater/run.py.

    Returns:
        asgn: cluster_assignments.csv — sector → cluster_id per C value.
        traj: q_trajectory.csv        — Q score at each C.
        diag: cluster_diagnostics.csv — per-cluster stats (q_c, best_k, …).
    """
    asgn = pd.read_csv(
        RESULTS / "cluster_assignments.csv", dtype={"sector_id": str}
    )
    traj = pd.read_csv(RESULTS / "q_trajectory.csv")
    diag = pd.read_csv(RESULTS / "cluster_diagnostics.csv")
    return asgn, traj, diag


def _load_geojson() -> dict:
    """Load the BH census sector GeoJSON FeatureCollection."""
    with open(GEOJSON_PATH) as fh:
        return json.load(fh)


def _load_sector_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw sector-level egg and dengue time series.

    Returns:
        eggs:   DataFrame [sector_id, biweek, idw_egg_value]
        dengue: DataFrame [sector_id, biweek, eb_rate_per_1000, population]
    """
    eggs = pd.read_csv(
        EGGS_PATH,
        usecols=["CD_SETOR", "biweek", "idw_egg_value"],
        dtype={"CD_SETOR": str},
    ).rename(columns={"CD_SETOR": "sector_id"})

    dengue = pd.read_csv(
        DENGUE_PATH,
        usecols=["sector_id", "biweek", "eb_rate_per_1000", "population"],
        dtype={"sector_id": str},
    )
    return eggs, dengue


# ── View 1: choropleth map + C slider ─────────────────────────────────

def _discrete_colorscale(n: int) -> list[list]:
    """Build a Plotly colorscale that maps integer cluster IDs to distinct colours.

    Plotly's choropleth expects a continuous [0,1] colorscale.  We create
    one stop per cluster so each integer cluster_id maps to a unique colour.

    Args:
        n: Number of distinct clusters (= max cluster_id + 1).

    Returns:
        List of [position, colour] pairs for Plotly colorscale format.
    """
    if n <= 1:
        return [[0.0, CLUSTER_COLORS[0]], [1.0, CLUSTER_COLORS[0]]]
    stops = []
    for k in range(n):
        col = CLUSTER_COLORS[k % 30]
        stops.append([k / (n - 1), col])
    return stops


def _z_for_c(asgn: pd.DataFrame, sector_order: list[str], c: int) -> list:
    """Extract cluster_id values aligned to the GeoJSON sector order.

    Args:
        asgn:         Full assignments DataFrame (all C values).
        sector_order: Sector IDs in the order GeoJSON features appear.
        c:            The C value (number of clusters) to extract.

    Returns:
        List of float cluster IDs aligned to sector_order.
        Sectors missing from the assignments get cluster 0.
    """
    df = asgn[asgn["C"] == c].set_index("sector_id")["cluster_id"]
    return [float(df.get(s, 0)) for s in sector_order]


def build_map_figure(
    asgn: pd.DataFrame, geojson: dict, diag: pd.DataFrame
) -> go.Figure:
    """Build the cluster choropleth map with a C-value slider.

    Uses a single Choroplethmap trace.  The slider updates only the z
    array (cluster ID per sector) via Plotly 'restyle' — this avoids
    embedding the full GeoJSON once per C value, keeping file size small.

    Args:
        asgn:    Cluster assignments DataFrame [C, sector_id, cluster_id].
        geojson: GeoJSON FeatureCollection used as the map base.
        diag:    Cluster diagnostics (unused here, available for extension).

    Returns:
        A go.Figure with choropleth + slider layout.
    """
    c_values = sorted(asgn["C"].unique().tolist())
    c0 = c_values[0]
    n_max = int(asgn["cluster_id"].max()) + 1

    # GeoJSON sector order determines which z value maps to which polygon
    sector_order = [
        str(f["properties"]["CD_SETOR"]) for f in geojson["features"]
    ]

    z_init = _z_for_c(asgn, sector_order, c0)

    choro = go.Choroplethmap(
        geojson=geojson,
        locations=sector_order,
        z=z_init,
        featureidkey="properties.CD_SETOR",
        colorscale=_discrete_colorscale(n_max),
        zmin=0,
        zmax=n_max - 1,
        marker_opacity=0.75,
        marker_line_width=0.1,
        showscale=False,
        name="Clusters",
        hovertemplate="<b>%{location}</b><br>cluster %{z:.0f}<extra></extra>",
    )

    # Each slider step only swaps the z array — not the whole trace
    steps = [
        {
            "label": str(c),
            "method": "restyle",
            "args": [{"z": [_z_for_c(asgn, sector_order, c)]}],
        }
        for c in c_values
    ]

    fig = go.Figure(data=[choro])
    fig.update_layout(
        map={
            "style": "open-street-map",
            "zoom": 11,
            "center": BH_CENTER,
        },
        sliders=[{
            "active": 0,
            "steps": steps,
            "x": 0.05,
            "len": 0.9,
            "y": 0.0,
            "currentvalue": {"prefix": "C = ", "visible": True},
        }],
        height=900,
        margin={"t": 60, "b": 80, "l": 20, "r": 20},
        title=f"SKATER Clusters — C={c0}",
    )
    return fig


# ── View 2: Q trajectory + per-cluster time series ────────────────────

def _agg_cluster_series(
    asgn: pd.DataFrame,
    eggs: pd.DataFrame,
    dengue: pd.DataFrame,
    c_val: int,
    diag: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate sector-level time series into cluster-level series for one C.

    Eggs: simple mean across sectors (each ovitrap has equal weight).
    Dengue: population-weighted mean so large sectors don't dominate.

    The result is joined with diagnostics (best_k, q_c) so the time series
    plots can show the optimal lag and correlation score in their tooltips.

    Args:
        asgn:  Cluster assignments for all C values.
        eggs:  Sector-level IDW egg counts [sector_id, biweek, idw_egg_value].
        dengue: Sector-level EB dengue rates [sector_id, biweek, eb_rate_per_1000, population].
        c_val: The C value to aggregate.
        diag:  Cluster diagnostics [C, cluster_id, best_k, q_c, …].

    Returns:
        DataFrame [cluster_id, biweek, eggs, dengue, best_k, q_c].
    """
    c_asgn = asgn[asgn["C"] == c_val][["sector_id", "cluster_id"]]
    eggs_m = eggs.merge(c_asgn, on="sector_id")
    dengue_m = dengue.merge(c_asgn, on="sector_id")

    # Mean IDW eggs per cluster per biweek
    eggs_agg = (
        eggs_m.groupby(["cluster_id", "biweek"])["idw_egg_value"]
        .mean()
        .reset_index()
        .rename(columns={"idw_egg_value": "eggs"})
    )

    # Population-weighted dengue rate per cluster per biweek
    dng_agg = (
        dengue_m.assign(
            pop_rate=lambda d: d["population"] * d["eb_rate_per_1000"]
        )
        .groupby(["cluster_id", "biweek"])
        .apply(
            lambda g: g["pop_rate"].sum() / g["population"].sum(),
            include_groups=False,
        )
        .reset_index(name="dengue")
    )

    result = eggs_agg.merge(dng_agg, on=["cluster_id", "biweek"])

    # Join best_k and q_c so they appear in hover tooltips
    bk = diag[diag["C"] == c_val][["cluster_id", "best_k", "q_c"]]
    return result.merge(bk, on="cluster_id", how="left")


def build_analysis_figure(
    traj: pd.DataFrame,
    asgn: pd.DataFrame,
    eggs: pd.DataFrame,
    dengue: pd.DataFrame,
    diag: pd.DataFrame,
) -> go.Figure:
    """Build the two-panel analysis figure (Q trajectory + time series).

    All time-series traces are created upfront but start hidden.
    The dropdown menu shows/hides the subset corresponding to the chosen C.

    Eggs and dengue are both min-max normalised to [0,1] per cluster so
    they can be plotted on the same axis.  Eggs = solid line, dengue = dotted.
    Matching colours link each eggs trace to its dengue counterpart.

    Args:
        traj:   Q-trajectory DataFrame [C, Q].
        asgn:   Cluster assignments [C, sector_id, cluster_id].
        eggs:   Raw sector egg series.
        dengue: Raw sector dengue series.
        diag:   Per-cluster diagnostics.

    Returns:
        A go.Figure with two subplots and a C-value dropdown.
    """
    c_values = sorted(asgn["C"].unique().tolist())

    # Identify epidemic biweeks for filtering the time series plot
    epic_bws = set(
        bw for bw in eggs["biweek"].unique()
        if any(bw.startswith(yr) for yr in EPIDEMIC_YEARS)
    )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Q-vs-C trajectory", "Per-cluster series"],
        vertical_spacing=0.12,
        row_heights=[0.3, 0.7],
    )

    # ── Top panel: Q trajectory ───────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=traj["C"], y=traj["Q"],
            mode="lines+markers",
            marker={"size": 6},
            line={"color": "steelblue"},
            name="Q",
            hovertemplate="C=%{x}<br>Q=%{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── Bottom panel: per-cluster time series ─────────────────────────
    # Build all traces upfront (hidden), then use the dropdown to reveal
    # only the C-value subset the user selects.
    all_series_traces: list[go.Scatter] = []
    # vis_map[c] = list of trace indices (in all_series_traces) for that C
    vis_map: dict[int, list[int]] = {}

    for c in c_values:
        series = _agg_cluster_series(asgn, eggs, dengue, c, diag)
        # Show only epidemic biweeks where the lagged correlation was computed
        epic = series[series["biweek"].isin(epic_bws)].copy()
        n_new = 0

        for cid in sorted(epic["cluster_id"].unique()):
            sub = epic[epic["cluster_id"] == cid].sort_values("biweek")
            bk = int(sub["best_k"].iloc[0]) if not sub.empty else 0
            qc = float(sub["q_c"].iloc[0]) if not sub.empty else 0.0
            color = CLUSTER_COLORS[cid % 30]

            # Min-max normalise eggs to [0,1] for overlay with dengue
            e_vals = sub["eggs"].values
            e_norm = (e_vals - np.nanmin(e_vals)) / (
                np.nanmax(e_vals) - np.nanmin(e_vals) + 1e-12
            )
            all_series_traces.append(
                go.Scatter(
                    x=sub["biweek"], y=e_norm,
                    mode="lines",
                    line={"color": color, "dash": "solid"},
                    name=f"C={c} cid={cid} eggs",
                    visible=False,
                    legendgroup=f"c{c}_cid{cid}",
                    showlegend=True,
                    hovertemplate=(
                        f"Cluster {cid} eggs (norm)<br>"
                        f"lag={bk} q={qc:.3f}<extra></extra>"
                    ),
                )
            )

            # Min-max normalise dengue to [0,1]; same colour, dotted line
            d_vals = sub["dengue"].values
            d_norm = (d_vals - np.nanmin(d_vals)) / (
                np.nanmax(d_vals) - np.nanmin(d_vals) + 1e-12
            )
            all_series_traces.append(
                go.Scatter(
                    x=sub["biweek"], y=d_norm,
                    mode="lines",
                    line={"color": color, "dash": "dot"},
                    name=f"C={c} cid={cid} dengue",
                    visible=False,
                    legendgroup=f"c{c}_cid{cid}",
                    showlegend=False,
                    hovertemplate=(
                        f"Cluster {cid} dengue (norm, lag={bk})"
                        f"<extra></extra>"
                    ),
                )
            )
            n_new += 2

        # Record which trace indices belong to this C value
        vis_map[c] = list(range(
            len(all_series_traces) - n_new, len(all_series_traces)
        ))

    for tr in all_series_traces:
        fig.add_trace(tr, row=2, col=1)

    # ── Dropdown: show/hide traces for selected C ─────────────────────
    # The Q trace (index 0) is always visible; series traces follow.
    n_q_traces = 1
    n_total_series = len(all_series_traces)
    buttons = []
    for c in c_values:
        vis = [True]  # keep Q trace visible
        series_vis = [False] * n_total_series
        for idx in vis_map[c]:
            series_vis[idx - n_q_traces] = True
        vis += series_vis
        buttons.append({
            "label": f"C = {c}",
            "method": "update",
            "args": [
                {"visible": vis},
                {"title": f"SKATER Analysis — C={c}"},
            ],
        })

    # Pre-reveal the first C value's traces
    for idx in vis_map[c_values[0]]:
        all_series_traces[idx - n_q_traces].visible = True

    fig.update_layout(
        title=f"SKATER Analysis — C={c_values[0]}",
        height=1100,
        xaxis2_title="Biweek",
        yaxis2_title="Normalised value",
        xaxis_title="C (number of clusters)",
        yaxis_title="Q",
        updatemenus=[{
            "type": "dropdown",
            "direction": "down",
            "showactive": True,
            "buttons": buttons,
            "x": 0.01,
            "xanchor": "left",
            "y": 0.62,
            "yanchor": "top",
        }],
        margin={"t": 80, "b": 60, "l": 60, "r": 20},
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Load all data, build both figures, write HTML files."""
    logger.info("Loading SKATER results…")
    asgn, traj, diag = _load_results()
    geojson = _load_geojson()
    eggs, dengue = _load_sector_series()

    logger.info("Building map dashboard…")
    fig_map = build_map_figure(asgn, geojson, diag)
    out_map = RESULTS / "dashboard_map.html"
    fig_map.write_html(str(out_map))
    logger.info("Saved %s", out_map)

    logger.info("Building analysis dashboard…")
    fig_ana = build_analysis_figure(traj, asgn, eggs, dengue, diag)
    out_ana = RESULTS / "dashboard_analysis.html"
    fig_ana.write_html(str(out_ana))
    logger.info("Saved %s", out_ana)


if __name__ == "__main__":
    main()
