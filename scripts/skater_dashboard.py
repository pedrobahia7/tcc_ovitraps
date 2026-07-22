"""SKATER results dashboard — two interactive HTML files.

dashboard_map.html
  Choropleth map of BH census sectors coloured by cluster assignment,
  with ovitrap locations shown as fixed black dots.  A slider sweeps
  C=2..C_max and simultaneously updates:
    1. The choropleth z-array (cluster colours on the map).
    2. A 5-panel bar chart below the map showing q_c per cluster for
       the window [C-2, C-1, C, C+1, C+2].  The center panel (C)
       renders at full opacity; neighbours at 70 %.  All panels share
       a fixed y-axis range for direct comparison.

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
import yaml
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────
RESULTS_BASE = Path("results/skater")
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
OVITRAP_PATH = Path("data/processed/ovitraps_data.csv")
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

def _load_results(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSV outputs produced by src/skater/run.py.

    Args:
        results_dir: Run-specific output directory (results/skater/<run_label>/).

    Returns:
        asgn: cluster_assignments.csv — sector → cluster_id per C value.
        traj: q_trajectory.csv        — Q score at each C.
        diag: cluster_diagnostics.csv — per-cluster stats (q_c, best_k, …).
    """
    asgn = pd.read_csv(
        results_dir / "cluster_assignments.csv", dtype={"sector_id": str}
    )
    traj = pd.read_csv(results_dir / "q_trajectory.csv")
    diag = pd.read_csv(results_dir / "cluster_diagnostics.csv")
    return asgn, traj, diag


def _load_geojson() -> dict:
    """Load the BH census sector GeoJSON FeatureCollection."""
    with open(GEOJSON_PATH) as fh:
        return json.load(fh)


def _load_ovitrap_locations() -> pd.DataFrame:
    """Load unique ovitrap deployment coordinates.

    Reads every row of ovitraps_data.csv but keeps only the first
    occurrence of each trap ID so each physical trap contributes
    one dot to the map.

    Returns:
        DataFrame [idarmad, latitude, longitude] — one row per trap.
    """
    df = pd.read_csv(
        OVITRAP_PATH,
        usecols=["idarmad", "latitude", "longitude"],
        low_memory=False,
    )
    return df.drop_duplicates("idarmad").reset_index(drop=True)


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


def _build_bar_traces(
    diag: pd.DataFrame,
    c_values: list[int],
) -> tuple[list[go.Bar], dict[tuple[int, int], int]]:
    """Pre-build all bar traces for the 5-panel q_c window view.

    For each (c_shown, position) pair that appears in any valid slider
    window, one go.Bar trace is created.  All traces start hidden; the
    slider step reveals exactly the 5 traces for the active window.

    position ∈ {1..5}: 1=C-2, 2=C-1, 3=C (center), 4=C+1, 5=C+2.
    Bars are sorted by q_c descending and coloured by cluster_id.
    Opacity is 1.0 for the center panel and 0.7 for neighbour panels.

    Args:
        diag:     Cluster diagnostics [C, cluster_id, q_c, n_sectors, …].
        c_values: Sorted list of available C values.

    Returns:
        traces:    List of go.Bar, all initially hidden.
        trace_map: Mapping {(c_shown, pos): index_in_traces}.
    """
    c_min, c_max = min(c_values), max(c_values)
    _suf = {1: "", 2: "2", 3: "3", 4: "4", 5: "5"}
    traces: list[go.Bar] = []
    trace_map: dict[tuple[int, int], int] = {}

    for c_shown in c_values:
        c_diag = (
            diag[diag["C"] == c_shown]
            .sort_values("q_c", ascending=False)
            .reset_index(drop=True)
        )
        cluster_ids = c_diag["cluster_id"].tolist()
        q_vals = c_diag["q_c"].tolist()
        n_sec = c_diag["n_sectors"].tolist()
        colors = [CLUSTER_COLORS[cid % 30] for cid in cluster_ids]
        x_lbl = [str(cid) for cid in cluster_ids]

        for pos in range(1, 6):
            # c_sel is the slider value that would put c_shown at this pos
            c_sel = c_shown - pos + 3
            if c_sel < c_min or c_sel > c_max:
                continue

            suf = _suf[pos]
            trace = go.Bar(
                x=x_lbl,
                y=q_vals,
                marker_color=colors,
                opacity=1.0 if pos == 3 else 0.7,
                name=f"C={c_shown}",
                xaxis=f"x{suf}",
                yaxis=f"y{suf}",
                visible=False,
                customdata=list(zip(cluster_ids, n_sec)),
                hovertemplate=(
                    "cluster %{customdata[0]}<br>"
                    "q_c=%{y:.3f}<br>"
                    "sectors=%{customdata[1]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
            trace_map[(c_shown, pos)] = len(traces)
            traces.append(trace)

    return traces, trace_map


def build_combined_figure(
    asgn: pd.DataFrame,
    geojson: dict,
    diag: pd.DataFrame,
    ovitrap_locs: pd.DataFrame,
    metric_label: str = "",
) -> go.Figure:
    """Build the combined map + ovitrap dots + q_c bar chart figure.

    Layout (manual axis domains, no make_subplots):
      • Choropleth map occupies y=[0.32, 1.0] of figure height.
      • Slider sits between map and bars at y≈0.30.
      • 5 bar subplots occupy y=[0.0, 0.25], equally spaced in x.

    The slider (method='update') simultaneously:
      1. Swaps the choropleth z-array for the selected C.
      2. Toggles bar trace visibility for the [C-2,C-1,C,C+1,C+2] window.
      3. Updates each panel's x-axis title to show which C is displayed.

    Center panel renders at full opacity; neighbour panels at 70 %.
    All bar panels share a fixed y-axis range for direct comparison.
    Edge C values (C=2, C=30) leave out-of-range panels empty.

    Args:
        asgn:         Cluster assignments [C, sector_id, cluster_id].
        geojson:      GeoJSON FeatureCollection for BH sectors.
        diag:         Cluster diagnostics [C, cluster_id, q_c, n_sectors, …].
        ovitrap_locs: Unique ovitrap locations [idarmad, latitude, longitude].
        metric_label: Short string appended to all slider titles, e.g.
                      'mst=egg_corr_dist | obj=corr_q'.  Empty → no suffix.

    Returns:
        go.Figure ready to write as standalone HTML.
    """
    c_values = sorted(asgn["C"].unique().tolist())
    c_min, c_max = c_values[0], c_values[-1]
    c0 = c_values[0]
    n_max = int(asgn["cluster_id"].max()) + 1
    max_qc = float(diag["q_c"].max())

    # GeoJSON sector order determines which z value maps to which polygon
    sector_order = [
        str(f["properties"]["CD_SETOR"]) for f in geojson["features"]
    ]

    # ── Fixed map traces ──────────────────────────────────────────────
    choro = go.Choroplethmap(
        geojson=geojson,
        locations=sector_order,
        z=_z_for_c(asgn, sector_order, c0),
        featureidkey="properties.CD_SETOR",
        colorscale=_discrete_colorscale(n_max),
        zmin=0,
        zmax=n_max - 1,
        marker_opacity=0.75,
        marker_line_width=0.1,
        showscale=False,
        name="Clusters",
        hovertemplate=(
            "<b>%{location}</b><br>cluster %{z:.0f}<extra></extra>"
        ),
    )

    scatter_traps = go.Scattermap(
        lat=ovitrap_locs["latitude"],
        lon=ovitrap_locs["longitude"],
        mode="markers",
        marker={"size": 4, "color": "black", "opacity": 0.7},
        name="Ovitraps",
        customdata=ovitrap_locs["idarmad"],
        hovertemplate=(
            "Ovitrap %{customdata}<br>"
            "lat=%{lat:.4f}, lon=%{lon:.4f}"
            "<extra></extra>"
        ),
        showlegend=True,
    )

    # ── Bar traces for 5-panel q_c window ────────────────────────────
    bar_traces, trace_map = _build_bar_traces(diag, c_values)
    n_bars = len(bar_traces)

    # Reveal the initial window (c0 as center)
    for pos in range(1, 6):
        key = (c0 + pos - 3, pos)
        if key in trace_map:
            bar_traces[trace_map[key]].visible = True

    # ── Slider steps ──────────────────────────────────────────────────
    _suf = {1: "", 2: "2", 3: "3", 4: "4", 5: "5"}
    steps = []

    for c_sel in c_values:
        z = _z_for_c(asgn, sector_order, c_sel)

        bar_vis = [False] * n_bars
        for pos in range(1, 6):
            key = (c_sel + pos - 3, pos)
            if key in trace_map:
                bar_vis[trace_map[key]] = True

        # Index 0=choro, 1=ovitraps always True; rest are bar traces
        vis = [True, True] + bar_vis

        _suffix = f" | {metric_label}" if metric_label else ""
        layout_upd: dict = {
            "title.text": f"SKATER — C={c_sel}{_suffix}"
        }
        for pos in range(1, 6):
            c_shown = c_sel + pos - 3
            ax_key = f"xaxis{_suf[pos]}.title.text"
            if c_min <= c_shown <= c_max:
                label = (
                    f"<b>C = {c_shown}</b>" if pos == 3
                    else f"C = {c_shown}"
                )
                layout_upd[ax_key] = label
            else:
                layout_upd[ax_key] = ""

        steps.append({
            "label": str(c_sel),
            "method": "update",
            "args": [{"z": [z], "visible": vis}, layout_upd],
        })

    # ── Bar subplot axis domains — 5 equal panels, gap=0.025 ─────────
    # panel width = (1 - 4×0.025) / 5 = 0.18; total = 5×0.18 + 4×0.025 = 1.0
    # bar_y top set to 0.20 so slider labels at y=0.30 have blank space above
    pw, gap = 0.18, 0.025
    bar_y = [0.0, 0.20]

    ax_layout: dict = {}
    for pos in range(1, 6):
        suf = _suf[pos]
        x0 = round((pos - 1) * (pw + gap), 4)
        x1 = round(x0 + pw, 4)
        ax_layout[f"xaxis{suf}"] = {
            "domain": [x0, x1],
            "anchor": f"y{suf}",
            "title": {"text": ""},  # filled by slider on each step
            "tickvals": [],         # hide cluster-id ticks — too many
        }
        ax_layout[f"yaxis{suf}"] = {
            "domain": bar_y,
            "anchor": f"x{suf}",
            "range": [0.0, max_qc * 1.05],
            "title": {"text": "q_c"} if pos == 1 else {"text": ""},
            "showticklabels": pos == 1,
        }

    # ── Assemble figure ───────────────────────────────────────────────
    fig = go.Figure(data=[choro, scatter_traps, *bar_traces])

    fig.update_layout(
        **ax_layout,
        map={
            "style": "open-street-map",
            "zoom": 11,
            "center": BH_CENTER,
            "domain": {"x": [0, 1], "y": [0.32, 1.0]},
        },
        sliders=[{
            "active": 0,
            "steps": steps,
            "x": 0.05,
            "len": 0.9,
            "y": 0.30,
            "yanchor": "top",
            "currentvalue": {
                "prefix": "C = ",
                "visible": True,
                "xanchor": "center",
            },
        }],
        height=1400,
        margin={"t": 60, "b": 20, "l": 50, "r": 20},
        title={
            "text": (
                f"SKATER — C={c0}"
                + (f" | {metric_label}" if metric_label else "")
            ),
            "x": 0.5,
        },
        legend={
            "x": 0.01, "y": 0.99,
            "xanchor": "left", "yanchor": "top",
        },
    )
    return fig


# ── Size vs metric correlation ────────────────────────────────────────

def _size_metric_correlations(
    diag: pd.DataFrame, c_values: list[int], n_min: int
) -> pd.DataFrame:
    """Compute per-C Pearson/Spearman correlation between n_sectors and q_c.

    Clusters with n_valid_pairs < n_min are excluded — their q_c is
    force-zeroed by prune.py's data-sparsity guard, not a real measurement,
    and would bias both coefficients.

    Args:
        diag:     Cluster diagnostics [C, cluster_id, q_c, n_sectors, n_valid_pairs].
        c_values: Sorted list of C values present in diag.
        n_min:    N_min threshold used by the originating run.

    Returns:
        DataFrame [C, pearson_r, spearman_rho, n_points_used]. Coefficients
        are NaN when fewer than 2 non-zeroed clusters exist at that C.
    """
    records = []
    for c in c_values:
        real = diag[(diag["C"] == c) & (diag["n_valid_pairs"] >= n_min)]
        n_used = len(real)
        pr = sr = float("nan")
        if n_used >= 2:
            x, y = real["n_sectors"].to_numpy(dtype=float), real["q_c"].to_numpy(dtype=float)
            if x.std() > 1e-12 and y.std() > 1e-12:
                pr = float(pearsonr(x, y)[0])
                sr = float(spearmanr(x, y).statistic)
        records.append({
            "C": c, "pearson_r": pr, "spearman_rho": sr,
            "n_points_used": n_used,
        })
    return pd.DataFrame(records)


def _ols_trend(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a degree-1 OLS line and return its two plottable endpoints.

    Args:
        x: Predictor values (n_sectors).
        y: Response values (q_c).

    Returns:
        (x_line, y_line): two-point arrays spanning [x.min(), x.max()],
        or empty arrays when fewer than 2 distinct x values exist.
    """
    if len(x) < 2 or np.ptp(x) < 1e-12:
        return np.array([]), np.array([])
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    return x_line, slope * x_line + intercept


def _corr_text(stats_by_c: pd.DataFrame, c: int) -> str:
    """Render the in-plot correlation annotation text for one C.

    Args:
        stats_by_c: _size_metric_correlations() output, indexed by C.
        c:          The C value to describe.

    Returns:
        e.g. "Pearson r=0.312, Spearman ρ=0.341" or a fallback message
        when too few non-zeroed clusters exist.
    """
    if c not in stats_by_c.index or pd.isna(stats_by_c.loc[c, "pearson_r"]):
        return "not enough non-zeroed clusters for correlation"
    r = stats_by_c.loc[c]
    return (
        f"Pearson r={r['pearson_r']:.3f}, "
        f"Spearman ρ={r['spearman_rho']:.3f}"
    )


def _size_x_range(c_diag: pd.DataFrame) -> list[float]:
    """Compute a padded n_sectors x-axis range for one C's clusters.

    Rows 3-4 rescale their x-axis every time C changes (small clusters at
    high C would otherwise be squeezed into a sliver of a fixed wide axis).

    Args:
        c_diag: Cluster diagnostics rows for a single C value.

    Returns:
        [x_min, x_max] with ~10% padding, floored at 0.
    """
    x_vals = c_diag["n_sectors"].to_numpy(dtype=float)
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    pad = max((x_max - x_min) * 0.1, 1.0)
    return [max(0.0, x_min - pad), x_max + pad]


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
    stats: pd.DataFrame,
    n_min: int,
    metric_label: str = "",
) -> go.Figure:
    """Build the four-panel analysis figure (Q trajectory + 3 C-synced views).

    Rows:
      1. Q-vs-C trajectory — static, always visible.
      2. Per-cluster eggs/dengue series for the selected C.
      3. Size (n_sectors) vs metric (q_c) scatter for the selected C, with
         an OLS trend line and the Pearson/Spearman coefficients in the
         title. Clusters with n_valid_pairs < n_min (guard-zeroed q_c) are
         drawn hollow/grey and excluded from both the trend line and the
         coefficients.
      4. Cluster-size rug for the selected C — one semi-transparent tick
         per cluster at its n_sectors value, same grey treatment for
         guard-zeroed clusters.

    Rows 2-4 are all driven by a single shared slider so they never show
    conflicting C values.

    Args:
        traj:         Q-trajectory DataFrame [C, Q].
        asgn:         Cluster assignments [C, sector_id, cluster_id].
        eggs:         Raw sector egg series.
        dengue:       Raw sector dengue series.
        diag:         Per-cluster diagnostics [C, cluster_id, q_c, n_sectors,
                      pop, n_valid_pairs].
        stats:        _size_metric_correlations() output.
        n_min:        N_min threshold used to classify guard-zeroed clusters.
        metric_label: Short string appended to the title, e.g.
                      'mst=egg_spearman | obj=corr_q_spearman'.

    Returns:
        A go.Figure with four subplots and one shared C slider.
    """
    c_values = sorted(asgn["C"].unique().tolist())
    stats_by_c = stats.set_index("C")

    # Identify epidemic biweeks for filtering the time series plot
    epic_bws = set(
        bw for bw in eggs["biweek"].unique()
        if any(bw.startswith(yr) for yr in EPIDEMIC_YEARS)
    )

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "Q-vs-C trajectory", "Per-cluster series",
            "Size vs Metric (q_c)", "Cluster size distribution",
        ],
        vertical_spacing=0.09,
        row_heights=[0.15, 0.28, 0.32, 0.17],
    )

    # ── Row 1: Q trajectory — static, no C dependence ─────────────────
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
    n_fixed = 1

    # ── q_c y-axis stays fixed on row 3 so correlation strength is
    # comparable across C; n_sectors x-axis (rows 3-4) rescales per C.
    q_c_min, q_c_max = float(diag["q_c"].min()), float(diag["q_c"].max())
    q_c_pad = (q_c_max - q_c_min) * 0.08 or 0.05

    # ── Build every C-dependent trace upfront, hidden by default ──────
    dyn_traces: list[tuple[int, go.Scatter | go.Histogram]] = []
    # vis_map[c] = list of dyn_traces indices belonging to that C
    vis_map: dict[int, list[int]] = {}
    x_range_map: dict[int, list[float]] = {}
    # Positions (within dyn_traces) of every C's size-histogram trace —
    # the bin-size dropdown restyles all of these together, regardless
    # of which C the shared slider currently shows.
    hist_positions: list[int] = []

    for c in c_values:
        start = len(dyn_traces)
        c_diag = diag[diag["C"] == c]
        x_range_map[c] = _size_x_range(c_diag)

        # -- Row 2: per-cluster eggs/dengue series (unchanged logic) ----
        series = _agg_cluster_series(asgn, eggs, dengue, c, diag)
        epic = series[series["biweek"].isin(epic_bws)].copy()

        for cid in sorted(epic["cluster_id"].unique()):
            sub = epic[epic["cluster_id"] == cid].sort_values("biweek")
            bk = int(sub["best_k"].iloc[0]) if not sub.empty else 0
            qc = float(sub["q_c"].iloc[0]) if not sub.empty else 0.0
            color = CLUSTER_COLORS[cid % 30]

            e_vals = sub["eggs"].values
            e_norm = (e_vals - np.nanmin(e_vals)) / (
                np.nanmax(e_vals) - np.nanmin(e_vals) + 1e-12
            )
            dyn_traces.append((2, go.Scatter(
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
            )))

            d_vals = sub["dengue"].values
            d_norm = (d_vals - np.nanmin(d_vals)) / (
                np.nanmax(d_vals) - np.nanmin(d_vals) + 1e-12
            )
            dyn_traces.append((2, go.Scatter(
                x=sub["biweek"], y=d_norm,
                mode="lines",
                line={"color": color, "dash": "dot"},
                name=f"C={c} cid={cid} dengue",
                visible=False,
                legendgroup=f"c{c}_cid{cid}",
                showlegend=False,
                hovertemplate=(
                    f"Cluster {cid} dengue (norm, lag={bk})<extra></extra>"
                ),
            )))

        # -- Row 3: size vs q_c scatter + OLS trend ----------------------
        is_real = c_diag["n_valid_pairs"] >= n_min
        real, zeroed = c_diag[is_real], c_diag[~is_real]

        dyn_traces.append((3, go.Scatter(
            x=real["n_sectors"], y=real["q_c"], mode="markers",
            marker={
                "color": [CLUSTER_COLORS[int(cid) % 30] for cid in real["cluster_id"]],
                "size": 9,
            },
            name=f"C={c} clusters", visible=False, showlegend=False,
            customdata=np.stack(
                [real["cluster_id"], real["pop"], real["n_valid_pairs"]], axis=-1
            ) if len(real) else None,
            hovertemplate=(
                "cluster %{customdata[0]:.0f}<br>"
                "n_sectors=%{x}<br>q_c=%{y:.3f}<br>"
                "pop=%{customdata[1]:.0f}<br>"
                "n_valid_pairs=%{customdata[2]:.0f}<extra></extra>"
            ),
        )))
        dyn_traces.append((3, go.Scatter(
            x=zeroed["n_sectors"], y=zeroed["q_c"], mode="markers",
            marker={
                "color": "rgba(0,0,0,0)",
                "line": {"color": "#999999", "width": 1.5},
                "size": 9,
            },
            name=f"C={c} guard-zeroed", visible=False, showlegend=False,
            customdata=np.stack(
                [zeroed["cluster_id"], zeroed["n_valid_pairs"]], axis=-1
            ) if len(zeroed) else None,
            hovertemplate=(
                "cluster %{customdata[0]:.0f}<br>"
                "n_sectors=%{x}<br>q_c=0 (zeroed — "
                "n_valid_pairs=%{customdata[1]:.0f} < N_min)<extra></extra>"
            ),
        )))
        x_line, y_line = _ols_trend(
            real["n_sectors"].to_numpy(dtype=float),
            real["q_c"].to_numpy(dtype=float),
        )
        dyn_traces.append((3, go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line={"color": "#444444", "dash": "dash", "width": 1.5},
            name=f"C={c} trend", visible=False, showlegend=False,
            hoverinfo="skip",
        )))

        # -- Row 4: cluster-size histogram + rug -------------------------
        # xbins.size=1 (default) puts every distinct n_sectors value in
        # its own bin; the bin-size dropdown below can widen this.
        dyn_traces.append((4, go.Histogram(
            x=c_diag["n_sectors"],
            xbins={"size": 1}, autobinx=False,
            marker={"color": "#8fb8de", "opacity": 0.6},
            name=f"C={c} size histogram", visible=False, showlegend=False,
            hovertemplate="n_sectors=%{x}<br>count=%{y}<extra></extra>",
        )))
        hist_positions.append(len(dyn_traces) - 1)
        dyn_traces.append((4, go.Scatter(
            x=real["n_sectors"], y=[0] * len(real), mode="markers",
            marker={
                "symbol": "line-ns", "size": 22, "opacity": 0.5,
                "color": [CLUSTER_COLORS[int(cid) % 30] for cid in real["cluster_id"]],
                "line": {"width": 3},
            },
            name=f"C={c} sizes", visible=False, showlegend=False,
            customdata=real["cluster_id"].to_numpy(),
            hovertemplate="cluster %{customdata:.0f}<br>n_sectors=%{x}<extra></extra>",
        )))
        dyn_traces.append((4, go.Scatter(
            x=zeroed["n_sectors"], y=[0] * len(zeroed), mode="markers",
            marker={
                "symbol": "line-ns", "size": 22, "opacity": 0.5,
                "color": "#999999", "line": {"width": 3},
            },
            name=f"C={c} sizes (zeroed)", visible=False, showlegend=False,
            customdata=zeroed["cluster_id"].to_numpy(),
            hovertemplate=(
                "cluster %{customdata:.0f}<br>"
                "n_sectors=%{x} (guard-zeroed)<extra></extra>"
            ),
        )))

        vis_map[c] = list(range(start, len(dyn_traces)))

    # Pre-reveal the first C value's traces, then add everything to fig
    c0 = c_values[0]
    for idx in vis_map[c0]:
        dyn_traces[idx][1].visible = True
    for row_idx, tr in dyn_traces:
        fig.add_trace(tr, row=row_idx, col=1)

    # ── In-plot correlation annotation, anchored inside row 3 ─────────
    # Index 4: make_subplots already created 4 subplot-title annotations
    # (indices 0-3) — this is appended right after them.
    fig.add_annotation(
        xref="x3 domain", yref="y3 domain",
        x=0.02, y=0.98, xanchor="left", yanchor="top",
        text=_corr_text(stats_by_c, c0),
        showarrow=False,
        font={"size": 12, "color": "#333333"},
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="#cccccc", borderwidth=1,
    )
    corr_annotation_idx = len(fig.layout.annotations) - 1

    # ── Bin-size / density dropdown for row 4's histogram ──────────────
    # Restyles every C's histogram trace at once (via absolute trace
    # indices) regardless of which one the C slider currently shows.
    # method="update" (3-arg form) lets each button also fix the y-axis
    # title, since count and density share the same underlying trace.
    hist_trace_indices = [n_fixed + p for p in hist_positions]
    bin_sizes = [1, 2, 3, 5, 10]
    bin_buttons = [
        {
            "label": "bin=1 (per size, default)" if b == 1 else f"bin={b}",
            "method": "update",
            "args": [
                {"xbins.size": b, "autobinx": False, "histnorm": ""},
                {"yaxis4.title.text": "count"},
                hist_trace_indices,
            ],
        }
        for b in bin_sizes
    ]
    bin_buttons.append({
        "label": "density",
        "method": "update",
        "args": [
            {"xbins.size": 1, "autobinx": False, "histnorm": "probability density"},
            {"yaxis4.title.text": "density"},
            hist_trace_indices,
        ],
    })

    # ── Shared slider: drives rows 2-4 together ────────────────────────
    n_dyn = len(dyn_traces)
    _suffix = f" | {metric_label}" if metric_label else ""
    steps = []
    for c in c_values:
        vis = [True] * n_fixed + [False] * n_dyn
        for idx in vis_map[c]:
            vis[n_fixed + idx] = True
        steps.append({
            "label": str(c),
            "method": "update",
            "args": [
                {"visible": vis},
                {
                    "title.text": f"SKATER Analysis — C={c}{_suffix}",
                    "xaxis3.range": x_range_map[c],
                    "xaxis4.range": x_range_map[c],
                    f"annotations[{corr_annotation_idx}].text": _corr_text(stats_by_c, c),
                },
            ],
        })

    fig.update_xaxes(range=x_range_map[c0], row=3, col=1, title_text="n_sectors")
    fig.update_yaxes(
        range=[q_c_min - q_c_pad, q_c_max + q_c_pad], row=3, col=1, title_text="q_c"
    )
    fig.update_xaxes(range=x_range_map[c0], row=4, col=1, title_text="n_sectors")
    fig.update_yaxes(row=4, col=1, title_text="count")

    # Label for the bin-size dropdown, anchored beside it
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.01, y=0.30, xanchor="left", yanchor="bottom",
        text="Histogram bin size:", showarrow=False,
        font={"size": 10, "color": "#333333"},
    )

    fig.update_layout(
        title=f"SKATER Analysis — C={c0}{_suffix}",
        height=1600,
        xaxis2_title="Biweek",
        yaxis2_title="Normalised value",
        xaxis_title="C (number of clusters)",
        yaxis_title="Q",
        legend={
            "x": 1.01, "y": 1.0,
            "xanchor": "left", "yanchor": "top",
            "font": {"size": 9},
        },
        updatemenus=[{
            "type": "dropdown",
            "direction": "down",
            "showactive": True,
            "buttons": bin_buttons,
            "x": 1.01, "xanchor": "left",
            "y": 0.28, "yanchor": "top",
        }],
        sliders=[{
            "active": 0,
            "steps": steps,
            "x": 0.05,
            "len": 0.9,
            "y": -0.04,
            "yanchor": "top",
            "currentvalue": {
                "prefix": "C = ",
                "visible": True,
                "xanchor": "center",
            },
        }],
        margin={"t": 80, "b": 80, "l": 60, "r": 100},
    )
    return fig


# ── Stop condition banner ─────────────────────────────────────────────

_BANNER_BG: dict[str, str] = {
    "c_max": "#d0e8ff",
    "no_valid_cut_geometry": "#fff3cd",
    "no_valid_cut_local_degradation": "#fff3cd",
    "global_degradation": "#f8d7da",
}
_BANNER_BORDER: dict[str, str] = {
    "c_max": "#5b9bd5",
    "no_valid_cut_geometry": "#e6a817",
    "no_valid_cut_local_degradation": "#e6a817",
    "global_degradation": "#c0392b",
}


def _load_stop_info(results_dir: Path) -> dict | None:
    """Load stop_info.json from a run directory, or None if absent.

    Returns None for old runs that predate stop-condition tracking so
    the dashboard degrades gracefully without raising an error.

    Args:
        results_dir: Run-specific output directory.

    Returns:
        Parsed dict or None.
    """
    p = results_dir / "stop_info.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def _stop_banner_html(stop_info: dict) -> str:
    """Render a color-coded HTML banner describing why the run stopped.

    Args:
        stop_info: Dict loaded from stop_info.json with keys
                   reason, C_final, best_dQ, threshold.

    Returns:
        An HTML <div> string ready to embed at the top of a dashboard page.
    """
    reason = stop_info.get("reason", "unknown")
    c_final = stop_info.get("C_final", "?")
    best_dq = stop_info.get("best_dQ")
    threshold = stop_info.get("threshold")

    if reason == "c_max":
        msg = f"Run completed normally — C_max={c_final} reached."
    elif reason == "no_valid_cut_geometry":
        msg = (
            f"Stopped early at C={c_final} — "
            "S_min / N_min constraints exhausted all candidate cuts."
        )
    elif reason == "no_valid_cut_local_degradation":
        msg = (
            f"Stopped early at C={c_final} — "
            "local degradation guard rejected every remaining cut."
        )
    elif reason == "global_degradation" and best_dq is not None:
        thr_str = f"{threshold:.4f}" if threshold is not None else "?"
        msg = (
            f"Stopped early at C={c_final} — "
            f"best ΔQ={best_dq:.4f} below threshold={thr_str}."
        )
    else:
        msg = f"Stopped at C={c_final} — reason: {reason}."

    bg = _BANNER_BG.get(reason, "#f0f0f0")
    border = _BANNER_BORDER.get(reason, "#999")
    style = (
        f"background:{bg};border-left:5px solid {border};"
        "padding:12px 18px;margin-bottom:18px;"
        "font-family:sans-serif;font-size:14px;"
        "border-radius:3px;line-height:1.5"
    )
    return f"<div style='{style}'><b>Stop condition:</b> {msg}</div>"


def _write_dashboard_html(
    fig: go.Figure,
    out: Path,
    title: str,
    banner_html: str,
) -> None:
    """Assemble a standalone HTML page with an optional banner above the figure.

    Args:
        fig:         Plotly figure to embed.
        out:         Output file path.
        title:       <title> tag text.
        banner_html: HTML string prepended above the figure (empty → no banner).
    """
    fig_html = fig.to_html(full_html=False, include_plotlyjs=True)
    page = "\n".join([
        "<!DOCTYPE html><html>",
        "<head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:sans-serif;padding:16px;"
        "max-width:1600px;margin:auto}</style>",
        "</head><body>",
        banner_html,
        fig_html,
        "</body></html>",
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Load all data, build both figures, write HTML files."""
    # ── Read metric config and run_label for path resolution ──────────
    with open("params.yaml") as fh:
        _params = yaml.safe_load(fh).get("skater", {})
    run_label = _params.get("run_label", "default")
    results_dir = RESULTS_BASE / run_label
    metric_label = (
        f"mst={_params.get('mst_cost', '?')} | "
        f"obj={_params.get('prune_obj', '?')}"
    )
    logger.info("Metric config: %s | run=%s", metric_label, run_label)

    # ── Fall back to first valid run if configured label is missing ───
    def _is_valid_run(d: Path) -> bool:
        return (
            d.is_dir()
            and (d / "cluster_assignments.csv").exists()
            and (d / "q_trajectory.csv").exists()
            and (d / "cluster_diagnostics.csv").exists()
        )

    if not _is_valid_run(results_dir):
        valid_runs = sorted(
            d for d in RESULTS_BASE.iterdir() if _is_valid_run(d)
        )
        if not valid_runs:
            logger.error(
                "No valid run found under %s. Run `dvc repro skater` first.",
                RESULTS_BASE,
            )
            raise SystemExit(1)
        fallback = valid_runs[0]
        logger.warning(
            "Run '%s' not found — falling back to '%s'.",
            run_label, fallback.name,
        )
        results_dir = fallback
        run_label = fallback.name

    logger.info("Loading SKATER results…")
    asgn, traj, diag = _load_results(results_dir)
    geojson = _load_geojson()
    ovitrap_locs = _load_ovitrap_locations()
    eggs, dengue = _load_sector_series()

    # ── Resolve N_min for guard-zeroed classification ─────────────────
    # Prefer the run's own archived config so backfilled old runs with a
    # different N_min aren't misclassified against today's params.yaml.
    run_params_path = results_dir / "run_params.json"
    if run_params_path.exists():
        with open(run_params_path) as fh:
            _run_params = json.load(fh)
        n_min = _run_params.get("N_min", _params.get("N_min", 20))
    else:
        n_min = _params.get("N_min", 20)

    c_values = sorted(diag["C"].unique().tolist())
    stats = _size_metric_correlations(diag, c_values, n_min)
    stats.to_csv(results_dir / "size_metric_correlation.csv", index=False)
    logger.info("Saved size_metric_correlation.csv (N_min=%d)", n_min)

    stop_info = _load_stop_info(results_dir)
    if stop_info:
        logger.info(
            "Stop condition: %s (C_final=%d)",
            stop_info["reason"], stop_info["C_final"],
        )
    else:
        logger.info("No stop_info.json found — banner suppressed.")
    banner = _stop_banner_html(stop_info) if stop_info else ""

    logger.info("Building map dashboard…")
    fig_map = build_combined_figure(
        asgn, geojson, diag, ovitrap_locs, metric_label=metric_label
    )
    out_map = results_dir / "dashboard_map.html"
    _write_dashboard_html(fig_map, out_map, "SKATER Map", banner)
    logger.info("Saved %s", out_map)

    logger.info("Building analysis dashboard…")
    fig_ana = build_analysis_figure(
        traj, asgn, eggs, dengue, diag, stats, n_min, metric_label=metric_label
    )
    out_ana = results_dir / "dashboard_analysis.html"
    _write_dashboard_html(fig_ana, out_ana, "SKATER Analysis", banner)
    logger.info("Saved %s", out_ana)


if __name__ == "__main__":
    main()
