"""Synthetic SKATER dashboard — visual validation of 4-cluster recovery.

Runs the SKATER pipeline on a 4×4 unit-square grid with four planted
quadrant groups, then renders an interactive HTML dashboard with:
  1. Grid map: cells coloured by cluster assignment, slider for C=1..4.
  2. Q-vs-C trajectory: global objective per cut step.
  3. Time series: normalised eggs (solid) and dengue (dotted) per
     cluster, with a dropdown to select C.

Outputs:
  results/skater/dashboard_synthetic.html
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skater.adjacency import build_adjacency
from src.skater.config import SkaterConfig
from src.skater.mst import build_mst
from src.skater.prune import Snapshot, greedy_prune

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
OUT_PATH = Path("results/skater/dashboard_synthetic.html")
ROWS, COLS = 4, 4
N_BW = 50
LAG = 3
SEED = 0
CLUSTER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


# ── Synthetic data ─────────────────────────────────────────────────────

def _rect(x0: float, y0: float, x1: float, y1: float, sid: str) -> dict:
    """GeoJSON Feature for a unit-square cell."""
    coords = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    return {
        "type": "Feature",
        "properties": {"CD_SETOR": sid},
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def _grid_geojson(rows: int, cols: int) -> dict:
    """Build a rows×cols unit-square grid GeoJSON FeatureCollection.

    Args:
        rows: Number of grid rows.
        cols: Number of grid columns.

    Returns:
        GeoJSON FeatureCollection dict with sector IDs 's{r}{c}'.
    """
    feats = [
        _rect(c, r, c + 1, r + 1, f"s{r}{c}")
        for r in range(rows)
        for c in range(cols)
    ]
    return {"type": "FeatureCollection", "features": feats}


def _planted_data(
    sector_list: list[str],
    groups: list[list[str]],
    n_bw: int = N_BW,
    lag: int = LAG,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate planted eggs/dengue signals for N sector groups.

    Each group receives a distinct random signal; dengue lags eggs by
    `lag` biweeks with σ=0.02 noise for near-perfect correlation.

    Args:
        sector_list: All sector IDs ordered to match matrix row indices.
        groups:      One list of sector IDs per planted group.
        n_bw:        Number of biweek time steps.
        lag:         Biweek lag from eggs to dengue.
        seed:        RNG seed.

    Returns:
        Tuple (eggs, dengue, pop, year_ids) aligned to sector_list.
    """
    rng = np.random.default_rng(seed)
    n = len(sector_list)
    idx = {s: i for i, s in enumerate(sector_list)}
    year_ids = np.array(["epic"] * n_bw)
    eggs = np.zeros((n, n_bw))
    dengue = np.zeros((n, n_bw))
    pop = np.ones(n)
    for group in groups:
        sig = rng.standard_normal(n_bw)
        for s in group:
            i = idx[s]
            eggs[i] = sig + rng.standard_normal(n_bw) * 0.02
            dengue[i, lag:] = (
                sig[:-lag] + rng.standard_normal(n_bw - lag) * 0.02
            )
    return eggs, dengue, pop, year_ids


# ── Pipeline ──────────────────────────────────────────────────────────

def run_pipeline() -> tuple[
    list[str],
    list[list[str]],
    list[Snapshot],
    np.ndarray,
    np.ndarray,
]:
    """Build synthetic data and run the full SKATER pipeline.

    Returns:
        (sector_list, groups, snapshots, eggs, dengue)
    """
    sector_list = [f"s{r}{c}" for r in range(ROWS) for c in range(COLS)]
    groups = [
        [f"s{r}{c}" for r in range(2) for c in range(2)],    # BL
        [f"s{r}{c}" for r in range(2) for c in range(2, 4)], # BR
        [f"s{r}{c}" for r in range(2, 4) for c in range(2)], # TL
        [f"s{r}{c}" for r in range(2, 4) for c in range(2, 4)],  # TR
    ]
    cfg = SkaterConfig(
        k_min=2, k_max=5, C_max=4,
        N_min=3, S_min=1,
        mst_min_overlap=5,
        epidemic_years=["epic"],
    )
    gj = _grid_geojson(ROWS, COLS)
    eggs, dengue, pop, year_ids = _planted_data(sector_list, groups)
    adjacency = build_adjacency(gj)
    mst = build_mst(adjacency, eggs, sector_list, cfg)
    sector_idx = {s: i for i, s in enumerate(sector_list)}
    snaps, _stop_info = greedy_prune(
        mst, sector_idx, eggs, dengue, pop, year_ids, cfg
    )
    logger.info("Pipeline done: %d snapshots, final C=%d", len(snaps), snaps[-1].C)
    return sector_list, groups, snaps, eggs, dengue


# ── Helpers ───────────────────────────────────────────────────────────

def _assignment_matrix(snap: Snapshot) -> np.ndarray:
    """Convert sector assignments to (ROWS, COLS) cluster-ID matrix.

    Args:
        snap: Pipeline snapshot at a given C.

    Returns:
        Integer array of shape (ROWS, COLS) with cluster IDs.
    """
    mat = np.full((ROWS, COLS), -1, dtype=int)
    for r in range(ROWS):
        for c in range(COLS):
            mat[r, c] = snap.assignments.get(f"s{r}{c}", -1)
    return mat


def _discrete_colorscale(n: int) -> list[list]:
    """Build a plotly discrete colorscale mapping integers 0..(n-1).

    Args:
        n: Number of discrete values to colour.

    Returns:
        Plotly colorscale list (pairs of [fraction, color]).
    """
    step = 1.0 / n
    scale: list[list] = []
    for i in range(n):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        scale.extend([[i * step, color], [(i + 1) * step, color]])
    return scale


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalise x to [0, 1]; return zeros if range is zero."""
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


# ── Figure builders ───────────────────────────────────────────────────

def build_grid_figure(snaps: list[Snapshot]) -> go.Figure:
    """Interactive grid map coloured by cluster with a C slider.

    Args:
        snaps: Pipeline snapshots (one per C value).

    Returns:
        Plotly Figure with a slider control.
    """
    traces: list[go.Heatmap] = []
    for snap in snaps:
        # Flip rows so row 0 is at the bottom of the chart
        z = _assignment_matrix(snap)[::-1, :].astype(float)
        text = [
            [f"s{ROWS-1-r}{c} → cluster {int(z[r,c])}"
             for c in range(COLS)]
            for r in range(ROWS)
        ]
        traces.append(go.Heatmap(
            z=z,
            text=text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=_discrete_colorscale(4),
            zmin=0, zmax=3,
            showscale=False,
            visible=False,
            xgap=2, ygap=2,
        ))
    traces[0].visible = True

    steps = [
        dict(
            method="update",
            args=[{"visible": [j == i for j in range(len(snaps))]}],
            label=f"C={s.C}  Q={s.Q:.3f}",
        )
        for i, s in enumerate(snaps)
    ]
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Synthetic 4×4 Grid — Cluster Assignments",
        sliders=[dict(
            active=0, steps=steps,
            currentvalue=dict(prefix="", font=dict(size=14)),
            pad=dict(t=60),
        )],
        xaxis=dict(title="Column", tickvals=list(range(COLS)),
                   ticktext=[str(c) for c in range(COLS)]),
        yaxis=dict(title="Row", tickvals=list(range(ROWS)),
                   ticktext=[str(ROWS - 1 - r) for r in range(ROWS)]),
        height=500, width=520,
        margin=dict(t=100),
    )
    return fig


def build_qc_figure(snaps: list[Snapshot]) -> go.Figure:
    """Q-vs-C line chart.

    Args:
        snaps: Pipeline snapshots.

    Returns:
        Plotly Figure.
    """
    c_vals = [s.C for s in snaps]
    q_vals = [s.Q for s in snaps]
    fig = go.Figure(go.Scatter(
        x=c_vals, y=q_vals,
        mode="lines+markers",
        marker=dict(size=10, color="#377eb8"),
        line=dict(width=2),
        hovertemplate="C=%{x}<br>Q=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Global Objective Q vs Number of Clusters",
        xaxis=dict(title="C", tickvals=c_vals),
        yaxis=dict(title="Q"),
        height=380,
    )
    return fig


def build_timeseries_figure(
    snaps: list[Snapshot],
    sector_list: list[str],
    eggs: np.ndarray,
    dengue: np.ndarray,
) -> go.Figure:
    """Per-cluster normalised eggs/dengue time series with C dropdown.

    Args:
        snaps:       Pipeline snapshots.
        sector_list: Sector IDs aligned to matrix rows.
        eggs:        (n_sectors, n_bw) egg count matrix.
        dengue:      (n_sectors, n_bw) dengue rate matrix.

    Returns:
        Plotly Figure with dropdown to select C.
    """
    sector_idx = {s: i for i, s in enumerate(sector_list)}
    biweeks = list(range(N_BW))

    # Build all traces; track which belong to which snapshot
    all_traces: list[go.Scatter] = []
    traces_per_snap: list[int] = []

    for snap in snaps:
        count = 0
        for ci, cluster in enumerate(snap.clusters):
            idxs = [sector_idx[s] for s in cluster.sectors]
            color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            eggs_mean = _minmax(eggs[idxs, :].mean(axis=0))
            dng_mean = _minmax(dengue[idxs, :].mean(axis=0))
            all_traces.append(go.Scatter(
                x=biweeks, y=eggs_mean,
                name=f"C={snap.C} cl{ci} eggs",
                mode="lines",
                line=dict(color=color, width=2, dash="solid"),
                visible=snap.C == snaps[0].C,
            ))
            all_traces.append(go.Scatter(
                x=biweeks, y=dng_mean,
                name=f"C={snap.C} cl{ci} dengue",
                mode="lines",
                line=dict(color=color, width=2, dash="dot"),
                visible=snap.C == snaps[0].C,
            ))
            count += 2
        traces_per_snap.append(count)

    # Build visibility vector per dropdown selection
    def _vis(snap_idx: int) -> list[bool]:
        vis = [False] * len(all_traces)
        offset = sum(traces_per_snap[:snap_idx])
        for k in range(traces_per_snap[snap_idx]):
            vis[offset + k] = True
        return vis

    buttons = [
        dict(method="update", args=[{"visible": _vis(i)}],
             label=f"C={s.C}")
        for i, s in enumerate(snaps)
    ]
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title="Per-cluster Normalised Eggs (solid) & Dengue (dotted)",
        updatemenus=[dict(
            type="dropdown", buttons=buttons, active=0,
            x=0.0, y=1.18, xanchor="left",
        )],
        xaxis=dict(title="Biweek"),
        yaxis=dict(title="Normalised value [0–1]"),
        height=460,
        legend=dict(orientation="v", x=1.02),
        margin=dict(t=100),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    """Run pipeline, build all figures, write single HTML file."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sector_list, groups, snaps, eggs, dengue = run_pipeline()

    fig_grid = build_grid_figure(snaps)
    fig_qc = build_qc_figure(snaps)
    fig_ts = build_timeseries_figure(snaps, sector_list, eggs, dengue)

    # ── Write combined HTML ───────────────────────────────────────────
    html = "\n".join([
        "<html><head><meta charset='utf-8'>",
        "<title>SKATER Synthetic Dashboard</title></head>",
        "<body style='font-family:sans-serif;max-width:900px;margin:auto'>",
        "<h2>SKATER Synthetic Validation — 4×4 Grid, 4 Planted Quadrants</h2>",
        fig_grid.to_html(full_html=False, include_plotlyjs="cdn"),
        fig_qc.to_html(full_html=False, include_plotlyjs=False),
        fig_ts.to_html(full_html=False, include_plotlyjs=False),
        "</body></html>",
    ])
    OUT_PATH.write_text(html, encoding="utf-8")
    logger.info("Dashboard → %s", OUT_PATH)


if __name__ == "__main__":
    main()
