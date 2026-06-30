"""Cluster concordance analysis across SKATER runs — ARI and NMI.

Discovers all completed runs in results/skater/*/ and computes pairwise
Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) for
every shared C value across every pair of runs.

A valid run directory must contain:
  - cluster_assignments.csv  (written by src/skater/run.py)
  - run_params.json          (written by src/skater/run.py)

Output:
  results/skater/concordance_dashboard.html — single interactive HTML.

Run manually (NOT a DVC stage):
  python scripts/skater_concordance_dashboard.py

Inputs:
  results/skater/*/cluster_assignments.csv — per-run sector assignments.
  results/skater/*/run_params.json         — per-run config snapshot.
Outputs:
  results/skater/concordance_dashboard.html — ARI/NMI line charts +
    pairwise heatmap with C slider + run params table.
"""
from __future__ import annotations

import json
import logging
import math
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE = Path("results/skater")
OUT_PATH = RESULTS_BASE / "concordance_dashboard.html"

# Colour palette for run pairs (cycles if more than 10 pairs)
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Params shown in the run configuration table
KEY_PARAMS = [
    "mst_cost", "prune_obj", "C_max",
    "stop_local_degradation", "global_degradation_threshold",
    "N_min", "S_min", "k_min", "k_max",
]


# ── Run discovery ─────────────────────────────────────────────────────


def discover_runs(base: Path) -> list[dict]:
    """Find all valid SKATER run directories under base.

    A valid run must contain cluster_assignments.csv and run_params.json.
    Directories that exist but are missing either file are skipped with
    a warning so partially-completed runs don't crash the analysis.

    Args:
        base: Root directory to search (results/skater/).

    Returns:
        List of run dicts — keys: label, path, params, assignments.
        Sorted by label for deterministic ordering.
    """
    runs = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        asgn_path = subdir / "cluster_assignments.csv"
        params_path = subdir / "run_params.json"
        if not asgn_path.exists() or not params_path.exists():
            logger.warning(
                "Skipping '%s' — missing cluster_assignments.csv or "
                "run_params.json",
                subdir.name,
            )
            continue
        with open(params_path) as fh:
            params = json.load(fh)
        asgn = pd.read_csv(asgn_path, dtype={"sector_id": str})
        runs.append({
            "label": subdir.name,
            "path": subdir,
            "params": params,
            "assignments": asgn,
        })
        logger.info(
            "Found run '%s' — %d C values, %d sectors",
            subdir.name,
            asgn["C"].nunique(),
            asgn[asgn["C"] == asgn["C"].min()]["sector_id"].nunique(),
        )
    return runs


# ── Concordance computation ───────────────────────────────────────────


def _align_labels(
    asgn_a: pd.DataFrame,
    asgn_b: pd.DataFrame,
    c: int,
) -> tuple[list[int], list[int], int]:
    """Extract aligned label vectors for a given C on the sector intersection.

    ARI and NMI are valid on any common subset, so sector sets that differ
    between runs are handled by restricting to the intersection. A warning
    is emitted by the caller when the intersection is smaller than either set.

    Args:
        asgn_a: Assignments from run A [C, sector_id, cluster_id].
        asgn_b: Assignments from run B [C, sector_id, cluster_id].
        c:      Number of clusters to compare.

    Returns:
        (labels_a, labels_b, n_common): aligned integer label lists and
        the count of common sectors used.
    """
    a = asgn_a[asgn_a["C"] == c].set_index("sector_id")["cluster_id"]
    b = asgn_b[asgn_b["C"] == c].set_index("sector_id")["cluster_id"]
    common = a.index.intersection(b.index)
    return a[common].tolist(), b[common].tolist(), len(common)


def compute_concordance(runs: list[dict]) -> pd.DataFrame:
    """Compute pairwise ARI and NMI for all run pairs at all shared C values.

    Steps:
      1. Iterate over all unique (run_a, run_b) pairs.
      2. Find C values present in both runs.
      3. For each shared C: align sector labels on their intersection,
         then call sklearn's ARI and NMI (arithmetic normalisation).
      4. Warn once per pair when sector sets differ.

    Args:
        runs: List of run dicts from discover_runs().

    Returns:
        DataFrame [run_a, run_b, pair, C, ARI, NMI, n_common_sectors].
        Empty DataFrame (with correct columns) if fewer than 2 runs.
    """
    _cols = ["run_a", "run_b", "pair", "C", "ARI", "NMI", "n_common_sectors"]
    if len(runs) < 2:
        logger.warning("Fewer than 2 runs found — nothing to compare.")
        return pd.DataFrame(columns=_cols)

    records = []
    for run_a, run_b in combinations(runs, 2):
        la, lb = run_a["label"], run_b["label"]
        pair = f"{la} vs {lb}"

        c_shared = sorted(
            set(run_a["assignments"]["C"].unique())
            & set(run_b["assignments"]["C"].unique())
        )
        if not c_shared:
            logger.warning("No shared C values for pair '%s'", pair)
            continue

        warned_mismatch = False
        for c in c_shared:
            vec_a, vec_b, n_common = _align_labels(
                run_a["assignments"], run_b["assignments"], c
            )
            if n_common == 0:
                continue

            # Warn once per pair when sector sets differ
            n_a = run_a["assignments"][
                run_a["assignments"]["C"] == c
            ]["sector_id"].nunique()
            if not warned_mismatch and n_common < n_a:
                logger.warning(
                    "Pair '%s': sector sets differ — using %d "
                    "common sectors (run A has %d)",
                    pair, n_common, n_a,
                )
                warned_mismatch = True

            ari = adjusted_rand_score(vec_a, vec_b)
            nmi = normalized_mutual_info_score(
                vec_a, vec_b, average_method="arithmetic"
            )
            records.append({
                "run_a": la, "run_b": lb, "pair": pair,
                "C": c,
                "ARI": round(ari, 6),
                "NMI": round(nmi, 6),
                "n_common_sectors": n_common,
            })

        logger.info("Pair '%s': %d C values computed", pair, len(c_shared))

    return pd.DataFrame(records) if records else pd.DataFrame(columns=_cols)


# ── Figure builders ───────────────────────────────────────────────────


def build_line_figure(concordance: pd.DataFrame) -> go.Figure:
    """Build ARI and NMI vs C line charts (one line per run pair).

    Two vertically stacked subplots share the x-axis.  Each pair gets
    one solid line (ARI) and one dotted line (NMI) in the same colour.
    Plotly's native legend allows the user to click pairs to hide/show.

    Args:
        concordance: Output of compute_concordance().

    Returns:
        go.Figure with rows=[ARI, NMI], shared x-axis.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["ARI vs C", "NMI vs C"],
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    for i, pair in enumerate(concordance["pair"].unique()):
        sub = concordance[concordance["pair"] == pair].sort_values("C")
        color = PALETTE[i % len(PALETTE)]

        # ── ARI — solid line ──────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=sub["C"], y=sub["ARI"],
                mode="lines+markers",
                name=pair,
                line={"color": color},
                marker={"size": 5},
                legendgroup=pair,
                hovertemplate=(
                    f"<b>{pair}</b><br>"
                    "C=%{x}<br>ARI=%{y:.4f}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        # ── NMI — dotted line, same colour, linked legend group ───────
        fig.add_trace(
            go.Scatter(
                x=sub["C"], y=sub["NMI"],
                mode="lines+markers",
                name=pair,
                line={"color": color, "dash": "dot"},
                marker={"size": 5},
                legendgroup=pair,
                showlegend=False,
                hovertemplate=(
                    f"<b>{pair}</b><br>"
                    "C=%{x}<br>NMI=%{y:.4f}"
                    "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
    fig.update_layout(
        title="Cluster concordance — ARI (solid) and NMI (dotted) vs C",
        height=700,
        margin={"t": 80, "b": 60, "l": 60, "r": 20},
        xaxis2_title="C (number of clusters)",
        yaxis_title="ARI",
        yaxis2_title="NMI",
        legend={"title": "Pair (click to hide)"},
    )
    return fig


def build_heatmap_figure(
    concordance: pd.DataFrame,
    runs: list[dict],
) -> go.Figure:
    """Build a symmetric pairwise ARI heatmap with a C slider.

    One heatmap trace per C value — the slider toggles visibility.
    Diagonal cells are fixed at 1.0 (a run compared to itself).
    Missing pairs (no shared C) appear as NaN (grey).

    Args:
        concordance: Output of compute_concordance().
        runs:        List of run dicts (defines axis label order).

    Returns:
        go.Figure with one Heatmap trace per C value and a slider.
    """
    c_values = sorted(concordance["C"].unique())
    labels = [r["label"] for r in runs]
    n = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}

    traces: list[go.Heatmap] = []
    for c in c_values:
        sub = concordance[concordance["C"] == c]
        mat = [
            [1.0 if i == j else float("nan") for j in range(n)]
            for i in range(n)
        ]
        for _, row in sub.iterrows():
            i, j = idx[row["run_a"]], idx[row["run_b"]]
            mat[i][j] = mat[j][i] = float(row["ARI"])

        text = [
            [f"{v:.3f}" if not math.isnan(v) else "" for v in row]
            for row in mat
        ]
        traces.append(go.Heatmap(
            z=mat,
            x=labels, y=labels,
            colorscale="RdYlGn",
            zmin=0.0, zmax=1.0,
            visible=False,
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>ARI=%{z:.4f}<extra></extra>",
            showscale=True,
            colorbar={"title": "ARI"},
        ))

    if traces:
        traces[0].visible = True

    steps = [
        {
            "label": str(c),
            "method": "update",
            "args": [
                {"visible": [k == i for k in range(len(c_values))]},
                {"title.text": f"Pairwise ARI — C={c}"},
            ],
        }
        for i, c in enumerate(c_values)
    ]
    c0 = c_values[0] if c_values else "?"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Pairwise ARI — C={c0}",
        height=max(400, 280 + 40 * n),
        margin={"t": 120, "b": 80, "l": 120, "r": 20},
        sliders=[{
            "active": 0,
            "steps": steps,
            "x": 0.1, "len": 0.8,
            "y": -0.05, "yanchor": "top",
            "currentvalue": {
                "prefix": "C = ",
                "visible": True,
                "xanchor": "center",
            },
        }],
    )
    return fig


# ── Run params table ──────────────────────────────────────────────────


def _params_table_html(runs: list[dict]) -> str:
    """Render a plain HTML table of key params for each run.

    Args:
        runs: List of run dicts from discover_runs().

    Returns:
        HTML <table> string for embedding in the dashboard page.
    """
    th = "border:1px solid #ccc;padding:4px 8px;background:#f0f0f0"
    td = "border:1px solid #ccc;padding:4px 8px"
    header = (
        f"<tr><th style='{th}'>run</th>"
        + "".join(f"<th style='{th}'>{k}</th>" for k in KEY_PARAMS)
        + "</tr>"
    )
    body = "".join(
        f"<tr><td style='{td}'><b>{r['label']}</b></td>"
        + "".join(
            f"<td style='{td}'>{r['params'].get(k, '—')}</td>"
            for k in KEY_PARAMS
        )
        + "</tr>"
        for r in runs
    )
    style = (
        "border-collapse:collapse;font-family:monospace;"
        "font-size:12px;width:100%;margin-bottom:16px"
    )
    return (
        f"<table style='{style}'>"
        f"<thead>{header}</thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Discover runs, compute concordance, write concordance_dashboard.html."""
    logger.info("Discovering runs in %s …", RESULTS_BASE)
    runs = discover_runs(RESULTS_BASE)
    if len(runs) < 2:
        logger.error(
            "Need ≥2 completed runs. "
            "Set run_label in params.yaml and run dvc repro for each config."
        )
        return

    logger.info("Computing pairwise ARI/NMI for %d runs …", len(runs))
    concordance = compute_concordance(runs)
    if concordance.empty:
        logger.error("Concordance table empty — check run outputs.")
        return

    # ── Save raw concordance table alongside the HTML ─────────────────
    concordance.to_csv(
        RESULTS_BASE / "concordance_table.csv", index=False
    )
    logger.info("Saved concordance_table.csv")

    logger.info("Building figures …")
    fig_lines = build_line_figure(concordance)
    fig_heatmap = build_heatmap_figure(concordance, runs)

    # ── Assemble single-page HTML ─────────────────────────────────────
    # include_plotlyjs="cdn" on first figure; False on second to avoid
    # embedding the library twice (CDN link is one small <script> tag).
    lines_html = fig_lines.to_html(
        full_html=False, include_plotlyjs="cdn"
    )
    heatmap_html = fig_heatmap.to_html(
        full_html=False, include_plotlyjs=False
    )
    table_html = _params_table_html(runs)

    page = "\n".join([
        "<!DOCTYPE html><html>",
        "<head><meta charset='utf-8'>",
        "<title>SKATER Concordance</title>",
        "<style>body{font-family:sans-serif;padding:20px;"
        "max-width:1400px;margin:auto}</style>",
        "</head><body>",
        "<h1>SKATER cluster concordance</h1>",
        "<h2>Run configurations</h2>",
        table_html,
        "<h2>ARI and NMI vs C</h2>",
        lines_html,
        "<h2>Pairwise ARI heatmap</h2>",
        heatmap_html,
        "</body></html>",
    ])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(page, encoding="utf-8")
    logger.info("Saved %s", OUT_PATH)


if __name__ == "__main__":
    main()
