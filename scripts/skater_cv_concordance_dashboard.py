"""Cluster concordance across SKATER-CV folds — ARI and NMI.

Each leave-one-epidemic-year-out fold re-runs SKATER with that year's
data excluded from both the MST cost and the pruning objective, so the
learned regionalization can differ from fold to fold. This computes
pairwise Adjusted Rand Index (ARI) and Normalized Mutual Information
(NMI) between every pair of folds at every C they share, quantifying
how much the regionalization depends on which epidemic year was held
out — the CV-fold analogue of scripts/skater_concordance_dashboard.py
(which instead compares different all-years run_labels).

Both the source partitions and this script's own output are namespaced
by params.yaml[skater].run_label, same convention as the rest of the
skater_cv pipeline. Output is written alongside the partitions
themselves (not under skater_cv/) so it stays owned by the
`skater_cv_partitions` DVC stage's own output directory — this
analysis only depends on the partitions, never the MLP models.

Run:  python scripts/skater_cv_concordance_dashboard.py

Inputs:
  results/multiscale/partitions/<run_label>/fold_<year>/
    cluster_assignments.csv, fold_meta.json
Outputs:
  results/multiscale/partitions/<run_label>/concordance_dashboard.html
  results/multiscale/partitions/<run_label>/concordance_table.csv
"""
from __future__ import annotations

import json
import logging
import math
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PART_BASE = Path("results/multiscale/partitions")

# Colour palette for fold pairs (cycles if more than 10 pairs)
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _load_run_label() -> str:
    """Read params.yaml[skater].run_label — same source as the rest of
    the skater_cv pipeline, so this script always targets the currently
    configured run.
    """
    with open("params.yaml") as fh:
        return yaml.safe_load(fh)["skater"]["run_label"]


# ── Fold discovery ────────────────────────────────────────────────────

def discover_folds(part_dir: Path) -> list[dict]:
    """Find all completed CV fold directories under part_dir.

    A valid fold directory must contain cluster_assignments.csv and
    fold_meta.json. Directories missing either are skipped with a
    warning so a partially-completed sweep doesn't crash the analysis.

    Args:
        part_dir: results/multiscale/partitions/<run_label>/.

    Returns:
        List of fold dicts — keys: label (test_year), path, meta,
        assignments. Sorted by label for deterministic ordering.
    """
    folds = []
    for subdir in sorted(part_dir.glob("fold_*")):
        if not subdir.is_dir():
            continue
        asgn_path = subdir / "cluster_assignments.csv"
        meta_path = subdir / "fold_meta.json"
        if not asgn_path.exists() or not meta_path.exists():
            logger.warning(
                "Skipping '%s' — missing cluster_assignments.csv or "
                "fold_meta.json",
                subdir.name,
            )
            continue
        meta = json.loads(meta_path.read_text())
        asgn = pd.read_csv(asgn_path, dtype={"sector_id": str})
        folds.append({
            "label": meta["test_year"],
            "path": subdir,
            "meta": meta,
            "assignments": asgn,
        })
        logger.info(
            "Found fold '%s' — C=1..%d, %d sectors",
            meta["test_year"],
            asgn["C"].max(),
            asgn[asgn["C"] == asgn["C"].min()]["sector_id"].nunique(),
        )
    return folds


# ── Concordance computation ───────────────────────────────────────────

def _align_labels(
    asgn_a: pd.DataFrame, asgn_b: pd.DataFrame, c: int,
) -> tuple[list[int], list[int], int]:
    """Extract aligned label vectors for a given C on the sector intersection.

    Args:
        asgn_a: Assignments from fold A [C, sector_id, cluster_id].
        asgn_b: Assignments from fold B [C, sector_id, cluster_id].
        c:      Number of clusters to compare.

    Returns:
        (labels_a, labels_b, n_common): aligned integer label lists and
        the count of common sectors used.
    """
    a = asgn_a[asgn_a["C"] == c].set_index("sector_id")["cluster_id"]
    b = asgn_b[asgn_b["C"] == c].set_index("sector_id")["cluster_id"]
    common = a.index.intersection(b.index)
    return a[common].tolist(), b[common].tolist(), len(common)


def compute_concordance(folds: list[dict]) -> pd.DataFrame:
    """Compute pairwise ARI and NMI for all fold pairs at all shared C values.

    Args:
        folds: List of fold dicts from discover_folds().

    Returns:
        DataFrame [fold_a, fold_b, pair, C, ARI, NMI, n_common_sectors].
        Empty DataFrame (with correct columns) if fewer than 2 folds.
    """
    _cols = ["fold_a", "fold_b", "pair", "C", "ARI", "NMI", "n_common_sectors"]
    if len(folds) < 2:
        logger.warning("Fewer than 2 folds found — nothing to compare.")
        return pd.DataFrame(columns=_cols)

    records = []
    for fold_a, fold_b in combinations(folds, 2):
        la, lb = fold_a["label"], fold_b["label"]
        pair = f"{la} vs {lb}"

        c_shared = sorted(
            set(fold_a["assignments"]["C"].unique())
            & set(fold_b["assignments"]["C"].unique())
        )
        if not c_shared:
            logger.warning("No shared C values for pair '%s'", pair)
            continue

        warned_mismatch = False
        for c in c_shared:
            vec_a, vec_b, n_common = _align_labels(
                fold_a["assignments"], fold_b["assignments"], c
            )
            if n_common == 0:
                continue

            n_a = fold_a["assignments"][
                fold_a["assignments"]["C"] == c
            ]["sector_id"].nunique()
            if not warned_mismatch and n_common < n_a:
                logger.warning(
                    "Pair '%s': sector sets differ — using %d "
                    "common sectors (fold A has %d)",
                    pair, n_common, n_a,
                )
                warned_mismatch = True

            ari = adjusted_rand_score(vec_a, vec_b)
            nmi = normalized_mutual_info_score(
                vec_a, vec_b, average_method="arithmetic"
            )
            records.append({
                "fold_a": la, "fold_b": lb, "pair": pair,
                "C": c,
                "ARI": round(ari, 6),
                "NMI": round(nmi, 6),
                "n_common_sectors": n_common,
            })

        logger.info("Pair '%s': %d C values computed", pair, len(c_shared))

    return pd.DataFrame(records) if records else pd.DataFrame(columns=_cols)


# ── Figure builders ───────────────────────────────────────────────────

def build_line_figure(concordance: pd.DataFrame) -> go.Figure:
    """Build ARI and NMI vs C line charts (one line per fold pair).

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

        fig.add_trace(
            go.Scatter(
                x=sub["C"], y=sub["ARI"],
                mode="lines+markers", name=pair,
                line={"color": color}, marker={"size": 5},
                legendgroup=pair,
                hovertemplate=(
                    f"<b>{pair}</b><br>C=%{{x}}<br>ARI=%{{y:.4f}}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["C"], y=sub["NMI"],
                mode="lines+markers", name=pair,
                line={"color": color, "dash": "dot"}, marker={"size": 5},
                legendgroup=pair, showlegend=False,
                hovertemplate=(
                    f"<b>{pair}</b><br>C=%{{x}}<br>NMI=%{{y:.4f}}"
                    "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
    fig.update_layout(
        title="SKATER-CV fold concordance — ARI (solid) and NMI (dotted) vs C",
        height=700,
        margin={"t": 80, "b": 60, "l": 60, "r": 20},
        xaxis2_title="C (number of clusters)",
        yaxis_title="ARI",
        yaxis2_title="NMI",
        legend={"title": "Fold pair (click to hide)"},
    )
    return fig


def build_heatmap_figure(
    concordance: pd.DataFrame, folds: list[dict],
) -> go.Figure:
    """Build a symmetric pairwise ARI heatmap with a C slider.

    Args:
        concordance: Output of compute_concordance().
        folds:       List of fold dicts (defines axis label order).

    Returns:
        go.Figure with one Heatmap trace per C value and a slider.
    """
    c_values = sorted(concordance["C"].unique())
    labels = [f["label"] for f in folds]
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
            i, j = idx[row["fold_a"]], idx[row["fold_b"]]
            mat[i][j] = mat[j][i] = float(row["ARI"])

        text = [
            [f"{v:.3f}" if not math.isnan(v) else "" for v in row]
            for row in mat
        ]
        traces.append(go.Heatmap(
            z=mat, x=labels, y=labels,
            colorscale="RdYlGn", zmin=0.0, zmax=1.0,
            visible=False, text=text, texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>ARI=%{z:.4f}<extra></extra>",
            showscale=True, colorbar={"title": "ARI"},
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
            "active": 0, "steps": steps,
            "x": 0.1, "len": 0.8, "y": -0.05, "yanchor": "top",
            "currentvalue": {
                "prefix": "C = ", "visible": True, "xanchor": "center",
            },
        }],
    )
    return fig


# ── Fold metadata table ────────────────────────────────────────────────

def _fold_table_html(folds: list[dict]) -> str:
    """Render a plain HTML table of test/train years for each fold.

    Args:
        folds: List of fold dicts from discover_folds().

    Returns:
        HTML <table> string for embedding in the dashboard page.
    """
    th = "border:1px solid #ccc;padding:4px 8px;background:#f0f0f0"
    td = "border:1px solid #ccc;padding:4px 8px"
    header = (
        f"<tr><th style='{th}'>fold (test year)</th>"
        f"<th style='{th}'>train years</th></tr>"
    )
    body = "".join(
        f"<tr><td style='{td}'><b>{f['label']}</b></td>"
        f"<td style='{td}'>{', '.join(f['meta']['train_years'])}</td></tr>"
        for f in folds
    )
    style = (
        "border-collapse:collapse;font-family:monospace;"
        "font-size:12px;width:100%;margin-bottom:16px"
    )
    return (
        f"<table style='{style}'><thead>{header}</thead>"
        f"<tbody>{body}</tbody></table>"
    )


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Discover CV folds, compute concordance, write the dashboard HTML."""
    run_label = _load_run_label()
    part_dir = PART_BASE / run_label
    out_dir = part_dir

    logger.info("Discovering CV folds in %s (run_label=%s)…", part_dir, run_label)
    folds = discover_folds(part_dir)
    if len(folds) < 2:
        logger.error(
            "Need ≥2 completed folds under %s. "
            "Run `python -m src.multiscale.skater_partitions` first.",
            part_dir,
        )
        return

    logger.info("Computing pairwise ARI/NMI for %d folds …", len(folds))
    concordance = compute_concordance(folds)
    if concordance.empty:
        logger.error("Concordance table empty — check fold outputs.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    concordance.to_csv(out_dir / "concordance_table.csv", index=False)
    logger.info("Saved concordance_table.csv")

    logger.info("Building figures …")
    fig_lines = build_line_figure(concordance)
    fig_heatmap = build_heatmap_figure(concordance, folds)

    lines_html = fig_lines.to_html(full_html=False, include_plotlyjs="cdn")
    heatmap_html = fig_heatmap.to_html(full_html=False, include_plotlyjs=False)
    table_html = _fold_table_html(folds)

    page = "\n".join([
        "<!DOCTYPE html><html>",
        "<head><meta charset='utf-8'>",
        "<title>SKATER-CV Fold Concordance</title>",
        "<style>body{font-family:sans-serif;padding:20px;"
        "max-width:1400px;margin:auto}</style>",
        "</head><body>",
        f"<h1>SKATER-CV fold concordance — {run_label}</h1>",
        "<h2>Folds</h2>",
        table_html,
        "<h2>ARI and NMI vs C</h2>",
        lines_html,
        "<h2>Pairwise ARI heatmap</h2>",
        heatmap_html,
        "</body></html>",
    ])
    out_path = out_dir / "concordance_dashboard.html"
    out_path.write_text(page, encoding="utf-8")
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
