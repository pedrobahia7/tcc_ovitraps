"""Spatial-scale vs prediction-accuracy trade-off dashboard.

Merges the baseline (city / district / sector) and SKATER-CV per-C
results into one master summary and renders the accuracy-vs-granularity
trade-off: population-weighted mean held-out RMSE against the number of
spatial units (log axis), with the SKATER curve over C spanning the
region between the coarse (city) and fine (sector) administrative
baselines.

Inputs:
  results/multiscale/baselines/summary_baselines.csv
  results/multiscale/skater_cv/skater_by_c.csv
Outputs:
  results/multiscale/summary.csv           — master per-scale/per-C table.
  results/multiscale/multiscale_tradeoff.html — interactive figure.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_BASE = Path("results/multiscale")
_BASELINES = _BASE / "baselines" / "summary_baselines.csv"
_SKATER = _BASE / "skater_cv" / "skater_by_c.csv"
_SUMMARY_OUT = _BASE / "summary.csv"
_HTML_OUT = _BASE / "multiscale_tradeoff.html"

# Number of spatial units per fixed baseline scale (x-axis position).
_BASELINE_X = {"city": 1, "district": 9}


def build_master_summary(
    baselines: pd.DataFrame, skater: pd.DataFrame
) -> pd.DataFrame:
    """Combine baseline and SKATER rows into one master summary table.

    Args:
        baselines: summary_baselines.csv contents.
        skater:    skater_by_c.csv contents.

    Returns:
        DataFrame [scale, C, n_regions, pop_wt_rmse, naive_pop_wt_rmse,
        mean_r2] with SKATER rows carrying their C value and n_regions.
    """
    rows = []
    for _, r in baselines.iterrows():
        rows.append(
            {
                "scale": r["scale"],
                "C": pd.NA,
                "n_regions": _BASELINE_X.get(r["scale"], int(r["n_units"])),
                "pop_wt_rmse": r["pop_wt_rmse"],
                "naive_pop_wt_rmse": r["naive_pop_wt_rmse"],
                "mean_r2": r["mean_r2"],
            }
        )
    for _, r in skater.iterrows():
        rows.append(
            {
                "scale": "skater",
                "C": int(r["C"]),
                "n_regions": int(r["C"]),
                "pop_wt_rmse": r["pop_wt_rmse"],
                "naive_pop_wt_rmse": r["naive_pop_wt_rmse"],
                "mean_r2": r["mean_r2"],
            }
        )
    return pd.DataFrame(rows)


def build_figure(master: pd.DataFrame) -> go.Figure:
    """Render the accuracy-vs-granularity trade-off figure.

    Args:
        master: Output of build_master_summary.

    Returns:
        A Plotly figure: SKATER RMSE curve over C plus the three fixed
        administrative baselines as reference markers (log x-axis).
    """
    sk = master[master["scale"] == "skater"].sort_values("n_regions")
    fig = go.Figure()

    # ── SKATER curve over C ───────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=sk["n_regions"], y=sk["pop_wt_rmse"],
            mode="lines+markers", name="SKATER (per C)",
            line={"color": "#1f77b4", "width": 2},
            marker={"size": 7},
            hovertemplate="C=%{x}<br>RMSE=%{y:.4f}<extra></extra>",
        )
    )

    # ── Fixed administrative baselines ────────────────────────────────
    palette = {"city": "#d62728", "district": "#2ca02c", "sector": "#9467bd"}
    for scale in ("city", "district", "sector"):
        row = master[master["scale"] == scale]
        if row.empty:
            continue
        row = row.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[row["n_regions"]], y=[row["pop_wt_rmse"]],
                mode="markers+text", name=scale,
                marker={"size": 14, "color": palette[scale],
                        "symbol": "diamond"},
                text=[scale], textposition="top center",
                hovertemplate=(
                    f"{scale}<br>units=%{{x}}<br>RMSE=%{{y:.4f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            "Spatial scale vs prediction accuracy — "
            "population-weighted held-out RMSE (EB dengue rate)"
        ),
        xaxis={
            "title": "Number of spatial units (log)",
            "type": "log",
        },
        yaxis={"title": "Pop-weighted mean test RMSE"},
        height=650,
        legend={"x": 1.01, "xanchor": "left", "y": 1.0},
        margin={"t": 80, "r": 160},
    )
    return fig


def main() -> None:
    """Build the master summary CSV and the trade-off dashboard."""
    baselines = pd.read_csv(_BASELINES)
    skater = pd.read_csv(_SKATER)

    master = build_master_summary(baselines, skater)
    master.to_csv(_SUMMARY_OUT, index=False)
    logger.info("Saved %s\n%s", _SUMMARY_OUT, master.to_string(index=False))

    fig = build_figure(master)
    fig.write_html(str(_HTML_OUT))
    logger.info("Saved %s", _HTML_OUT)


if __name__ == "__main__":
    main()
