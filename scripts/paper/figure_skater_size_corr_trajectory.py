"""SKATER size-vs-objective correlation trajectory -- correlation vs C, all runs.

Builds a Results figure (sections/results.tex,
\\label{fig:skater_size_corr_trajectory}): at each pruning step,
the Spearman correlation between cluster size (n_sectors) and
pruning objective (rho_c) across that step's clusters, as a
function of the number of clusters C, one line per S_min run.
Mirrors figure_skater_q_trajectory.py's style (RUN_COLORS palette,
marker at each run's final stopping point).

Clusters with n_valid_pairs < N_min are excluded per step -- their
rho_c is force-zeroed by prune.py's data-sparsity guard, not a real
measurement, and would bias the correlation. Each run uses its own
N_min (run_params.json), not a shared constant. Steps with fewer
than 2 non-zeroed clusters (or zero variance in either variable)
have no defined correlation and are left as a gap in the line.

Inputs:
  results/skater/spearman_{3,30,50,100}/cluster_diagnostics.csv
  results/skater/spearman_{3,30,50,100}/run_params.json
  results/skater/spearman_{3,30,50,100}/stop_info.json
Output:
  6a441e20c1f1a66c183b3c38/Figures/figure_skater_size_corr_trajectory.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/skater")
RUN_LABELS = ["spearman_3", "spearman_30", "spearman_50", "spearman_100"]
OUTPUT_PATH = Path(
    "6a441e20c1f1a66c183b3c38/Figures/figure_skater_size_corr_trajectory.pdf"
)

RUN_COLORS = {
    "spearman_3": "#1b9e77",
    "spearman_30": "#d95f02",
    "spearman_50": "#7570b3",
    "spearman_100": "#e7298a",
}


def load_run(run_dir: Path) -> tuple[pd.DataFrame, int, int, dict]:
    """Load one run's cluster diagnostics, S_min, N_min, and stop reason.

    Args:
        run_dir: Path to a results/skater/<run_label> directory.

    Returns:
        (diag, s_min, n_min, stop_info):
          diag      -- DataFrame [C, cluster_id, q_c, n_sectors,
                        n_valid_pairs].
          s_min     -- the run's configured minimum region size.
          n_min     -- the run's configured N_min guard threshold.
          stop_info -- parsed stop_info.json dict.
    """
    diag = pd.read_csv(run_dir / "cluster_diagnostics.csv")
    with open(run_dir / "run_params.json") as fh:
        params = json.load(fh)
    with open(run_dir / "stop_info.json") as fh:
        stop_info = json.load(fh)
    return diag, params["S_min"], params["N_min"], stop_info


def size_metric_correlations(diag: pd.DataFrame, n_min: int) -> pd.DataFrame:
    """Compute per-C Spearman correlation between n_sectors and rho_c.

    Args:
        diag:  Cluster diagnostics [C, cluster_id, q_c, n_sectors,
            n_valid_pairs].
        n_min: N_min guard threshold used by the originating run.

    Returns:
        DataFrame [C, spearman_rho, n_points_used], sorted by C.
        spearman_rho is NaN when fewer than 2 non-zeroed clusters or
        zero variance in n_sectors/rho_c exist at that step.
    """
    records = []
    for c, step in diag.groupby("C"):
        real = step[step["n_valid_pairs"] >= n_min]
        n_used = len(real)
        rho = float("nan")
        if n_used >= 2:
            x = real["n_sectors"].to_numpy(dtype=float)
            y = real["q_c"].to_numpy(dtype=float)
            if x.std() > 1e-12 and y.std() > 1e-12:
                rho = float(spearmanr(x, y).statistic)
        records.append({"C": c, "spearman_rho": rho, "n_points_used": n_used})
    return pd.DataFrame(records).sort_values("C")


def build_figure(runs: list[tuple[str, pd.DataFrame, int, dict]]) -> None:
    """Draw the rho-vs-C trajectory figure and save it to OUTPUT_PATH.

    Args:
        runs: List of (run_label, trajectory, s_min, stop_info) tuples,
            trajectory being size_metric_correlations()'s output.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for run_label, trajectory, s_min, stop_info in runs:
        color = RUN_COLORS.get(run_label)
        label = f"$S_{{\\min}}$ = {s_min}"
        ax.plot(
            trajectory["C"], trajectory["spearman_rho"],
            color=color, linewidth=1.8, label=label,
        )
        final_c = stop_info["C_final"]
        final_row = trajectory[trajectory["C"] == final_c]
        if not final_row.empty and not np.isnan(
            final_row["spearman_rho"].iloc[0]
        ):
            ax.scatter(
                final_row["C"], final_row["spearman_rho"],
                color=color, zorder=5, s=40,
                edgecolor="white", linewidth=0.8,
            )

    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Number of regions (K)")
    ax.set_ylabel("Spearman correlation")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", OUTPUT_PATH)


def main() -> None:
    """Load all four runs, compute trajectories, and build the figure."""
    runs = []
    for run_label in RUN_LABELS:
        diag, s_min, n_min, stop_info = load_run(RESULTS_DIR / run_label)
        trajectory = size_metric_correlations(diag, n_min)
        runs.append((run_label, trajectory, s_min, stop_info))
        logger.info(
            "%s: S_min=%d, N_min=%d, C_final=%d, reason=%s",
            run_label, s_min, n_min, stop_info["C_final"], stop_info["reason"],
        )
    build_figure(runs)


if __name__ == "__main__":
    main()
