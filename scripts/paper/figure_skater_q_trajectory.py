"""SKATER objective-trajectory figure -- F vs C for all four runs.

Builds a Results figure (sections/results.tex,
\\label{fig:skater_q_trajectory}): the global population-weighted
objective F (Section~\\ref{sec:methodology}) as a function of the
number of regions K, one line per S_min run, with a marker at each
run's final stopping point (results/skater/<run>/stop_info.json).

Inputs:
  results/skater/spearman_{3,30,50,100}/q_trajectory.csv
  results/skater/spearman_{3,30,50,100}/run_params.json
  results/skater/spearman_{3,30,50,100}/stop_info.json
Output:
  6a441e20c1f1a66c183b3c38/Figures/figure_skater_q_trajectory.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/skater")
RUN_LABELS = ["spearman_3", "spearman_30", "spearman_50", "spearman_100"]
OUTPUT_PATH = Path(
    "6a441e20c1f1a66c183b3c38/Figures/figure_skater_q_trajectory.pdf"
)

RUN_COLORS = {
    "spearman_3": "#1b9e77",
    "spearman_30": "#d95f02",
    "spearman_50": "#7570b3",
    "spearman_100": "#e7298a",
}


def load_run_trajectory(run_dir: Path) -> tuple[pd.DataFrame, int, dict]:
    """Load one run's F-vs-K trajectory, S_min, and stop reason.

    Args:
        run_dir: Path to a results/skater/<run_label> directory.

    Returns:
        (trajectory, s_min, stop_info):
          trajectory -- DataFrame with columns [C, Q] (K and F in the
            paper's notation; column names match the source CSV).
          s_min      -- the run's configured minimum region size.
          stop_info  -- parsed stop_info.json dict.
    """
    trajectory = pd.read_csv(run_dir / "q_trajectory.csv")
    with open(run_dir / "run_params.json") as fh:
        s_min = json.load(fh)["S_min"]
    with open(run_dir / "stop_info.json") as fh:
        stop_info = json.load(fh)
    return trajectory, s_min, stop_info


def build_figure(runs: list[tuple[str, pd.DataFrame, int, dict]]) -> None:
    """Draw the F-vs-K trajectory figure and save it to OUTPUT_PATH.

    Args:
        runs: List of (run_label, trajectory, s_min, stop_info) tuples.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for run_label, trajectory, s_min, stop_info in runs:
        color = RUN_COLORS.get(run_label)
        label = f"$S_{{\\min}}$ = {s_min}"
        ax.plot(
            trajectory["C"], trajectory["Q"],
            color=color, linewidth=1.8, label=label,
        )
        final_row = trajectory.iloc[-1]
        ax.scatter(
            [final_row["C"]], [final_row["Q"]],
            color=color, zorder=5, s=40, edgecolor="white", linewidth=0.8,
        )

    ax.set_xlabel("Number of regions (K)")
    ax.set_ylabel("Global objective (F)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", OUTPUT_PATH)


def main() -> None:
    """Load all four runs' trajectories and build the figure."""
    runs = []
    for run_label in RUN_LABELS:
        trajectory, s_min, stop_info = load_run_trajectory(
            RESULTS_DIR / run_label
        )
        runs.append((run_label, trajectory, s_min, stop_info))
        logger.info(
            "%s: S_min=%d, C_final=%d, reason=%s",
            run_label, s_min, trajectory["C"].max(), stop_info["reason"],
        )
    build_figure(runs)


if __name__ == "__main__":
    main()
