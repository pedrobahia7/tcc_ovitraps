"""SKATER cluster-concordance figure -- ARI and NMI vs C.

Builds a Results figure (sections/results.tex,
\\label{fig:skater_concordance}): Adjusted Rand Index (ARI, top) and
Normalized Mutual Information (NMI, bottom) between every pair of the
four S_min runs, as a function of the number of clusters C. Mirrors
the layout of the interactive dashboard produced by
scripts/skater_concordance_dashboard.py (two stacked subplots, one
line per run pair, shared color across the ARI/NMI rows), rendered
statically for the paper.

Inputs:
  results/skater/concordance_table.csv
Output:
  6a441e20c1f1a66c183b3c38/Figures/figure_skater_concordance.pdf
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
CONCORDANCE_PATH = Path("results/skater/concordance_table.csv")
OUTPUT_PATH = Path(
    "6a441e20c1f1a66c183b3c38/Figures/figure_skater_concordance.pdf"
)

PAIR_COLORS = {
    "spearman_100 vs spearman_3": "#1b9e77",
    "spearman_100 vs spearman_30": "#d95f02",
    "spearman_100 vs spearman_50": "#7570b3",
    "spearman_3 vs spearman_30": "#e7298a",
    "spearman_3 vs spearman_50": "#66a61e",
    "spearman_30 vs spearman_50": "#e6ab02",
}


def load_concordance(path: Path) -> pd.DataFrame:
    """Load pairwise ARI/NMI-vs-C concordance data.

    Args:
        path: Path to concordance_table.csv, with columns
            [run_a, run_b, pair, C, ARI, NMI, n_common_sectors].

    Returns:
        The loaded DataFrame, unmodified.
    """
    return pd.read_csv(path)


def build_figure(concordance: pd.DataFrame) -> None:
    """Draw the two-row ARI/NMI-vs-C figure and save it to OUTPUT_PATH.

    Args:
        concordance: DataFrame from load_concordance().
    """
    fig, (ax_ari, ax_nmi) = plt.subplots(
        2, 1, figsize=(6.5, 6.5), sharex=True
    )

    for pair, sub in concordance.groupby("pair"):
        sub = sub.sort_values("C")
        color = PAIR_COLORS.get(pair)
        ax_ari.plot(sub["C"], sub["ARI"], color=color, label=pair, linewidth=1.6)
        ax_nmi.plot(sub["C"], sub["NMI"], color=color, label=pair, linewidth=1.6)

    for ax, title in ((ax_ari, "ARI vs C"), (ax_nmi, "NMI vs C")):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(title.split(" ")[0])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    ax_nmi.set_xlabel("Number of clusters (C)")
    ax_ari.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.35),
        ncol=2, frameon=False, fontsize=7,
    )

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", OUTPUT_PATH)


def main() -> None:
    """Load concordance data and build the figure."""
    concordance = load_concordance(CONCORDANCE_PATH)
    logger.info("Loaded %d concordance rows", len(concordance))
    build_figure(concordance)


if __name__ == "__main__":
    main()
