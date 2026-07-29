"""SKATER cluster-concordance figure -- ARI vs K.

Builds a Results figure (sections/results.tex,
\\label{fig:skater_concordance}): Adjusted Rand Index (ARI) between
every pair of the four S_min runs, as a function of the number of
regions K (the `C` column in concordance_table.csv). Mirrors the
layout of the interactive dashboard produced by
scripts/skater_concordance_dashboard.py (one line per run pair),
rendered statically for the paper.

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
OUTPUT_PATH = Path("6a441e20c1f1a66c183b3c38/Figures/figure_skater_concordance.pdf")

PAIR_COLORS = {
    "spearman_100 vs spearman_3": "#1b9e77",
    "spearman_100 vs spearman_30": "#d95f02",
    "spearman_100 vs spearman_50": "#7570b3",
    "spearman_3 vs spearman_30": "#e7298a",
    "spearman_3 vs spearman_50": "#66a61e",
    "spearman_30 vs spearman_50": "#e6ab02",
}


def load_concordance(path: Path) -> pd.DataFrame:
    """Load pairwise ARI-vs-C concordance data.

    Args:
        path: Path to concordance_table.csv, with columns
            [run_a, run_b, pair, C, ARI, NMI, n_common_sectors].

    Returns:
        The loaded DataFrame, unmodified.
    """
    return pd.read_csv(path)


def build_figure(concordance: pd.DataFrame) -> None:
    """Draw the ARI-vs-C figure and save it to OUTPUT_PATH.

    Args:
        concordance: DataFrame from load_concordance().
    """
    fig, ax_ari = plt.subplots(figsize=(6.5, 3.5))

    for pair, sub in concordance.groupby("pair"):
        sub = sub.sort_values("C")
        color = PAIR_COLORS.get(pair)
        # "spearman_" dropped entirely (title="S_min" on the legend
        # carries that) so each entry is short enough for a single
        # vertical column of six rows.
        label = pair.replace("spearman_", "")
        ax_ari.plot(sub["C"], sub["ARI"], color=color, label=label, linewidth=1.6)

    # Tight y-limits -- ARI dips down to ~0.5 here, so a fixed
    # [-0.05, 1.05] scale wastes much of the panel on empty space.
    ymin = concordance["ARI"].min()
    pad = (1.0 - ymin) * 0.08
    ax_ari.set_ylabel("ARI")
    ax_ari.set_ylim(ymin - pad, 1.0 + pad)
    ax_ari.grid(alpha=0.3)

    ax_ari.set_xlabel("Number of regions (K)")
    ax_ari.legend(
        loc="lower right", ncol=1, frameon=False, fontsize=8, title="S_min",
        title_fontsize=8,
    )

    fig.tight_layout(pad=0.3)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    logger.info("Saved figure to %s", OUTPUT_PATH)


def main() -> None:
    """Load concordance data and build the figure."""
    concordance = load_concordance(CONCORDANCE_PATH)
    logger.info("Loaded %d concordance rows", len(concordance))
    build_figure(concordance)


if __name__ == "__main__":
    main()
