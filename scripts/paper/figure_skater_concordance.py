"""SKATER cluster-concordance figure -- ARI and NMI vs K.

Builds a Results figure (sections/results.tex,
\\label{fig:skater_concordance}): Adjusted Rand Index (ARI, top) and
Normalized Mutual Information (NMI, bottom) between every pair of the
four S_min runs, as a function of the number of regions K (the
`C` column in concordance_table.csv). Mirrors
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
    fig, (ax_ari, ax_nmi) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True)

    for pair, sub in concordance.groupby("pair"):
        sub = sub.sort_values("C")
        color = PAIR_COLORS.get(pair)
        # "spearman_" renamed to the paper's "S_min=" notation -- at
        # column width the raw pair name ("spearman_100 vs
        # spearman_3") doesn't leave room for six legend entries.
        label = pair.replace("spearman_", "S_min=")
        ax_ari.plot(sub["C"], sub["ARI"], color=color, label=label, linewidth=1.6)
        ax_nmi.plot(sub["C"], sub["NMI"], color=color, label=label, linewidth=1.6)

    # Tight, metric-specific y-limits -- ARI and NMI occupy very
    # different ranges here (ARI down to ~0.5, NMI never below
    # ~0.8), so a shared [-0.05, 1.05] scale wastes most of the NMI
    # panel on empty space. NMI's floor is fixed at 0.6 rather than
    # the data min, for a consistent axis across runs of this script.
    y_mins = {"ARI": concordance["ARI"].min(), "NMI": 0.6}
    for ax, ylabel in ((ax_ari, "ARI"), (ax_nmi, "NMI")):
        ymin = y_mins[ylabel]
        pad = (1.0 - ymin) * 0.08
        ax.set_ylabel(ylabel)
        ax.set_ylim(ymin - pad, 1.0 + pad)
        ax.grid(alpha=0.3)

    ax_nmi.set_xlabel("Number of regions (K)")
    ax_nmi.legend(loc="lower right", ncol=2, frameon=False, fontsize=9)

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
