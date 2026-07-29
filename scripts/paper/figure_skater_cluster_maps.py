"""SKATER cluster-map comparison figure -- Belo Horizonte.

Builds a Results figure (sections/results.tex,
\\label{fig:skater_cluster_maps}): the final region clusterization
for each of the four SKATER runs that vary only the minimum region
size guard S_min (3, 30, 50, 100), letting the reader see how a
stricter size floor merges small regions into larger ones.

Inputs:
  data/processed/bh_sectors_2022_with_populations.geojson
  results/skater/spearman_{3,30,50,100}/run_params.json
  results/skater/spearman_{3,30,50,100}/cluster_assignments.csv
Output:
  6a441e20c1f1a66c183b3c38/Figures/figure_skater_cluster_maps.png
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
SECTORS_PATH = Path(
    "data/processed/bh_sectors_2022_with_populations.geojson"
)
RESULTS_DIR = Path("results/skater")
RUN_LABELS = ["spearman_3", "spearman_30", "spearman_50", "spearman_100"]
OUTPUT_PATH = Path(
    "6a441e20c1f1a66c183b3c38/Figures/figure_skater_cluster_maps.png"
)

EDGE_COLOR = "white"
MAP_PADDING_FRAC = 0.02


def load_sectors(path: Path) -> gpd.GeoDataFrame:
    """Load BH census sector polygons.

    Args:
        path: Path to the sectors GeoJSON.

    Returns:
        GeoDataFrame with a CD_SETOR (str) column plus geometry.
    """
    gdf = gpd.read_file(path)
    gdf["CD_SETOR"] = gdf["CD_SETOR"].astype(str)
    return gdf


def load_final_partition(run_dir: Path) -> tuple[pd.DataFrame, int, int]:
    """Load a run's final cluster assignment and its S_min setting.

    Args:
        run_dir: Path to a results/skater/<run_label> directory.

    Returns:
        (assignments, s_min, k_final):
          assignments -- DataFrame [sector_id, cluster_id] at the
            run's final pruning step (max C).
          s_min       -- the run's configured minimum region size.
          k_final     -- number of clusters in the final partition.
    """
    assign = pd.read_csv(
        run_dir / "cluster_assignments.csv", dtype={"sector_id": str}
    )
    final = assign[assign["C"] == assign["C"].max()]
    with open(run_dir / "run_params.json") as fh:
        s_min = json.load(fh)["S_min"]
    k_final = int(final["cluster_id"].nunique())
    return final[["sector_id", "cluster_id"]], s_min, k_final


def style_map_panel(
    ax: Axes, bounds: np.ndarray, mean_lat: float, title: str
) -> None:
    """Apply shared framing, aspect, and title to a map panel.

    Args:
        ax: Target axes.
        bounds: (minx, miny, maxx, maxy) shared across all panels.
        mean_lat: Mean latitude, used for a geographic aspect ratio.
        title: Bold title drawn above the map.
    """
    minx, miny, maxx, maxy = bounds
    pad_x = (maxx - minx) * MAP_PADDING_FRAC
    pad_y = (maxy - miny) * MAP_PADDING_FRAC
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)


def main() -> None:
    """Build and save the four-panel cluster-map comparison figure."""
    # ── Load data ─────────────────────────────────────────────
    sectors = load_sectors(SECTORS_PATH)
    bounds = sectors.total_bounds
    mean_lat = float((bounds[1] + bounds[3]) / 2.0)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.0))

    for ax, run_label in zip(axes.flat, RUN_LABELS):
        run_dir = RESULTS_DIR / run_label
        assignments, s_min, k_final = load_final_partition(run_dir)
        merged = sectors.merge(
            assignments, left_on="CD_SETOR", right_on="sector_id", how="inner"
        )
        merged.plot(
            ax=ax, column="cluster_id", cmap="tab20",
            edgecolor=EDGE_COLOR, linewidth=0.1,
        )
        style_map_panel(
            ax, bounds, mean_lat,
            f"$S_{{\\min}}$ = {s_min} (K = {k_final})",
        )
        logger.info("%s: S_min=%d, K=%d", run_label, s_min, k_final)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
