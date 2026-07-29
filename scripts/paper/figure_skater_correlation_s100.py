"""SKATER early-split correlation map -- S_min=100, C in {1,2,3,4}.

Builds a Results figure (sections/results.tex,
\\label{fig:skater_correlation_s100}): the first four pruning steps
of the S_min=100 run, colored by each region's pruning objective
rho_c (best signed Spearman correlation, lagged eggs vs dengue
rate) instead of by categorical cluster identity. Region
shape/identity is conveyed by the boundary lines alone, letting one
continuous colorbar stand in for what would otherwise be a separate
bar chart.

The color scale (VMIN, VMAX) is fixed to the combined rho_c range
across BOTH this figure and figure_skater_correlation_s3.py's
panels, so the two figures are directly comparable.

Inputs:
  data/processed/bh_sectors_2022_with_populations.geojson
  results/skater/spearman_100/cluster_assignments.csv
  results/skater/spearman_100/cluster_diagnostics.csv
  results/skater/spearman_3/cluster_diagnostics.csv (for shared scale)
Output:
  6a441e20c1f1a66c183b3c38/Figures/figure_skater_correlation_s100.png
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
SECTORS_PATH = Path(
    "data/processed/bh_sectors_2022_with_populations.geojson"
)
RESULTS_DIR = Path("results/skater")
RUN_LABEL = "spearman_100"
C_VALUES = [1, 2, 3, 4]
OUTPUT_PATH = Path(
    "6a441e20c1f1a66c183b3c38/Figures/figure_skater_correlation_s100.png"
)

# Other figure's run/C set, needed only to compute the shared color scale.
OTHER_RUN_LABEL = "spearman_3"
OTHER_C_VALUES = [18, 19, 24, 25]

CMAP = "coolwarm"


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


def load_correlation_snapshot(
    run_dir: Path, c: int
) -> pd.DataFrame:
    """Load one pruning step's sector-to-correlation mapping.

    Args:
        run_dir: Path to a results/skater/<run_label> directory.
        c: Number of clusters at the desired pruning step.

    Returns:
        DataFrame [sector_id, cluster_id, q_c] for that step.
    """
    assign = pd.read_csv(
        run_dir / "cluster_assignments.csv", dtype={"sector_id": str}
    )
    diagnostics = pd.read_csv(run_dir / "cluster_diagnostics.csv")
    step = assign[assign["C"] == c][["sector_id", "cluster_id"]]
    q_at_c = diagnostics[diagnostics["C"] == c][["cluster_id", "q_c"]]
    return step.merge(q_at_c, on="cluster_id", how="left")


def shared_color_range() -> tuple[float, float]:
    """Compute the combined rho_c range across both correlation figures.

    Returns:
        (vmin, vmax) spanning every panel in this figure and in
        figure_skater_correlation_s3.py, so both are on one scale.
    """
    frames = [
        load_correlation_snapshot(RESULTS_DIR / RUN_LABEL, c)
        for c in C_VALUES
    ] + [
        load_correlation_snapshot(RESULTS_DIR / OTHER_RUN_LABEL, c)
        for c in OTHER_C_VALUES
    ]
    all_q = pd.concat(frames)["q_c"]
    return float(all_q.min()), float(all_q.max())


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
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)


def main() -> None:
    """Build and save the four-panel early-split correlation figure."""
    sectors = load_sectors(SECTORS_PATH)
    bounds = sectors.total_bounds
    mean_lat = float((bounds[1] + bounds[3]) / 2.0)
    vmin, vmax = shared_color_range()
    logger.info("Shared color scale: [%.3f, %.3f]", vmin, vmax)

    fig, axes = plt.subplots(2, 2, figsize=(9, 9.5))

    for ax, c in zip(axes.flat, C_VALUES):
        snapshot = load_correlation_snapshot(RESULTS_DIR / RUN_LABEL, c)
        merged = sectors.merge(
            snapshot, left_on="CD_SETOR", right_on="sector_id", how="inner"
        )
        merged.plot(
            ax=ax, column="q_c", cmap=CMAP, vmin=vmin, vmax=vmax,
            edgecolor="#333333", linewidth=0.15,
        )
        style_map_panel(ax, bounds, mean_lat, f"C = {c}")

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=CMAP),
        cax=cbar_ax, label=r"$\rho_c$ (lagged Spearman correlation)",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
