"""SKATER transition correlation map -- S_min=3, C in {18,19,24,25}.

Builds a Results figure (sections/results.tex,
\\label{fig:skater_correlation_s3}): four pruning steps of the
S_min=3 run, colored by each region's pruning objective rho_c (best
signed Spearman correlation, lagged eggs vs dengue rate), on the
same color scale as figure_skater_correlation_s100.py.

The steps are two before/after pairs, (18,19) and (24,25), each
capturing one specific MST cut. Each pair creates its own new
region -- one born between C=18 and C=19, a different one born
between C=24 and C=25 -- detected programmatically
(find_new_small_region: smallest cluster present after the cut but
not before it) and marked with a black square (padded bounding box)
only in the "after" panel of its own pair (C=19, C=25) -- region
identity/highlighting is independent of the rho_c color scale.

Inputs:
  data/processed/bh_sectors_2022_with_populations.geojson
  results/skater/spearman_3/cluster_assignments.csv
  results/skater/spearman_3/cluster_diagnostics.csv
  results/skater/spearman_100/cluster_diagnostics.csv (for shared scale)
Output:
  6a441e20c1f1a66c183b3c38/Figures/figure_skater_correlation_s3.png
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
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
SECTORS_PATH = Path(
    "data/processed/bh_sectors_2022_with_populations.geojson"
)
RESULTS_DIR = Path("results/skater")
RUN_LABEL = "spearman_3"
C_VALUES = [18, 19, 24, 25]
# Each (before, after) pair's own newly-created region, marked only
# in that pair's "after" panel.
HIGHLIGHT_PAIRS = {19: (18, 19), 25: (24, 25)}
OUTPUT_PATH = Path(
    "6a441e20c1f1a66c183b3c38/Figures/figure_skater_correlation_s3.png"
)

# Other figure's run/C set, needed only to compute the shared color scale.
OTHER_RUN_LABEL = "spearman_100"
OTHER_C_VALUES = [1, 2, 3, 4]

CMAP = "coolwarm"
HIGHLIGHT_PADDING_FACTOR = 1.5
HIGHLIGHT_COLOR = "black"


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


def load_cluster_assignments(run_dir: Path) -> pd.DataFrame:
    """Load every pruning step's sector-to-cluster assignment.

    Args:
        run_dir: Path to a results/skater/<run_label> directory.

    Returns:
        DataFrame [C, sector_id, cluster_id], one row per
        (step, sector) pair.
    """
    return pd.read_csv(
        run_dir / "cluster_assignments.csv", dtype={"sector_id": str}
    )


def load_correlation_snapshot(
    assignments: pd.DataFrame, run_dir: Path, c: int
) -> pd.DataFrame:
    """Build one pruning step's sector-to-correlation mapping.

    Args:
        assignments: Full assignment history, from
            load_cluster_assignments().
        run_dir: Path to a results/skater/<run_label> directory.
        c: Number of clusters at the desired pruning step.

    Returns:
        DataFrame [sector_id, cluster_id, q_c] for that step.
    """
    diagnostics = pd.read_csv(run_dir / "cluster_diagnostics.csv")
    step = assignments[assignments["C"] == c][["sector_id", "cluster_id"]]
    q_at_c = diagnostics[diagnostics["C"] == c][["cluster_id", "q_c"]]
    return step.merge(q_at_c, on="cluster_id", how="left")


def find_new_small_region(
    assignments: pd.DataFrame, c_before: int, c_after: int
) -> frozenset[str]:
    """Find the smallest region newly created between two steps.

    Args:
        assignments: Full assignment history, from
            load_cluster_assignments().
        c_before: Number of clusters at the earlier step.
        c_after: Number of clusters at the later step (c_after ==
            c_before + 1 for a single MST cut).

    Returns:
        Frozenset of sector_ids belonging to the smallest cluster at
        c_after whose exact sector set did not exist at c_before.
    """
    before_sets = (
        assignments[assignments["C"] == c_before]
        .groupby("cluster_id")["sector_id"].apply(frozenset)
    )
    after_sets = (
        assignments[assignments["C"] == c_after]
        .groupby("cluster_id")["sector_id"].apply(frozenset)
    )
    before_values = set(before_sets.values)
    new_sets = [s for s in after_sets.values if s not in before_values]
    return min(new_sets, key=len)


def shared_color_range() -> tuple[float, float]:
    """Compute the combined rho_c range across both correlation figures.

    Returns:
        (vmin, vmax) spanning every panel in this figure and in
        figure_skater_correlation_s100.py, so both are on one scale.
    """
    own_assignments = load_cluster_assignments(RESULTS_DIR / RUN_LABEL)
    other_assignments = load_cluster_assignments(
        RESULTS_DIR / OTHER_RUN_LABEL
    )
    frames = [
        load_correlation_snapshot(own_assignments, RESULTS_DIR / RUN_LABEL, c)
        for c in C_VALUES
    ] + [
        load_correlation_snapshot(
            other_assignments, RESULTS_DIR / OTHER_RUN_LABEL, c
        )
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


def draw_highlight_square(
    ax: Axes,
    merged: gpd.GeoDataFrame,
    highlight_sectors: frozenset[str],
) -> None:
    """Mark the highlighted region's bounding box with a black square.

    The highlighted region is only ~0.006 x 0.004 degrees, invisible
    as a shape outline at full-city scale, so instead of tracing its
    exact boundary this draws a padded bounding-box square around it
    -- easy to spot without needing a zoomed inset.

    Args:
        ax: Main map panel axes to draw the square on.
        merged: Sector polygons for this pruning step, already
            carrying a q_c column, from main()'s per-panel merge.
        highlight_sectors: Sector ids making up the highlighted
            region, from find_new_small_region().
    """
    highlight = merged[merged["CD_SETOR"].isin(highlight_sectors)]
    hx0, hy0, hx1, hy1 = highlight.total_bounds
    pad_x = (hx1 - hx0) * HIGHLIGHT_PADDING_FACTOR
    pad_y = (hy1 - hy0) * HIGHLIGHT_PADDING_FACTOR
    ax.add_patch(Rectangle(
        (hx0 - pad_x, hy0 - pad_y),
        (hx1 - hx0) + 2 * pad_x, (hy1 - hy0) + 2 * pad_y,
        fill=False, edgecolor=HIGHLIGHT_COLOR, linewidth=1.8,
        zorder=5,
    ))


def main() -> None:
    """Build and save the four-panel transition correlation figure."""
    sectors = load_sectors(SECTORS_PATH)
    bounds = sectors.total_bounds
    mean_lat = float((bounds[1] + bounds[3]) / 2.0)
    vmin, vmax = shared_color_range()
    logger.info("Shared color scale: [%.3f, %.3f]", vmin, vmax)

    run_dir = RESULTS_DIR / RUN_LABEL
    assignments = load_cluster_assignments(run_dir)
    highlights_by_after_c = {}
    for after_c, (before_c, after_c_) in HIGHLIGHT_PAIRS.items():
        region = find_new_small_region(assignments, before_c, after_c_)
        highlights_by_after_c[after_c] = region
        logger.info(
            "Highlighted region: %d sectors, born between C=%d and C=%d",
            len(region), before_c, after_c_,
        )

    fig, axes = plt.subplots(2, 2, figsize=(9, 9.5))

    for ax, c in zip(axes.flat, C_VALUES):
        snapshot = load_correlation_snapshot(assignments, run_dir, c)
        merged = sectors.merge(
            snapshot, left_on="CD_SETOR", right_on="sector_id", how="inner"
        )
        merged.plot(
            ax=ax, column="q_c", cmap=CMAP, vmin=vmin, vmax=vmax,
            edgecolor="#333333", linewidth=0.15,
        )
        style_map_panel(ax, bounds, mean_lat, f"C = {c}")

        if c in highlights_by_after_c:
            draw_highlight_square(ax, merged, highlights_by_after_c[c])

    fig.suptitle(r"$S_{\min} = 3$", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])

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
