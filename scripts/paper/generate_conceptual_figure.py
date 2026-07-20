"""Generate Figure 1 (conceptual trade-off) for the AAAI draft.

Follows the layout of the original Overleaf placeholder (recovered
from the project's Overleaf git history, commit f089338): three
real-data map panels laid out by spatial scale, plus a conceptual
trade-off curve underneath, replacing the placeholder referenced in
6a441e20c1f1a66c183b3c38/sections/introduction.tex
(\\label{fig:conceptual_tradeoff}).

Panels (left -> right, fine -> coarse spatial scale):
  a) Individual ovitrap locations — one model per trap.
  b) Predefined administrative regions (BH's 9 official Regionais,
     built up from IBGE census sectors) — region-specific models.
  c) Entire city as a single unit — one citywide model.
Below: a schematic trade-off curve (prediction accuracy rising,
operational usefulness falling with spatial aggregation), whose
crossing point is horizontally aligned with panel (b) — the actual,
currently used administrative partition of the city.

The SKATER-learned partition is intentionally NOT part of this
figure (it belongs to the methodology/results discussion). It is
still rendered and saved separately for later reuse.

Inputs:
  data/processed/bh_sectors_2022_with_populations.geojson — census
    sector polygons, carries the NM_SUBDIST (Regional) label.
  data/processed/ovitraps_data.csv — ovitrap records with lat/long.
  results/skater/spearman_100/cluster_assignments.csv — learned
    SKATER partition (final step, C=20), for the standalone figure.
Outputs:
  6a441e20c1f1a66c183b3c38/Figures/conceptual_figure.png — Figure 1.
  6a441e20c1f1a66c183b3c38/Figures/skater_learned_regions.png —
    standalone SKATER partition map, saved for later use.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
SECTORS_PATH = Path(
    "data/processed/bh_sectors_2022_with_populations.geojson"
)
OVITRAPS_PATH = Path("data/processed/ovitraps_data.csv")
CLUSTERS_PATH = Path(
    "results/skater/spearman_100/cluster_assignments.csv"
)
FIGURES_DIR = Path("6a441e20c1f1a66c183b3c38/Figures")
COMPOSITE_OUTPUT_PATH = FIGURES_DIR / "conceptual_figure.png"
SKATER_OUTPUT_PATH = FIGURES_DIR / "skater_learned_regions.png"

# ── Style constants ───────────────────────────────────────────────
CITY_FILL = "#5A7EA6"
OVITRAP_COLOR = "#B0302A"
EDGE_COLOR = "white"
ACCURACY_COLOR = "#2E7D32"
USEFULNESS_COLOR = "#C62828"
MAP_PADDING_FRAC = 0.03


def load_sectors(path: Path) -> gpd.GeoDataFrame:
    """Load BH census sector polygons with Regional labels.

    Args:
        path: Path to the sectors GeoJSON.

    Returns:
        GeoDataFrame with CD_SETOR (str) and NM_SUBDIST (Regional
        name) columns plus geometry.
    """
    gdf = gpd.read_file(path)
    gdf["CD_SETOR"] = gdf["CD_SETOR"].astype(str)
    return gdf


def load_ovitraps(path: Path) -> pd.DataFrame:
    """Load unique ovitrap coordinates.

    Args:
        path: Path to the ovitrap records CSV.

    Returns:
        DataFrame with one row per trap (idarmad, latitude,
        longitude), duplicate collection events dropped.
    """
    cols = ["idarmad", "latitude", "longitude"]
    df = pd.read_csv(path, usecols=cols)
    return df.dropna(subset=["latitude", "longitude"]).drop_duplicates(
        "idarmad"
    )


def attach_learned_clusters(
    sectors: gpd.GeoDataFrame, path: Path
) -> gpd.GeoDataFrame:
    """Join the final SKATER partition onto the sector polygons.

    Args:
        sectors: Sector GeoDataFrame from load_sectors().
        path: Path to a SKATER cluster_assignments.csv, which
            stores every pruning step under column "C" — only the
            final step (max C) is the finished partition.

    Returns:
        Copy of sectors with a "cluster_id" column, restricted to
        sectors present in the final partition.
    """
    assign = pd.read_csv(path, dtype={"sector_id": str})
    final = assign[assign["C"] == assign["C"].max()]
    merged = sectors.merge(
        final[["sector_id", "cluster_id"]],
        left_on="CD_SETOR",
        right_on="sector_id",
        how="inner",
    )
    return merged


def style_map_panel(
    ax: Axes,
    bounds: np.ndarray,
    mean_lat: float,
    title: str,
    caption: str,
) -> None:
    """Apply shared framing, aspect, title and caption to a map panel.

    Args:
        ax: Target axes.
        bounds: (minx, miny, maxx, maxy) shared across all panels.
        mean_lat: Mean latitude, used for a geographic aspect ratio.
        title: Bold title drawn above the map.
        caption: Caption drawn below the map.
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
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.text(
        0.5, -0.05, caption,
        transform=ax.transAxes,
        ha="center", va="top", fontsize=9,
    )


def plot_ovitrap_panel(
    ax: Axes, sectors: gpd.GeoDataFrame, ovitraps: pd.DataFrame
) -> None:
    """Fine spatial scale: city outline + individual ovitraps."""
    sectors.plot(
        ax=ax, color="#EAEAEA", edgecolor="#BBBBBB", linewidth=0.2
    )
    ax.scatter(
        ovitraps["longitude"], ovitraps["latitude"],
        s=3, color=OVITRAP_COLOR, alpha=0.6, linewidths=0,
    )


def plot_admin_panel(ax: Axes, sectors: gpd.GeoDataFrame) -> None:
    """Intermediate spatial scale: predefined administrative regions."""
    sectors.plot(
        ax=ax, column="NM_SUBDIST", cmap="tab10",
        edgecolor=EDGE_COLOR, linewidth=0.15,
    )
    n_regions = sectors["NM_SUBDIST"].nunique()
    ax.text(
        0.98, 0.02, f"{n_regions} regions (Regionais)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
    )


def plot_city_panel(ax: Axes, sectors: gpd.GeoDataFrame) -> None:
    """Coarse spatial scale: entire city as one operational unit."""
    boundary = sectors.dissolve()
    boundary.plot(ax=ax, color=CITY_FILL, edgecolor="#333333", linewidth=0.8)


def plot_learned_panel(ax: Axes, sectors_learned: gpd.GeoDataFrame) -> None:
    """Learned SKATER operational regions (standalone figure only)."""
    sectors_learned.plot(
        ax=ax, column="cluster_id", cmap="tab20",
        edgecolor=EDGE_COLOR, linewidth=0.15,
    )
    n_regions = sectors_learned["cluster_id"].nunique()
    ax.text(
        0.98, 0.02, f"K={n_regions} regions",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
    )


def plot_tradeoff(ax: Axes) -> None:
    """Schematic accuracy/operational-usefulness trade-off curve.

    Draws two logistic curves, symmetric about x=0.5, so the two
    curves always cross exactly at the midpoint — which lines up
    with the administrative-regions panel (b) directly above it,
    since that panel occupies the middle third of the same
    GridSpec row.

    Args:
        ax: Target axes, expected to span the full figure width.
    """
    x = np.linspace(0.0, 1.0, 200)
    steepness = 9.0
    accuracy = 1.0 / (1.0 + np.exp(-steepness * (x - 0.5)))
    usefulness = 1.0 - accuracy

    ax.plot(
        x, accuracy, color=ACCURACY_COLOR, linewidth=2.5,
        label="Prediction accuracy",
    )
    ax.plot(
        x, usefulness, color=USEFULNESS_COLOR, linewidth=2.5,
        label="Operational usefulness",
    )
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1.2)
    ax.scatter([0.5], [0.5], color="black", zorder=5, s=35)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(
        ["Fine\n(individual ovitraps)",
         "Intermediate\n(administrative regions)",
         "Coarse\n(entire city)"],
        fontsize=9,
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.22),
        ncol=2, frameon=False, fontsize=9,
    )
    ax.text(
        0.5, -0.28,
        "Goal: learn an operational geographic scale that balances "
        "predictive accuracy and geographically targeted intervention.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, fontweight="bold", wrap=True,
    )


def save_skater_panel(
    sectors_learned: gpd.GeoDataFrame,
    bounds: np.ndarray,
    mean_lat: float,
    path: Path,
) -> None:
    """Render and save the SKATER partition as its own figure.

    Args:
        sectors_learned: Sector polygons with a cluster_id column.
        bounds: Shared (minx, miny, maxx, maxy) map extent.
        mean_lat: Mean latitude, for the geographic aspect ratio.
        path: Output image path.
    """
    fig, ax = plt.subplots(figsize=(5, 5.3))
    plot_learned_panel(ax, sectors_learned)
    style_map_panel(
        ax, bounds, mean_lat,
        "Learned operational regions (SKATER)",
        "Data-driven operational partition",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved standalone SKATER panel to %s", path)


def main() -> None:
    """Build and save the composite figure plus the standalone SKATER map."""
    # ── Load data ─────────────────────────────────────────────
    sectors = load_sectors(SECTORS_PATH)
    ovitraps = load_ovitraps(OVITRAPS_PATH)
    sectors_learned = attach_learned_clusters(sectors, CLUSTERS_PATH)
    logger.info("Sectors: %d, ovitraps: %d", len(sectors), len(ovitraps))

    bounds = sectors.total_bounds
    mean_lat = float((bounds[1] + bounds[3]) / 2.0)

    # ── Composite: 3 map panels + trade-off row ──────────────
    fig = plt.figure(figsize=(12, 8.5))
    gs = GridSpec(2, 3, height_ratios=[3, 1.3], hspace=0.55, wspace=0.15)

    ax_fine = fig.add_subplot(gs[0, 0])
    ax_admin = fig.add_subplot(gs[0, 1])
    ax_coarse = fig.add_subplot(gs[0, 2])
    ax_tradeoff = fig.add_subplot(gs[1, :])

    plot_ovitrap_panel(ax_fine, sectors, ovitraps)
    plot_admin_panel(ax_admin, sectors)
    plot_city_panel(ax_coarse, sectors)

    style_map_panel(
        ax_fine, bounds, mean_lat,
        "Fine spatial scale", "One model per ovitrap",
    )
    style_map_panel(
        ax_admin, bounds, mean_lat,
        "Intermediate spatial scale", "Region-specific models",
    )
    style_map_panel(
        ax_coarse, bounds, mean_lat,
        "Coarse spatial scale", "One citywide model",
    )

    plot_tradeoff(ax_tradeoff)

    COMPOSITE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPOSITE_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", COMPOSITE_OUTPUT_PATH)

    # ── Standalone SKATER panel, saved for later use ─────────
    save_skater_panel(sectors_learned, bounds, mean_lat, SKATER_OUTPUT_PATH)


if __name__ == "__main__":
    main()
